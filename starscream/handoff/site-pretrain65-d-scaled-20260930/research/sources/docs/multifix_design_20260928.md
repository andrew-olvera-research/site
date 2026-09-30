# Multi-fix pretraining design: loss, collection, replay (28 Sep 2026)

Follow-up to [objective_bottleneck_diagnosis_20260928.md](objective_bottleneck_diagnosis_20260928.md). Measurements below are on bounded r287, using fresh shard round-00290 (never trained on), unless stated.

## 0. Input bug to fix first (keeps 40 m/s² authority)

The previous-action observation is encoded with the legacy linear CTBR map, which clips collective at 30 m/s². `previous_action_feature_mapping` is unset, so it defaults to `legacy_linear`. Labels and the executed actions use the 0–40 logit-sigmoid contract and are correct.

Verified in the container on a live environment: executed thrusts of 30 and 34 m/s² both produce an encoded feature of 1.0. The contract encoder yields 0.76 and 0.89. In replay, 47% of executed previous actions exceed 30 m/s², so the policy cannot see its own in-flight thrust about half the time. With 11 ms action delay, that thrust is part of the state.

Fix: `"previous_action_feature_mapping": "configured_action_contract"`. This is an observation-contract change, so recompute normalization statistics and train from scratch. Do not fine-tune the old checkpoint across it.

## 1. Loss

What the fresh-state error looks like:

| Dim | Share of action loss | Median abs error | Rows with abs error > 0.5 | Their share of that dim's loss | Large sign-flip rows (share of loss) |
|---|---:|---:|---:|---:|---:|
| collective | 13% | 0.022 | 5.3% | 47% | 2.5% (27%) |
| roll rate | 27% | 0.064 | 13.6% | 62% | 7.7% (41%) |
| pitch rate | 21% | 0.043 | 10.5% | 58% | 5.7% (37%) |
| yaw rate | 39% | 0.089 | 22.1% | 74% | 14.1% (55%) |

The fit is good on typical rows and bad on a heavy tail. The tail comes from labels at the actuator limits: body-rate labels reach 5.7 of 6 rad/s at the 90th percentile, and collective has 5% of labels at each end. The teacher label also moves more than 0.5 away from the previous executed action on 8–23% of rows. Regression is scored on **when** a bang-bang switch happens. A few steps of timing error costs a full-range error, which is why the fit feels "hard". Yaw, the least path-critical axis for thrust direction, takes the largest share of the gradient.

Changes, in priority order:

1. **Per-dimension weights.** Add an `action_dimension_weights` vector, which does not exist yet, applied inside `action_per_sample` and the physical term before the mean. Start from `[1.0, 1.0, 1.0, 0.35]` and renormalize so the total action scale is unchanged. Do not zero yaw, because camera heading matters for the vision student. If heading drift appears, add a small heading-consistency auxiliary instead of raising the yaw weight.
2. **Timing-tolerant targets from the MPCC plan.** Each query already solves a horizon plan; only the first control is fitted. Store the first k planned controls; the infrastructure exists (`action_chunk_target_source: mpcc_open_loop`, `action_chunk_steps`), but `quality_config` currently rejects chunks, so that guard needs lifting. Two options:
   - an auxiliary chunk loss, which gives k targets per state and acts as regularization against memorization;
   - a tolerant first-action loss: `min over d in {0,1,2}` of `Huber(pred_t - plan_{t+d})`, which scores the value of a switch and forgives a timing lag of 1–2 steps.

   Start with the auxiliary only (weight 0.25, k = 8 ≈ 60 ms).
3. **Weight the turn-in window.** Multiply row loss by `1 + a * exp(-max(0, -pre_gate_side - 2) / 2)` for rows before the plane (`pre_gate_side` is already in telemetry). This targets the 2–4 m region where fresh error peaks and misses are decided. Start with a = 1. Normalize the batch mean weight to 1.
4. **Regularization.** Weight decay 1e-5 → 1e-4 (AdamW already), `transformer_dropout` 0 → 0.05–0.1, and optional small Gaussian noise on the normalized history (about 0.02) during training only.
5. **Keep Huber β at 0.05** for this run. Changing the loss shape together with items 1–3 would confound the result. Clipping the heavy tail is also unsafe, because the turn-in decisions live in that tail.

## 2. Collection: wide in space, short in time, learner-flown

Broad collection generalizes because the learner gets labels on its own mistakes. Bounded collection generalizes worse because takeover makes recovery expert-flown behavior cloning, and the corridor stops the learner from reaching the states it will actually visit. Keep the width and bound the duration and the share of retained data:

- **No spatial takeover.** The learner keeps control after a miss or deviation, and the teacher only labels. Remove the `intervention_distance` trigger. Keep `maximum_distance` only as an invalid-state guard, at 5 m or more.
- **Time budget per miss.** After a miss, the learner has `T_rec` (calibrate to the p90 of successful learner recoveries on training courses, likely 3–4 s) to pass the missed gate. If it fails, end the segment and restart from an expert-prefix state at the next gate. Add a no-progress watchdog on the same clock. This removes long loops without narrowing the corridor.
- **Precursors.** Keep every row from the 1 s before each miss, via `precursor_seconds: 1.0`. These are learner states at the decision point, with the expert's alternative as the label; they are the highest-value rows collected.
- **Coherent roll-in.** Per-step β = 0.35 mixing interleaves two controllers and blurs where states come from. Prefer segment roll-in: the teacher flies to a random gate, then the learner flies 2–4 gates. The existing expert-prefix start mode is 50% of episodes; move toward about 70% prefix starts with β = 0 inside learner segments.
- **Courses.** Stream generated, MPCC-qualified courses (the online bank / motif generator) into collection rather than only a fixed training set. Per-gate miss rate is 0.09 on training courses, 0.15 on the same generator's held-out courses, and 0.22 on real100.

## 3. Replay shards: turnover over memorization

Current regime: about 68k retained rows per round out of 0.5–0.7M labels; a 3.95M FIFO (about 58 rounds); about 75 lifetime draws per online row. The permanent pool is 35% of every batch, frozen at round 4, drawn about 460 times per row, at loss 0.022. It is memorized and no longer anchors anything.

- **Retention:** raise `dagger_online_replay_rows_per_round` about 3× (to about 200k), concentrated in the 0–6 m pre-gate window, precursors and per-encounter recovery samples.
- **Window:** shrink `online_replay_capacity` to about 15 rounds of data (≈3M rows, about 3 GB fp16). Lifetime draws per row fall to about 25, or about 15 if `updates_per_round` is also cut to about 3,000.
- **Permanent pool → rolling expert pool:** about 10% of batches, refilled every round from pure-teacher episodes with light DART (existing noise settings), FIFO over about 10 rounds. Implementing refresh needs code; today the pool is only filled in the first `dagger_permanent_expert_rounds` rounds.
- **Recovery rows:** cap by encounter (`balance_corrective_encounters: true`; recovery is already grouped by segment), with a total retained share of about 10–15%. The learner-flown recovery supplies the representations; the cap keeps it from dominating counts.
- **Held-out fresh loss:** before any update, score the just-collected rows. Also hold out about 5% of episodes per round, never trained on. Log both by class, dimension and gate window. This becomes the main fitting metric; stop relying on training loss.

## What needs code vs config

| Change | Status |
|---|---|
| previous-action contract encoding | config (exists) |
| weight decay, dropout, precursor seconds, encounter balancing, retention/capacity/updates | config (exists) |
| per-dimension action weights | small code change in `imitation_loss` |
| gate-window weighting | small code change, telemetry exists |
| chunk auxiliary under the quality path | lift the guard in `quality_config`; replay already stores `action_chunks` |
| learner-flown recovery with time budget (no spatial takeover) | new `TrajectoryGuard` mode |
| rolling permanent pool | new code in the permanent-expert path |
| fresh-shard and held-out episode loss logging | new code, cheap |
| logging teacher controller mode (the feedback fallback is currently invisible; `teacher_modes` is 0 everywhere) | new code, cheap |

## Success criteria on the mini train/val split

Fresh-shard loss below the baseline arms' plateau. Lower per-gate first-attempt miss rate on val (probe `--train-manifest ... --split validation`). Clean/timely SR up without eventual SR falling below the broad arm. Post-miss survival, measured as success conditioned on at least one miss, at least as good as bounded.
