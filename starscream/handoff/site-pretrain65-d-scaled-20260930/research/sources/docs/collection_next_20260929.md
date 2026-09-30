# Collection formulation screen and fresh evaluations — 29 September 2026

The fresh evaluations are complete. The next 12-arm screen uses the corrected
inputs, losses, normalization and replay recipe; it is not a scale-up. Training
remains the nine-course mini-slalom corpus. All new arms use seed 2026092913,
24 rounds, 64 scheduled episodes/round, 626 updates/round and batch size 1536.
The existing expert/DART bootstrap and 50% expert-prefix start distribution stay
fixed. Equal updates/episodes do not imply equal environment steps or labels.

## Fresh evaluation findings

Latest round-24 A, D and fixed-bounded checkpoints were evaluated on paired new
seeds: 20 starts/course on the existing five-course panel, plus one start/course
on the frozen real100 protocol. No teacher intervenes during evaluation. Real100
uses reference-aware misses, the 6000-step horizon, and no shorter retry/dwell
cutoff. Related-panel results use the existing process evaluator and deadlines;
real100 uses the canonical fixed-batch evaluator and its own frozen deadlines.
Do not pool these different panels into a single score.

| Five-course panel, 100 episodes | Fixed broad A | Learner segments D | Fixed bounded |
|---|---:|---:|---:|
| Eventual completion | 20 | 45 | 10 |
| Timely completion | 15 | 14 | 10 |
| Clean/timely completion | 14 | 13 | 10 |
| Completions after reference-aware miss | 6 | 32 | 0 |
| Crashes | 76 | 49 | 90 |
| Mean reference misses | 2.30 | 4.01 | 0.91 |

D succeeds on 32 paired starts where A fails, while A succeeds on seven where D
fails. For timely and clean/timely, the paired discordances are 10 D-only versus
11 A-only. The earlier five-point weighted clean/timely endpoint difference is
not a robust basis for preferring A. D retains a strong completion advantage,
largely through recoveries. On the external hard-slalom portion D completes
6/40, all late, versus A's 0/40. These are five reused course geometries, not
100 independent course samples or replicated training runs.

| Real100, 100 episodes | A | D | Fixed bounded |
|---|---:|---:|---:|
| Eventual | 0 | 2 | 0 |
| Timely / clean-timely | 0 / 0 | 0 / 0 | 0 / 0 |
| Crashes | 99 | 89 | 100 |

D's two completions are late recoveries. Observed post-miss time accounts for
approximately 61% A, 69% D and 46% bounded of elapsed steps on this hard panel.
This includes time before eventual crashes, is defined from geometric miss
events, and is not a latent-mode classifier or replay sampling share. It is
consistent with the concern that difficult courses amplify extended recovery.
No recipe here demonstrates useful broad timely generalization yet.

## Sequential experiment order

| Arm | Difference / question |
|---|---|
| a_repeat | Fixed broad A, second training seed |
| d_repeat | Learner 2–4-gate segments D, second training seed |
| full_learner | Beta-zero full episodes after bootstrap/prefix; isolates segment endings |
| gates_1_2 | Shorter learner segments |
| gates_4_6 | Longer learner segments |
| time_2s | Two seconds of learner control after prefix; bounds time rather than successful passes |
| time_4s | Four-second counterpart |
| full_expert20 | Choose expert or learner once per episode, 20% expert probability after prefix |
| segments_expert20 | Same coherent ownership choice with 2–4 accepted-gate endpoints |
| recovery_4s | Full learner rollout; genuine miss or 6 s no-progress starts 4 s recovery; 50 m hard guard; invalid queries do not stop execution |
| recovery_8s | Same recovery formulation with 8 s allowance |
| d_equal_axes | D collection, equal axis weights rather than yaw downweighting |

The relaxed recovery arms retain only valid expert labels, just like broad
collection. Invalid labels are never invented or stored. They change the old
C/E spatial and invalid-query stop conditions together to test an actual
time-budget formulation; they do not isolate either old guard separately.
Miss clocks cannot reset on recrossings. Prefix/bootstrap steps do not consume
segment time. Segment termination is not a course success. Next episodes use
ordinary scheduled starts, with no teleport or skipped gates.

Coherent expert episodes provide fresh expert supervision through online replay;
they do not refresh the permanent expert pool. Every stored action target remains
the clean counterfactual teacher action, while history uses the executed action.
No new critic, cost-to-go objective or model architecture is introduced.

All arms save one best **family-weighted timely** checkpoint and latest. Final
and late-window comparisons must also report clean/timely, eventual, crashes,
misses, collected steps, retention shortfalls and collection ownership. Do not
select a winner on the best checkpoint's selection-panel score alone.

## Validation, execution and retention

105 regression tests passed. Four representative new mechanisms each completed
three scratch simulator/CUDA rounds and a resume through round four. All 21
online replay arrays matched exact reconstruction at 800 retained rows. Real
telemetry verified time truncation, learner-only execution, coherent expert
execution and learner-controlled recovery. An inherited smoke logging path was
caught and repaired; new fixture events were separated and the historical log
restored. Production log paths are checked for uniqueness.

The launcher requires completed evaluations and passed validation, verifies
frozen runtime/config hashes, refuses existing trainers or output collisions,
and executes one arm at a time. A child failure stops the queue; no automatic
retry, watcher, reminder or recurring automation is created. It checks each
successful child saved round 24 before advancing. Free disk below 10 GiB stops
the queue before the next run.

Cleanup inside the container removed 1,297 files totaling 48.856 GiB from old
runs: 1,274 replay shards and 23 redundant ranked ablation checkpoints. All
pretrain/RL ranked best and latest weights were preserved. Retired replay keeps
the newest three shards plus shards containing permanent expert data. A
`REPLAY_PRUNED.json` marker identifies runs whose exact replay resume is no
longer supported; their weights remain usable for evaluation/initialization.
Active ablations retain full replay until successful completion, then the queue
applies the same last-three-plus-permanent retention rule. Logs, evaluation
records, configs and comparison evidence remain available.

Authoritative files under `outputs/diagnostics/collection-next-20260929/`:

- `eval-summary.json` and `eval-{a,d,bounded}-{related,real100}.json`
- `jobs.json`, `validation.json`, `preflight.json`, `runtime-sha256.json`
- `queue.status.json`, `<arm>.status.json`, `<arm>.console.log`
- `cleanup-old-runs-complete.json` and per-completed-arm cleanup manifests

Entry point: `python scripts/audits/launch_collection_next.py` inside the
existing container. It is single-use: do not relaunch against reserved outputs.

Launch verified: queue PID 41692, first-arm trainer PID 41693. The first arm
is `a_repeat`; W&B initialized run `j7tr411f`. The queue was launched only after
all 600 evaluation episodes and the integration/preflight checks completed.
These PIDs are launch-time evidence; use the status files for current state.

Completion: all 12 arms finished round 24 with exit code 0. The aggregate and
interpretation are in [collection_next_results_20260929.md](collection_next_results_20260929.md).
