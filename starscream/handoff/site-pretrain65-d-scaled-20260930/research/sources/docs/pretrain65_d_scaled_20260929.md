# Scaled D on pretrain65 — 29 September 2026

This is a scratch run of the corrected collection-next D method on the original pretrain65 corpus, with the frozen selection25 suite. It is not a continuation of an ablation checkpoint.

## Recipe

- 218 rounds; 520 scheduled episodes per round (eight per original course); 5,081 optimizer updates per round; batch 1,536. These match v6.21.1 update-fix: 113,360 scheduled episodes and 1,107,658 updates in total.
- D's 2–4 accepted-gate learner segments, beta-zero active control, one expert/DART bootstrap round, and 50% expert-prefix starts are retained. No additional recovery cap or recovery takeover is introduced.
- Corrected 167-dimensional route/plant observations, configured-action previous-action mapping, action weights [1, 1, 1, 0.35], physical action loss 0.75, dynamics loss 0.15, constant learning rate 0.00012, gradient clip 2, batch 1536, and D's architecture are retained.
- Original pretrain65 teacher/planner profile maps and training manifest are restored; mini-slalom augmented courses and pace profiles do not enter training.
- Collection and replay families are the 65 original singleton families with uniform weights. Dynamic sampling remains disabled as in D.
- Replay preserves D's 90% online / 10% permanent mix. Scaling the D retention and capacities by 520/64 gives 203,142 retained online rows per round, online capacity 3,047,119, permanent capacity 423,199, and approximately a 15-round online horizon. This intentionally does not copy update-fix's 65/35 mix or its longer replay horizon.
- Actual retained rows can underfill and actual labels/steps can differ from broad collection. Matching episodes and updates does not claim equal environment exposure or equal updates per collected label.

## Normalization and validation

Normalization is recomputed from 260 expert/DART episodes (four scheduled per training course) using the corrected feature/action contracts. Rows are balanced equally by concrete training course; no selection course contributes statistics or replay. Only statistics initialize the scratch model.

The exact frozen `configs/eval/v6211_vision_selection25.json` suite supplies all 25 selection courses and family weights. Eight paired starts per course (200 episodes) run after every round. The 6000-step horizon, seed base, fixed gate-zero starts, and speed command 16.5 are preserved. Every course receives its frozen deadline from `real100_v2_timed_protocol_v1.json`. Best-three checkpoints use family-weighted timely success, with latest retained separately. Clean/timely, eventual, crashes and per-course outcomes remain in the full event log. These reused selection courses are development validation, not an untouched generalization test.

The trainer evaluates DAgger unconditionally each round; setting the PPO `evaluation_interval` key would not change that cadence. The small integration check exercises every training and selection course, CUDA updates, learner segment collection, checkpoint serialization and an exact replay resume. Regression and configuration audits must pass before the single-run launcher starts.

## Artifacts

- Config: `configs/exp/v6.21.1.1/pretrain65_d_scaled.json`
- Diagnostics: `outputs/diagnostics/pretrain65-d-20260929/`
- Run: `starscream-pretrain65-d-scaled-20260929`
- Launcher: `scripts/audits/launch_pretrain65_d.py`

The launcher checks input hashes and active trainers, freezes the Python runtime and config, refuses output collisions, and records a failure instead of automatically retrying. Full replay remains available for exact resume. It creates no recurring automation.

## Completed preflight and launch

The configuration audit passed. Eighty regression tests passed. The all-course integration run completed two scratch rounds plus a resumed third round, with finite losses and valid selection metrics. All 21 online replay arrays exactly matched reconstruction from the saved shards. Active D collection executed zero teacher-controlled steps. All 65 training and 25 selection geometries were distinct across the split.

Fresh normalization used 260 episodes, accepted 259, and covered every training course. It retained 1,818 rows per course (118,170 total), with 117,172 valid dynamics targets. There were 36 solver failures among 352,127 queries; invalid labels were excluded.

Production launched after validation on 2026-09-29 at approximately 12:01 UTC. Launcher PID 3307 and trainer PID 3308 are launch-time identifiers. W&B run: https://wandb.ai/andrewolvera/starscream/runs/iznvagmt . The live status is `outputs/diagnostics/pretrain65-d-20260929/train.status.json`; the console log is in the same directory. Configuration, normalization, data and code hashes are recorded in `validation.json`, with a frozen runtime and `production-config.json` alongside it.
