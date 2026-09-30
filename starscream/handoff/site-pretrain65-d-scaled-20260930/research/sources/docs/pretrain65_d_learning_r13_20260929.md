# Scaled D: early learning analysis through round 13

Snapshot: 29 September 2026, completed round 13, 7,746,879 training environment steps. The live run continues unchanged. Source snapshot: `outputs/diagnostics/pretrain65-d-20260929/learning-analysis-snapshot.json`.

## Learning efficiency

The closest historical comparator is the scratch plant/selection25 run: same 65 training courses, selection25 geometries, 520 scheduled episodes/round, 5,081 updates/round and batch 1,536.

| First family-weighted selection25 completion threshold | Scaled D | Earlier plant/selection25 |
| --- | --- | --- |
| 40% | Round 8; 5.30M steps | Round 116; 83.76M steps |
| 50% | Round 10; 6.37M steps | Round 151; 108.70M steps |
| Latest D level, 57.19% | Round 13; 7.75M steps | First reached/exceeded at round 208; 149.38M steps |

The 50% crossing uses approximately 15.1x fewer optimizer updates and 17.1x fewer recorded environment steps. At matched round 13, the earlier run was at 0%. At its nearest step count (round 12, 7.60M), it was also at 0%. D's round-13 completion is near that run's final round-218 level (57.5% weighted).

These are descriptive first crossings from single training runs, not replicated causal estimates. D evaluates eight starts/course and a 6000-step horizon; the old run used four starts/course and 10000 steps. Seed base and geometries match, but the episode panels are not identical. Multiple recipe changes are bundled: corrected previous-action inputs/losses, collection ownership and endpoints, replay fractions/retention, bootstrap length, and learning-rate schedule. This establishes a major recipe-level efficiency gain, not an isolated D-segmentation effect. Counts exclude normalization, smoke and evaluation work. The update-fix run used a different validation panel and should not be used for a direct success-rate ratio.

## Quality gap and emerging trend

| Round | Eventual | Timely | Clean | Clean + timely | Mean reference misses/episode |
| --- | --- | --- | --- | --- | --- |
| 8 | 45.19% | 5.94% | 1.00% | 0.38% | 3.180 |
| 10 | 57.00% | 11.50% | 6.69% | 3.00% | 2.705 |
| 12 | 56.00% | 15.25% | 10.06% | 5.56% | 2.320 |
| 13 | 57.19% | 14.81% | 11.56% | 4.56% | 2.205 |

Success columns are family-weighted; misses are the unweighted episode mean. Eventual completion is already locally flat over rounds 10–13 (54.06–57.19%), but that short window does not establish a terminal plateau. Clean completion is rising and misses fell 30.7% since round 8. Timely and clean/timely results remain noisy. The clean/timely-to-eventual weighted ratio is only 8.0% at round 13.

Unweighted round-13 outcomes: 112/200 completed, 30/200 timely, 21/200 clean, and 9/200 clean + timely. Among the 112 completions:

- 9 were clean + timely.
- 12 were clean but late.
- 21 were timely after a miss.
- 70 were late after a miss.

Thus 91/112 completions (81.3%) involved recovery after a reference-aware miss. Both accuracy and speed matter; faster recovery alone cannot make a run clean.

Course examples distinguish these problems. Slalom slots 010 and 019 complete 8/8 and 6/8, respectively, but neither has a clean or timely finish. Long-braking slot 077 completes 7/8, including 6/8 clean finishes, yet none is timely. Clean/timely success is confined to six of 25 courses. The two public-reference courses remain 0/8 eventual.

## Collection health and implications

At round 13, all 65 courses remain active, online replay contains 2,640,846 rows, and the round retained the full 203,142-row target from 454,180 valid labels. Permanent replay is full at 423,199 rows. Label validity is 99.33%. Active learner collection has no teacher-controlled steps; teacher execution is 35.3% overall because of expert prefixes. 391/520 scheduled episodes reached their segment endpoint. Available host memory is about 10.8 GiB, with zero recorded swap-in/out rate.

Zero recovery-tagged retained rows does not imply zero recovery experience: D's recovery-control mode is disabled, so its nominal/corrective tags do not measure all post-miss behavior. Teacher internal recovery was reported on about 9.9% of queries in this round.

Interpretation: D is rapidly acquiring a useful completion policy and is beginning to reduce misses while completion is locally flat. That supports allowing more pretraining before deciding on midtraining, but there is no scheduled objective transition. The unchanged imitation objective and constant learning rate do not automatically start optimizing clean/timely after eventual success saturates. Checkpoint selection currently rewards timely success, which still permits misses.

A sensible decision point is a sustained 5–10-checkpoint plateau in clean/timely and clean/eventual conversion, rather than eventual completion alone. If misses keep falling and clean success rises, continued pretraining is still improving the substrate. If only recovery completion improves, a separate midtraining objective or collection intervention should target first-pass gate execution and deadline performance. Preserve the current run as the pretraining control before evaluating that change.

The original plant log lacks the reference-aware clean/timely metrics, so this snapshot cannot establish that its old clean/timely ratio was numerically identical. That requires reevaluation of old checkpoints under the current frozen protocol.
