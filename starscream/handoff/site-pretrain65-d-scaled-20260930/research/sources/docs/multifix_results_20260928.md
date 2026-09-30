# Completed pilot and scratch baseline comparison

All three scratch pilots have completed 24 rounds. The combined fixes achieve
better observed flight outcomes than both scratch controls; see the final matched
comparison below. The two old baselines started from a mature plant policy, so
their higher absolute performance cannot measure the fix package's effect.

## Completed experiments

All use the same five-course validation panel, four fixed-seed episodes per
course. The selection score weights the 10-gate augmentation family 40%, the
4-gate augmentation family 40%, and the hard external slaloms 20%. Thus it
differs from the raw success count across 20 episodes.

| Final round 24 | Broad, warm | Bounded, warm | Fixes, scratch |
|---|---:|---:|---:|
| Environment steps | 1,584,694 | 1,044,272 | 1,465,405 |
| Weighted eventual success | 92.5% | 77.5% | 35.0% |
| Weighted timely success | 47.5% | 50.0% | 25.0% |
| Weighted clean/timely success | 45.0% | 50.0% | 25.0% |
| Raw eventual successes | 17/20 | 14/20 | 6/20 |
| Raw clean/timely successes | 8/20 | 9/20 | 4/20 |
| Raw crashes | 2/20 | 6/20 | 14/20 |
| Mean reference misses per episode | 2.2 | 1.5 | 1.9 |
| Successes involving recovery | 9/20 | 5/20 | 2/20 |

The warm policies began at 85% weighted eventual success and 50% weighted
clean/timely success. Broad ends with more eventual completion but slightly
less clean/timely success than that shared start. Bounded preserves the initial
clean/timely point estimate while losing eventual completion. This is consistent
with the existing reliability versus clean-flight tradeoff, not a demonstrated
overall improvement from bounded collection.

Across rounds 19–24, weighted eventual/clean-timely averages are 87.1%/40.0%
for warm broad and 79.6%/45.0% for warm bounded. The bounded final endpoint
looks worse than its late average on eventual completion. These are smoothing
summaries of the same fixed panel, not 120 independent trials or confidence
intervals. Do not select the best round and call it a replicated gain.

The scratch fix arm first completes a validation episode at round 13, and
reaches its best eventual score at rounds 23–24. Its late six-round averages
are 24.2% weighted eventual and 15.0% clean/timely success. The late improvement
means this short scratch run is still developing; it does not authorize an
extension or prove that more training will close the gap.

| Final per-course raw successes | Broad, warm | Bounded, warm | Fixes, scratch |
|---|---:|---:|---:|
| Held-out 10-gate augmentation | 4/4 | 3/4 | 1/4 |
| Held-out 4-gate augmentation A | 4/4 | 4/4 | 2/4 |
| Held-out 4-gate augmentation B | 4/4 | 4/4 | 3/4 |
| External hard slalom 010 | 2/4 | 1/4 | 0/4 |
| External hard slalom 019 | 3/4 | 2/4 | 0/4 |

The fix arm's current capability is concentrated in the related augmentation
family. Both external hard courses still fail every final episode.

## What the fitting measurements say

The old warm runs did not log pre-update fresh-row loss. Comparing their
training losses directly with the new fresh loss would mix different data,
sample weighting, optimization time and initialization.

For the fix arm, compare rounds 3–8 to 19–24, after beta reaches .35:

| Fresh pre-update metric | Rounds 3–8 mean | Rounds 19–24 mean |
|---|---:|---:|
| Unweighted action Huber | 0.2823 | 0.2469 |
| Weighted action Huber | 0.2678 | 0.2327 |
| Thrust MAE, m/s² | 4.453 | 3.745 |
| Roll-rate MAE, rad/s | 1.887 | 1.676 |
| Pitch-rate MAE, rad/s | 1.902 | 1.670 |
| Yaw-rate MAE, rad/s | 2.283 | 2.060 |

Unweighted fresh error falls 12.5%, so the trend is not merely the arithmetic
effect of reducing yaw's weight. Physical error improves on every axis.
However, fresh states change as the learner changes; these are not frozen-data
measurements and cannot attribute the improvement to a particular fix.

At round 24, fresh unweighted/weighted losses are 0.2661/0.2520, while averaged
weighted replay training action loss is 0.0515. The roughly 4.9x weighted gap
shows that fitting sampled replay is much easier than fitting the next
collection. It does not alone prove memorization: fresh rows are uniform and
scored before updates; training rows are hierarchically sampled, older, and
scored during updates. The gap remains a central diagnostic for the new
scratch controls, which now log fresh loss too.

Replay turnover works as configured: the fix arm retains 25,002 rows each
round and ends with 375,030 online rows, i.e. 15 retained rounds. Warm baselines
retain 8,334 per round and end with 200,016 rows, retaining all 24 rounds so far.
The nominal fully populated lifetime draw estimate changes from about 75 to
35 draws/online row, but hierarchical sampling makes individual reuse uneven.
The expert pool remains frozen and is sampled at 10% rather than 35%; rolling
expert refresh is still deferred. Correctness and persistence tests pass;
behavioral benefit remains to be measured.

## Scratch control setup (completed)

Configs: `mini_slalom_aug2_scratch_broad.yaml` and
`mini_slalom_aug2_scratch_bounded.yaml`, under `configs/exp/v6.21.1.1/`.
Both use random initialization, seed 2026092112, learning rate 1.2e-4,
24 rounds × 64 episodes, 626 updates/round and batch size 1536. The nine training
courses, teacher profiles, start/beta schedules and validation contract match
the fix arm. No checkpoint weights or replay are reused.

Both share newly collected legacy-input statistics from 36 expert-only episodes
with the same bootstrap seed/procedure as the fix pilot. The broad control
retains the old input encoding, equal-axis objective and old replay recipe.
Bounded adds the same previously defined bounded collection/admission/replay
package. That package changes more than just spatial exploration.

The controls execute a frozen copy of the pre-fix trainer, shard store and
sampler under `outputs/diagnostics/multifix-scratch-controls-20260928/baseline-runtime/`.
Only read-only fresh-loss diagnostics were added to that trainer. The rest
of the runtime is copied and hashed. This prevents current corrected code
from leaking into a supposedly pre-fix control. The comparison contract and
hashes are saved in that directory's parent as `comparison-contract.json`
and `runtime-sha256.json`.

Two startup issues were repaired without discarding completed training rounds:
the snapshot needed an explicit `/workspace/.env` path for W&B, and simultaneous
MPCC compilation collided in a shared build directory. Broad now has its own
compiler directory. Failed-start logs/status and its round-zero-only artifacts
are preserved under `failed-start-missing-env/` and `failed-broad-shared-build/`.
Bounded continued while broad restarted. Both subsequently run concurrently;
compare rounds/updates and collected steps, not wall time.

The authoritative per-arm statuses are in
`outputs/diagnostics/mini-slalom-recovery-baselines/mini_slalom_aug2_scratch_{broad,bounded}.status.json`.
The repair supervisor reconciles the pair status when both finish. The original
pair coordinator can temporarily retain the historical broad startup failure.

The useful next comparison is scratch broad versus scratch fixes at equal
round/update budgets, with environment steps reported alongside. That tests
the combined package, not which component caused a change. Scratch bounded
versus scratch broad characterizes the older bounded package from a matched
start. These remain single-seed pilots on a small repeated validation panel.

Reproduction: `python scripts/audits/analyze_multifix_completed.py`.
Machine-readable results and the learning-curve PNG are in
`outputs/diagnostics/multifix-20260928/results/`.


## Final matched scratch comparison

Both scratch controls completed with exit code 0. Latest checkpoints independently confirm round 24 and the environment-step totals below. All three scratch arms received 15,024 optimizer updates (24 × 626), batch size 1,536, and 64 collection episodes per round. Automatic monitoring remains paused at the user’s request.

The combined package has the strongest observed flight outcomes in this matched scratch pilot. This supersedes the earlier pending-control assessment; it is encouraging evidence for the package, not statistical significance or attribution to individual fixes.

| Final round 24 | Broad scratch | Bounded scratch | Fixes scratch |
|---|---:|---:|---:|
| Collected steps | 1,501,041 | 966,773 | 1,465,405 |
| Raw eventual success | 1/20 | 2/20 | 6/20 |
| Raw timely success | 0/20 | 2/20 | 4/20 |
| Raw clean/timely success | 0/20 | 2/20 | 4/20 |
| Raw crashes | 19/20 | 18/20 | 14/20 |
| Family-weighted eventual | 5.0% | 10.0% | 35.0% |
| Family-weighted timely | 0.0% | 10.0% | 25.0% |
| Family-weighted clean/timely | 0.0% | 10.0% | 25.0% |

Rounds 19–24 averages preserve the ordering, rather than relying only on the last checkpoint:

| Late-window validation | Broad | Bounded | Fixes |
|---|---:|---:|---:|
| full_course_success | 4.17% | 8.33% | 22.50% |
| timely_success | 2.50% | 8.33% | 15.00% |
| clean_timely_success | 2.50% | 8.33% | 13.33% |
| selection_suite_success | 4.17% | 8.33% | 24.17% |
| selection_suite_timely_success | 2.50% | 8.33% | 16.67% |
| selection_suite_clean_timely_success | 2.50% | 8.33% | 15.00% |
| crash_rate | 95.83% | 91.67% | 76.67% |

Fixes finish with five more completions and four more clean/timely flights than broad, with 2.4% fewer collected steps. Bounded collects 35.6% fewer steps than broad, so this comparison matches updates and collection episodes, not environment transitions. Bounded first succeeds at round 8, broad at 12, fixes at 13: the fix package improves later outcomes without an earlier first success. All three fail both external hard courses at the final checkpoint. Fix successes across the three augmentation courses are 1/4, 2/4, 3/4; broad 0/4, 0/4, 1/4; bounded 0/4, 2/4, 0/4.

### Fresh fitting and physical errors

Fresh diagnostics sample 4,096 newly collected rows before the round’s updates. They use the same calculation, but each arm visits its own state distribution. Bounded’s lower fresh error does not imply better closed-loop flying. Weighted losses additionally use different axis weights across old and fixed objectives; prioritize the unweighted comparison.

| Final fresh metric | Broad | Bounded | Fixes |
|---|---:|---:|---:|
| action_loss_unweighted | 0.2727 | 0.2291 | 0.2661 |
| action_loss_weighted | 0.2727 | 0.2291 | 0.2520 |
| thrust_physical_mae | 4.3975 | 3.6356 | 3.9424 |
| roll_physical_mae | 1.8422 | 1.5535 | 1.8364 |
| pitch_physical_mae | 1.8371 | 1.5791 | 1.7971 |
| yaw_physical_mae | 2.2040 | 1.8847 | 2.1731 |

Physical MAE units are m/s² for thrust and rad/s for body rates.

| Fresh metric: rounds 3–8 → 19–24 | Broad | Bounded | Fixes |
|---|---:|---:|---:|
| action_loss_unweighted | 0.2961 → 0.2693 | 0.2544 → 0.2256 | 0.2823 → 0.2469 |
| action_loss_weighted | 0.2961 → 0.2693 | 0.2544 → 0.2256 | 0.2678 → 0.2327 |
| thrust_physical_mae | 4.7040 → 4.2422 | 4.0438 → 3.4803 | 4.4527 → 3.7447 |
| roll_physical_mae | 1.9986 → 1.8100 | 1.7230 → 1.5638 | 1.8870 → 1.6759 |
| pitch_physical_mae | 1.9830 → 1.8196 | 1.7042 → 1.5123 | 1.9019 → 1.6701 |
| yaw_physical_mae | 2.4033 → 2.2062 | 2.0945 → 1.9014 | 2.2830 → 2.0598 |

Fixes have 8.3% lower late-window unweighted fresh error than broad; the final-round difference is only 2.4%. Within-arm early-to-late reductions are 9.0% broad, 11.3% bounded and 12.5% fixes. All axes improve in each arm. These changing-state measurements support learning, but do not isolate a component or establish held-out generalization.

### Replay exposure and interpretation

| Final replay metric | Broad | Bounded | Fixes |
|---|---:|---:|---:|
| action_loss | 0.0298 | 0.0270 | 0.0515 |
| physical_action_loss | 0.0700 | 0.0656 | 0.1149 |
| online_replay | 200,016 | 200,016 | 375,030 |
| permanent_expert_replay | 52,086 | 37,379 | 52,086 |
| new_online_replay_labels | 8,334 | 8,334 | 25,002 |

Final within-arm fresh-weighted / training-action loss ratios are 9.2× broad, 8.5× bounded and 4.9× fixes. The smaller fix gap is consistent with less emphasis on repeatedly fitting a small old pool, but different objectives, state distributions, hierarchical sampling and pre-update versus during-update timing prevent diagnosing memorization from this ratio. The fix arm’s higher replay loss alone is not evidence of worse learning.

Both controls retain all 24 rounds (200,016 online rows); fixes retain the newest 15 rounds (375,030 rows), admitting three times as many rows per round. Online/permanent sample fractions are 65%/35% for controls and 90%/10% for fixes. Across the common 23,076,864 sampled training rows, nominal online draws are about 15.00 million versus 20.77 million; these are configured exposure estimates, not measured unique-row counts. Permanent pools remain frozen. The earlier approximate lifetime reuse figures (75 versus 35) assume fully populated pools and are not empirical exposure measurements for these finite pilots.

The practical result is a promising combined-package improvement over the matching broad control, with substantial remaining failures (70% final crashes and no hard-course completions). Keep warm-start runs contextual. One seed and the same repeated 20-episode panel cannot establish significance, robustness across seeds, or which of input encoding, objective corrections, weighting and replay changes produced the difference. No additional training arms were launched.

Reproduce all five summaries with `python scripts/audits/analyze_multifix_completed.py`. `outputs/diagnostics/multifix-20260928/results/summary.json` includes per-round evaluations, training metrics and scratch fresh windows; `scratch-learning-curves.png` shows the matched scratch curves.
