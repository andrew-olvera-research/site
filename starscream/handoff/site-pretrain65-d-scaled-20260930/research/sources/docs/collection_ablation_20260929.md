# Collection ablation B–E, 29 September 2026

Arm A is the completed `mini_slalom_aug2_fixes` scratch pilot. These four arms
keep its corrected inputs, objective, normalization statistics, teacher profiles,
course split, model, seeds, replay recipe, and 24-round / 15,024-update budget.
Only collection control and run-specific output/build paths change. The configs
are `configs/exp/v6.21.1.1/mini_slalom_aug2_collection_{b,c,d,e}.yaml`.

| Arm | Execution after bootstrap/prefix | Recovery | Segment endpoint |
|---|---|---|---|
| B | Original per-step teacher beta | Calibrated teacher takeover | Original full episode |
| C | Original beta outside recovery; learner during recovery | Genuine miss or gate no-progress starts a 4 s learner recovery budget | Recovery failure or original endpoint |
| D | Learner only | Original broad behavior, no added recovery/distance guard | Random 2–4 actual gate passes after handoff |
| E | Learner only | Same learner recovery rule as C | Same 2–4-gate limit as D, or recovery failure |

All retain the first expert/DART bootstrap round and the 50% expert-prefix start
distribution. Controls are inactive during beta=1 collections, including expert
coverage retries, and during the forced prefix itself. D/E switch to beta=0
when the original schedule leaves expert-only collection. DART is unchanged;
there is no new recurring DART refresh or expert pool refresh.

B uses the existing training-only per-course calibration: approximately
1.66 m intervention / 3.0 s no-pass trigger / 3.76 s recovery on 10-gate slaloms,
and 1.5 m / 3.12 s / 5.24 s on 4-gate slaloms. Genuine misses and the teacher's
internal recovery signal also trigger B. Recovery requires uninterrupted clean
teacher execution, two actual gate passes (or course completion), and return
inside the intervention distance. Failed recovery terminates the episode.

C/E have no spatial takeover. A genuine missed crossing or six seconds without
an accepted gate pass starts learner-controlled recovery, with four seconds to
pass the current gate. The deadline cannot reset on repeated plane crossings.
A pass exactly at the deadline succeeds. Five meters from the ordered reference
is a terminal invalid-state guard, not a takeover threshold. Invalid teacher
queries terminate B/C/E active collection; their invalid labels are not stored.
The four-second budget is a prespecified pilot setting, not a claimed empirical
90th percentile of successful learner recoveries.

A truncated segment ends the current episode. The next ordinary scheduled
episode starts with the unchanged canonical/expert-prefix distribution. We do
not teleport the tracker to a subsequent gate, count skipped gates, or turn
segment completion into a full-course success. This keeps restart coverage
comparable while isolating the collection changes.

## Implementation and geometry

`starscream/dagger_collection.py` implements an opt-in state machine separate
from `TrajectoryGuard` and its quality-based replay admission. No old quality
quotas or retrospective rejection of failed recovery rows are enabled. All
valid teacher targets are eligible for the same broad replay sampler as A.
Teacher labels remain the clean counterfactual action; the previous-action
feature and teacher delay state use the action actually executed.

The process worker classifies crossings against the **pre-step active gate**.
The existing `planned_plane_crossing` helper requires proximity to an ordered
incoming reference segment that actually crosses outside the aperture. A
wrong-side position, a backward crossing, or the internal teacher recovery flag
alone does not become a geometric miss in C/E. Only plant gate acceptance
counts toward segment completion. Tests include rotated gate frames, distant
and wrong-order reference segments, wrong-side starts, and planned crossings.

The worker always receives a real learner action for these methods: lazy
teacher-only dispatch is disabled internally when collection controls are
present. Otherwise a C/E recovery handoff could execute a zero placeholder
from a teacher-drawn step. This is a dispatch-cost change, not a change to the
teacher labels or update budget.

The 18-field trajectory telemetry is stored for every valid label and survives
async handoff, shard retention, eviction and resume. It is diagnostic only for
these arms. The replay contract records the collection configuration so a resume
cannot silently change methods. `train/collection/*` reports execution ownership,
recovery completion/failure, timeouts, segment truncation, reference-aware misses,
invalid-state stops, and separate valid versus retained class counts. Ordinary
fresh weighted/unweighted action loss and physical errors remain enabled.

## Validation and execution evidence

Run `python scripts/audits/validate_collection_ablation.py` to validate the
prepared simulator evidence, run the regression suite, compare the configs to
A, and freeze the runtime. The machine-readable record is
`outputs/diagnostics/collection-ablation-20260929/validation.json`.

Each method has a three-round scratch simulator/CUDA smoke run followed by
resumes through rounds 4 and 5. Every online replay array is compared exactly
after restoring the newest 800 rows. Separate D/E probes start from the completed
fix policy to exercise actual successful segment truncation; they are validation
fixtures, not ablation results or initialization for the four scratch arms.

Two validation-fixture configuration mistakes were caught before launching any
experiment: zero permanent capacity violated the resume API's positive-capacity
requirement, and the handoff probe initially specified both a warm checkpoint
and scratch-only normalization initialization. The corrected fixtures use an
empty pool with capacity one and checkpoint-owned statistics, respectively.
The failed fixture logs are preserved. Production B–E retain A's original
positive permanent capacity and scratch initialization.

The one-shot launcher `scripts/audits/launch_collection_ablation.py` reserves all
four outputs, verifies hashes and absence of another trainer, then runs B/C
concurrently. D/E start only after both B/C exit successfully and save round-24
checkpoints. Each arm has an independent MPCC build directory. Child exit waits
implement the requested sequential queue; there is no recurring monitoring,
notification automation, restart loop or additional arm.

Authoritative status files are
`outputs/diagnostics/collection-ablation-20260929/{b,c,d,e}.status.json`, with
`queue.status.json` for the pair sequence. Logs are
`outputs/logs/mini_slalom_aug2_collection_{b,c,d,e}.console.log`.
Runs execute the frozen `runtime/` copy under the diagnostic directory.

Compare raw and family-weighted eventual/clean/timely outcomes, first-attempt
misses, per-axis fresh errors and late-window trends. Report actual environment
steps because shorter segments change collection volume at equal updates and
episode counts. Keep the completed A run as reference; this single-seed screen
does not identify statistical significance or replace a held-out-course and
multi-seed confirmation before scaling.

## Launch record

Validation passed: 207 tests passed; one historical v6.19 fixture was unavailable.
All four scratch smoke runs resumed through round 5; all 21 online replay arrays
were restored and compared exactly at 800 retained rows per arm. Actual D/E
segment-completion probes recorded 9 and 7 segment endings, respectively.
B/C launched together with in-container trainer PIDs 16048 / 16047; D/E are
reserved and queued. The one-shot queue PID is 16044. See per-arm status files
for current state; these PIDs are launch-time evidence. No automation was created.

## Completed results: A against B–E

All four queue arms completed round 24 with exit code 0. A is the previously
completed scratch fixes arm. Each used 24 rounds, 64 scheduled episodes per
round and 626 optimizer updates per round (15,024 updates). The collection
methods changed how many plant steps and valid labels those episodes produced.
All comparisons below use the same five-course, 20-episode fixed validation
panel; family weighting gives the 10-gate augmentation 40%, the pair of 4-gate
augmentations 40%, and the two external hard slaloms 20%.

| Round-24 result | A: broad mix | B: teacher takeover | C: learner recovery | D: learner segments | E: segments + recovery |
|---|---:|---:|---:|---:|---:|
| Collected steps | 1,465,405 | 1,061,367 | 1,007,201 | 1,661,931 | 628,655 |
| Valid teacher labels across 24 rounds | 1,439,375 | 1,060,689 | 1,006,397 | 1,517,272 | 627,316 |
| Retained online labels across 24 rounds | 600,048 | 600,048 | 600,048 | 600,048 | 568,292 |
| Raw eventual completion | 6/20 | 2/20 | 2/20 | **10/20** | 2/20 |
| Raw clean/timely completion | **4/20** | 2/20 | 2/20 | 3/20 | 2/20 |
| Raw crashes | 14/20 | 18/20 | 18/20 | **10/20** | 18/20 |
| Family-weighted eventual | 35% | 10% | 10% | **60%** | 10% |
| Family-weighted clean/timely | **25%** | 10% | 10% | 20% | 10% |
| Successful laps with recovery | 2/20 | 0/20 | 0/20 | **7/20** | 0/20 |
| Mean reference misses per episode | 1.90 | 1.00 | 1.20 | 2.55 | 1.15 |

The final difference between A and D is four more completed laps for D but one
fewer clean/timely lap. Among completed laps, 4/6 A laps were clean/timely,
versus 3/10 D laps. D's higher reference-miss count and seven recovered
completions fit a policy that survives more learner errors but has not learned
to avoid them. Lower miss counts in B/C/E accompany earlier termination and
should not be read as better first-attempt flight. Every arm scored 0/4 on
each external hard slalom at the final checkpoint.

| Rounds 19–24 average | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| Raw eventual | 22.5% | 8.3% | 12.5% | **42.5%** | 5.0% |
| Raw clean/timely | **13.3%** | 8.3% | 12.5% | 5.8% | 3.3% |
| Family-weighted eventual | 24.2% | 8.3% | 13.3% | **47.5%** | 5.0% |
| Family-weighted clean/timely | **15.0%** | 8.3% | 13.3% | 7.5% | 3.3% |

These averages reuse the same 20 evaluation episodes each round; they are
trend summaries, not 120 independent trials. D's completion gain and A's
clean/timely advantage both appear in the late window. D peaked at 70%
weighted eventual completion in round 23, but clean/timely completion was zero
at that checkpoint. Selection by eventual success alone would therefore hide
the main tradeoff.

### What collection actually produced

Across active rounds 2–24, B executed about 322k teacher steps after the prefix
and qualified 186 teacher recoveries, but also failed 279. C logged **zero**
completed learner recoveries against 319 failed; E logged one completed against
687 failed. Neither C nor E logged a recovery deadline timeout. Invalid-state
stops were 243 for C and 609 for E. The four-second clock is not the observed
limiter in these runs. Ground contact, invalid teacher queries or the 5 m
reference guard can end recovery first; the present counters do not partition
all failure causes cleanly enough to assign each failure to one.

D logged 409 completed 2–4-gate collection segments and no active teacher
steps after prefix handoff. Its retained online labels were about 42% nominal
and 58% corrective under the collection-only 0.75 m reference threshold; this
classification is absent from A and does not itself prove poor label quality.
E retained only 568,292 online labels versus the 600,048 target: 11 active
rounds could not fill the 25,002-row quota. Its 629k collected steps are only
38% of D's, despite equal scheduled episodes and updates. This is a material
exposure confound for D-versus-E. B/C produced roughly one million steps each,
about 30% fewer than A.

Fresh action loss also illustrates the change in state distribution. Final
unweighted loss was 0.266 A, 0.203 B, 0.222 C, 0.293 D and 0.265 E. D had
the best completion despite the highest fresh error; B's lower fresh error did
not translate into flight success. D's late-window thrust physical MAE was
4.99 m/s² against A's 3.74; roll/pitch/yaw were 1.88/1.87/2.28 rad/s against
A's 1.68/1.67/2.06. Fresh rows are generated by different policies, so these
figures measure fitting on each policy's own changing state distribution, not
a shared held-out imitation test.

### Collection direction

The evidence favors **preserving learner control** for eventual completion,
while adding support near the expert trajectory to improve first-attempt and
timely laps. D is the only arm with a meaningful completion gain; A remains
the clean/timely reference. B's spatial teacher takeover and C/E's hard
learner-recovery termination should not become the default from these pilots.
In particular, simply lengthening C/E's four-second clock is a weak next test
because that clock never fired.

The most informative next comparison is A, D, and a beta-zero **full rollout**
with no 2–4-gate truncation. That would separate the effect of coherent learner
control from segment endings, which A-versus-D changes together. Then try D
with one near-expert anchor changed at a time: a larger expert-prefix start
share, shorter 1–2-gate segments, or occasional coherent expert-flown episodes.
Measure first-attempt misses and clean/timely success as primary criteria, with
eventual completion and retained-row volume beside them. A sampler/retention
change to favor near-expert rows is a separate later ablation. Use additional
seeds and a larger untouched validation panel before scaling any winner.

Reproduce from immutable local logs with
`python scripts/audits/analyze_collection_ablation.py`. The per-round metrics,
collection totals and source paths are in
`outputs/diagnostics/collection-ablation-20260929/results/summary.json`; the
three-panel figure is `results/learning-curves.png`.
