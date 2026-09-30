# Combined fix pilot — implementation and validation

The authorized fix arm is `starscream-v6.21.1.1-mini-slalom-aug2-fixes-r24`.
Its resolved recipe is `configs/exp/v6.21.1.1/mini_slalom_aug2_fixes.yaml`.
It was launched only after both augmented broad and bounded arms completed
round 24 and their launchers recorded successful exits. This is a scratch
pilot with fresh normalization, not a continuation of either warm-start arm.

## Correctness fixes

* **Previous-action input:** enable `configured_action_contract`. Physical
  thrust authority remains 40 m/s²; action labels remain unchanged. The live
  feature regression distinguishes 30, 34 and 40 m/s². Weights and statistics
  with the legacy input mapping are rejected when this mapping is requested.
* **Offline/online consistency:** remapping offline action targets previously
  overwrote previous-action features even when online features still used the
  legacy encoding. Feature replacement now follows the requested feature map.
* **Masked objective gradients:** dynamics and chunk auxiliaries (also the
  topology and flow-dynamics paths) previously displayed a valid-row mean but
  diluted the gradient by the fraction of valid rows. The shared reduction
  preserves valid-row mean gradients and the existing detached imputation for
  group scoring. All-invalid batches have zero loss and zero gradient.
* **Replay quota starvation:** the cached broad sampler and its uncached path
  assigned remainder draws repeatedly to the first families/tracks/gates.
  Small quotas could permanently exclude later groups. Both now use the same
  randomized, unbiased quota rounding and preserve the requested batch size.
  A seeded one-row-batch regression reaches every one of 84 supported
  family/track/gate combinations over 5,000 draws. This is a reproduced
  mechanism, not evidence that every historical batch suffered starvation.
* **Shard integrity:** reject changed data written to an already committed
  shard, missing required arrays, inconsistent row counts, wrong shard
  identity, and nonfinite numeric arrays. Identical committed retries remain
  valid. Legacy optional metadata can still be defaulted. New replay
  fingerprints include previous-action encoding and actual feature/dynamics
  normalization statistics; legacy resumes retain their prior fingerprint
  and separately validate the checkpoint input mapping.

These checks do not establish that historical shard files were corrupt.
Old replay and checkpoints were preserved. Original copies of the main
edited training/replay/sampling files are under
`outputs/diagnostics/multifix-20260928/before/` because the checkout already
contained extensive unrelated changes.

## Configurable fitting changes included in this pilot

These are hypotheses from the design report, distinct from correctness fixes:

| Setting | Baseline augmented broad | Combined fix pilot |
|---|---:|---:|
| Action dimension weights | equal | [1, 1, 1, 0.35], normalized to mean 1 |
| Unique retained online rows/round | 8,334 | 25,002 |
| Online capacity | 260,000 | 375,030, about 15 full retained rounds |
| Online/permanent batch fraction | 65% / 35% | 90% / 10% |
| Updates/round | 626 | 626 |
| Rounds / episodes per round | 24 / 64 | 24 / 64 |
| Initialization | mature plant weights | scratch with fresh expert statistics |
| Learning rate | 3e-5 continuation | 1.2e-4 inherited scratch rate |

Dimension weights apply consistently to normalized, physical, delta and
action-chunk action reductions. The pilot enables only the existing single
action and dynamics heads; Huber beta remains 0.05. With full pools, nominal
average lifetime online draws drop from about 75 to about 35, before
hierarchical reweighting; individual rows can still receive more draws.

The broad collection policy, teacher profiles, start mix, beta schedule,
DART schedule, course corpus and validation suite are retained. The permanent
pool is still the initial expert pool, sampled less heavily. A rolling expert
pool would require fresh teacher episodes and is deferred with the collection
experiments. Recovery encounter caps and wider/time-limited learner recovery
also remain deferred. No new dropout, weight-decay change, gate weighting,
chunk-head objective, or timing-tolerant target is mixed into this pilot.

## Diagnostics and audit boundaries

Each round scores up to 4,096 uniformly selected newly collected rows **before
updates**, using a separate RNG and evaluation mode. Logs include unweighted
and weighted action loss, each action dimension's loss, and physical MAE.
The unweighted metric is essential: downweighting yaw alone can lower a
weighted score without improving prediction. This is pre-update fresh loss,
not a permanently held-out episode set. Existing held-out-course behavioral
evaluation remains the generalization measure; episode-held-out supervised
loss is deferred.

`teacher_modes` is legacy routing metadata, not the internal MPCC controller
mode. It remains unchanged to preserve collection and replay strata. The
new optional `dagger_log_teacher_controller_modes` records internal-mode,
internal-recovery, and solver-fallback query counts plus accepted fallback
label counts under `teacher_controller/`. These are round diagnostics; old
rows cannot be retroactively assigned controller modes. Extended worker
messages use the existing lossless transport fallback.

The delta objective algebraically compares the same prediction error in
different coordinates; it is not a temporal smoothing penalty. Its weight
is zero in this recipe, so no semantic replacement is made here. The quality
path's chunk guard remains: lifting it alone is not proof that every recovery
chunk target is valid. Chunk objectives belong in the later loss ablation.

## Validation evidence

`outputs/diagnostics/multifix-20260928/validation.json` records hashes of the
validated code, tests, config and statistics. Reproduce with:

```sh
python scripts/audits/validate_multifix.py
python scripts/audits/launch_multifix_leg.py
```

* 194 regression tests passed; one historical v6.19 fixture was unavailable
  and explicitly skipped. CUDA tests ran, including eager/captured parameter
  parity with weighted, partially masked objectives.
* Expert-only normalization collection: 36 episodes, 37,316 valid query
  rows; 16,011 balanced feature rows spanning all nine training courses,
  including 15,896 valid dynamics rows. No pretrained actor weights enter the
  statistics artifact.
* Real process-collector smoke: three rounds, 7,020 environment steps, finite
  training losses, real HDF5 persistence, CUDA updates and evaluation.
* Separate checkpoint continuation: restored and completed round four,
  reaching 9,360 steps. Every online replay array was compared exactly against
  the newest 800 rows reconstructed independently from the committed shards,
  including parent/child source chaining and capacity eviction.

The smoke intentionally shortens episodes and is not a behavioral success
test. The full pilot uses the complete training and evaluation horizons.

## Launch and interpretation

`scripts/audits/launch_multifix_leg.py --launch` checks both baseline status
files, zero exit codes, passing validation, artifact hashes and absence of
another racing trainer. It has a process lock, refuses existing run/log
destinations, and writes atomic lifecycle status. Failures never count as
completion and never trigger an automatic duplicate or restart.

Status: `outputs/diagnostics/mini-slalom-recovery-baselines/mini_slalom_aug2_fixes.status.json`.
Console: `outputs/logs/mini_slalom_aug2_fixes.console.log`.
The app heartbeat `supervise-starscream-combined-fix-arm` monitors completion
or meaningful failures every ten minutes and stays quiet on routine progress.

Both preceding arms finished: broad at 1,584,694 steps (last printed full
success .85), bounded at 1,044,272 steps (last printed full success .70).
These are their training-time validation summaries, not final causal rankings.
Compare eventual, clean and timely success together. Different initialization
and learning rate mean the new scratch pilot cannot isolate causal benefit
against those warm-start arms. A matched scratch control or separate fixed-data
loss/replay ablations is required for attribution. No improvement is claimed
until outcomes are measured.

Startup verification: the full pilot has completed round 2 / 24 at 101,360
environment steps. Fresh-loss and internal teacher-mode counters are present
in the full event log; checkpoints and shards are being written. Early scratch
validation is 0% full success and is not evidence of improvement.

## Completed result and follow-up

The fix pilot completed round 24 with exit code 0 at 1,465,405 environment
steps. Final weighted eventual/clean-timely validation success is 35%/25%;
raw counts are 6/20 and 4/20, with 14/20 crashes. Neither external hard
slalom has a final success. Fresh unweighted loss averages 0.2823 in rounds
3–8 and 0.2469 in rounds 19–24, but fresh versus training replay fitting
remains widely separated. The scratch-versus-warm-start limitation prevents
causal ranking against the two completed baselines.

The user subsequently authorized broad and bounded scratch controls. Both
are launched using preserved pre-fix code and fresh shared legacy statistics.
Full characterization, startup repairs, comparison contract and learning
curves are documented in [multifix_results_20260928.md](multifix_results_20260928.md).
