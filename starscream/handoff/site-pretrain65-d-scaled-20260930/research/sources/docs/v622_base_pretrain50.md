# v6.22 base pretrain50

Status: relaunched from scratch on 2026-09-18 with the restored v6.21 optimal
DAgger recipe and measured 1.5M/course budget. W&B run:
https://wandb.ai/andrewolvera/starscream/runs/h06yn7z3 . The superseded
curriculum run was stopped at round 64 and its checkpoint directory was deleted.

The experiment is `starscream-v6.22-base-pretrain50` in
`configs/exp/v6.22/pretraining_behavior50_dagger.yaml`. It trains from scratch
on the frozen 50-course pretrain set, selects checkpoints on all 60 real60
courses, and evaluates all 100 real100-hard courses every ten rounds for
reporting only.

## MPCC pace frontier

The CPU search tested a common high-authority controller envelope on a 2 m/s
grid up to 36 m/s. Each candidate used isolated solver state; only command
probes sharing the identical controller/planner envelope reused a backend.
Increases required 2/3 usable nominal and 5/6 usable randomized episodes with
solver failure <=6% and recovery <=10%. The stronger DART cohort determines
perturbation eligibility but cannot reject an otherwise valid pace frontier.

One course (`v622_pretrain50_technical_vertical_01_11`) cleared 12 -> 14 m/s;
its next rung failed. Final per-course commands are 11 courses at 8 m/s, 10 at
12 m/s, 1 at 14 m/s, 24 at 16.5 m/s, and 4 at 24 m/s.
Thirty-two courses pass the stronger perturbed cohort; 18 are explicitly
DART-exempt. Full evidence and the tuned manifest are under
`outputs/v622-pretraining50/mpcc-frontier-v3/`.

## DAgger recipe

The training method is the winning v6.21 scratch-5M recipe. Teacher beta falls
from 1.0 to .35 over 20 rounds. Replay remains 65% online and 35% permanent
expert, with nominal/critical/recovery strata .40/.35/.25. The first four
rounds build successful coverage and permanent expert replay and use DART on
eligible courses. Half of collection uses expert-prefix targeted starts. The
18 courses that failed the stronger perturbed teacher cohort remain explicitly
DART-exempt in both collector implementations; they still contribute nominal
expert and DAgger data.

Each course uses its single qualified MPCC command from the frozen manifest,
and the actor observes that command. There is no additional command-mixture,
recovery anneal, or pure-expert phase. The earlier v6.22 run changed all three
axes at once and therefore was not a clean test of the proven method.

The lower mean rollout speed in this recipe is acceptable pretraining behavior.
DAgger deliberately learns learner-induced deviations, braking, expert-prefix
handoffs, and recovery states rather than optimizing lap time. Those samples
broaden the representation used for zero-shot course generalization. The
v6.21 ablation advantage is much larger than the evidence for removing them;
the downstream RL stage owns the explicit speed and elapsed-time objective.

The budget is 75 rounds x 400 episodes. Rounds 8–12 of the stopped run, the
window actually operating near the restored beta .35, collected 4,990,719
transitions over 2,000 episodes. At that measured rate, 75 rounds gives
approximately 1.497M raw transitions per course (74.86M total), matching the
requested ~1.5M starting budget.
Updates, model,
losses, transition starts, expert-prefix behavior,
dynamics randomization, and physical-action objective remain the v6.21 recipe.

## Review and launch

Validation uses 240 fixed-seed episodes (four per real60 course) each round;
reporting uses 200 (two per real100 course) every ten rounds and on the final
round. These provide repeated seeds without dropping any course.
Per-course rates and time tails remain noisy at these counts. Compare success
and successful mean/median/p90 steps on matched courses and commands; pooled
time dispersion across different course lengths is not a measure of control
consistency. Checkpoint selection remains success-first.
Both held-out panels explicitly use their own admitted speed commands. Merely
enabling manifest speed conditioning previously fell back to the stage speed
because held-out courses are absent from the training manifest.

The training timeout is 10,000 steps (76.9 s): the longest admitted nominal lap
is 36.47 s, already longer than the inherited 3,000-step cap (23.1 s). Twenty
percent headroom requires about 5,689 steps. Seven admitted courses exceeded
the old cap even at their qualified command. Validation/reporting use
the qualification protocol's 6,000-step cap. Completed or crashed episodes
still terminate immediately; a cap is not a fixed rollout length.

The stopped run does not isolate recovery replay as the cause of slow times:
teacher beta, replay composition, replay retention, and command weights all
changed together. Its clearest failure occurs when the pure-expert phase
evicts historical learner states. The corrected experiment returns to the
v6.21 control recipe and leaves speed optimization to RL.

The launcher explicitly applies `dagger_local_16c_v625.yaml`: native exact
MPCC paths, compact collection observations, asynchronous lazy collection,
packed transport, host CUDA graphs, cached replay sampling, compiled backbone,
deferred update metrics, persistent grouped evaluation and native vector
stepping. Prefetch and packed update transfer remain off per measured results.
Preflight hashes cover the config, frozen manifest, runtime profile and fitted
normalization artifact.

The inherited dynamic family sampler is disabled: pretrain50 lacks its required
`source_family`/canonical correspondence, and real60 supplies no matching
training-course competence curves. Its original configuration failed at
trainer startup. Static balanced family/course collection, hierarchical replay,
and transition-start coverage remain enabled. Reintroducing adaptive weights
requires a defined training competence panel and measured evaluation overhead.

Focused schedule/replay/throughput tests and native MPCC/CUDA graph tests pass.
The corrected normalization bootstrap accepted 99/100 episodes, covered every
course and fit 90,550 equally weighted rows (1,811 per course). The isolated
GPU smoke completed collection, updates, evaluation and checkpointing.
Launch only with
`scripts/launchers/current/run_v622_base_pretrain50.sh`; it rechecks the config
and manifest hashes, split sizes, schedule, and absence of an existing run.
