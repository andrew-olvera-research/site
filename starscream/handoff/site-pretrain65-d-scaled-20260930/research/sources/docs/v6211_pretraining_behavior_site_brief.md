# Pretraining behavior: evidence for a site description

**Actor:** v6.21.1 update-fix, selected at 121,343,545 collected environment
steps. This is an MPCC-supervised DAgger checkpoint **before any downstream
RL**. It receives privileged true vehicle state, the ordered next six gate
records, a three-step causal history, previous controls, and a speed command.
Its output is one CTBR control command. The training objective also includes
a one-step task-dynamics prediction head (weight 0.15).

## Measured evidence

| Measurement | Result | Scope |
|---|---:|---|
| Fresh real60 geometry | **252/416 completed, 60.6%** | 52 courses, 8 randomized-plant episodes/course; course-bootstrap 95% interval 52.4–68.5% |
| Fresh real100-hard-v2 geometry | **335/736 completed, 45.5%** | 92 courses, 8 randomized-plant episodes/course; course-bootstrap 95% interval 39.4–51.6% |
| Previously exposed public/reference courses | **15/64 completed, 23.4%** | 8 courses shared by both suites; report separately from fresh geometry |
| Route-family examples, fresh real60 | Go-around 97.5%; ordered 3D 85.4%; slalom 81.2%; long-low 18.8% | 5–6 courses/family, 8 episodes/course; performance is uneven |
| Route-family examples, fresh hard-v2 | Compound reversal 85.0%; go-around 83.3%; ordered 3D 77.5%; long-low 6.2% | 3–10 courses/family, 8 episodes/course |
| One-step dynamics readout on three rendered hard-course laps | **pooled R² 0.483**, 4,356 valid steps | Normalized next-task delta, excluding gate-frame transitions; three selected successful laps, so exploratory |

The fresh-geometry intervals use 20,000 bootstrap resamples of courses with
seed 20260923, not individual timesteps; they reflect course variation in these
fixed eight-seed panels. Real100-hard-v2 is
benchmark-informed development geometry, not a protected final test.

Two replayed recoveries show the control pattern directly. On go-around, the
actor missed the last gate's directed aperture, returned to its approach side,
and crossed successfully about 1.2 s later. On long-low-braking, it missed gate
1 twice, returned to the approach side after each miss, and eventually crossed;
it later did the same after missing gate 6. A matched clean replay had no missed
gate crossings. The tracker never advanced on a miss. These are selected
examples, **not a population recovery-rate estimate**. Full state/action traces
and 320-dimensional actor representations are in
`outputs/evals/recovery-mode-probe/`.

At identical terminal states, replacing the repeated final-gate lookahead
with synthetic straight-through gate positions changed the normalized action
by mean L2 distances of 0.67, 0.57, and 0.16 on the three laps. This confirms
that the future route slots affect control. It is an offline input intervention,
not evidence that the synthetic route improves lap time.

## What likely produced it

The policy learned from an MPCC expert in closed loop. DAgger queried that
expert on learner-induced states, while early DART noise and expert-prefix
handoffs widened the states visited. The recipe requested a 40/35/25
nominal/critical/recovery replay split. Those are *requested sampling weights*;
they do not establish that 25% of actual labels came from MPCC recovery. In
three inspected saved replay shards, about half the rows are expert-prefix
occupancy, but none has a separate routed-recovery-teacher tag. The checkpoint
also has no separate recovery teacher configured. The plausible mechanism is
that the **same MPCC supplied corrective actions from off-line and wrong-side
states** as the learner wandered there. An ablation without those labels would
be needed to assign a causal share of the behavior to recovery supervision.

The ordered six-gate lookahead gives the actor a local course program; the
three-step history and privileged state give it velocity, attitude, and recent
control context. The dynamics head's positive out-of-sample R² on these three
courses shows that its latent retains information useful for predicting the
next task-state change. Its closed-loop recoveries show that this information
can support a return to the correct gate approach side. Neither result proves
that the actor contains a discrete “recovery mode,” an explicit viability
model, or a general-purpose state estimator. Here, the input is true state.
As an RL initialization, this is valuable because successful crossings and
some corrective continuations are already inside the policy's action
distribution. RL can optimize pace and consistency while collecting reward on
completed routes; a randomly initialized controller would first have to find
those routes. That is a mechanism hypothesis, not a measured RL sample-efficiency
result for this checkpoint.

## Site-ready copy

> Before reinforcement learning, the MPCC-supervised policy already flies
> complete laps on new gate layouts and recovers from some missed gates. On a
> randomized-plant evaluation, it completed 252 of 416 attempts across 52
> fresh real60 courses and 335 of 736 across 92 fresh hard courses. In captured
> laps, a missed gate remains the active target: the policy turns back to its
> approach side and tries the crossing again. Its learned representation also
> predicts the next local state change on three selected hard-course laps
> (pooled R² 0.48 over 4,356 valid control steps). These results describe the
> policy **before RL**. They motivate using RL to refine pace and consistency
> from a capable closed-loop starting point.

## Claim boundary

This workspace has **no controlled, matched RL-from-random-initialization
comparison** for this checkpoint. The numbers show substantial ability before
RL; they do not prove superior RL sample efficiency or final performance versus
RL from scratch. The recovery examples and dynamics probe are selected
successful laps. State estimation, general viability reasoning, and a distinct
latent recovery mode remain hypotheses for targeted ablations. Use “ordered
route-conditioned control” rather than “proven topological understanding.”

**Sources:** `outputs/evals/v6211-update-fix-final/update-fix-real60-e8.json`,
`update-fix-real100-hard-v2-e8.json`, the corresponding suite YAMLs,
`outputs/evals/recovery-mode-probe/summary.json`, and
`docs/v6211_recovery_route_audit.md`.
