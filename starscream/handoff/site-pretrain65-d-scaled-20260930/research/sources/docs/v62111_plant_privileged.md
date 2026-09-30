# v6.21.1.1: explicit plant-conditioned privileged teacher

This experiment starts from scratch with new replay. It preserves the v6.21.1
update-fix seed, 65 training courses, course geometry, dynamics distribution,
DART/beta schedules, 218 rounds, 520 episodes/round, 5,081 updates,
batch 1,536, 35% permanent/65% online replay, capacities, LR schedule, losses,
130 Hz action contract, three-step history and six route records. It adds the
plant descriptor and uses the qualified async/event pipeline with 64 validation
environments on eight CPU workers. Execution timing and actor lag therefore
prevent bit-identical trajectories. The revised experiment also replaces the
old validation/sampling feedback with the student selection25 real100-v2 suite,
as requested. Per-course MPCC pace and fresh DART qualification are documented
in `docs/v62111_train65_teacher_tuning.md`. This is no longer an isolated
plant-input-only comparison.

The launch recipe now uses the approved teacher frontier under
`outputs/diagnostics/v62111-train65-teacher-v4/frozen/approved/`. All 65
training teachers passed fresh nominal, randomized, and DART qualification.
Fifty-five have at least 1% faster randomized median laps; the equal-course
mean reduction is 9.57%. Three initially selected fast profiles regressed on a
separate 2x DART noise probe; the approved frontier uses two qualified faster
alternatives and one original qualified teacher. Original stress outcomes and
the approval decision remain local in `frozen/recovery-stress/` and
`frozen/approved/recovery-stress/`. Course comparison and controlled sample
cost measurements are in `frozen/approved/`.

## Observation and normalization

Contract: `starscream_route_plant_v1`, 167 values per history row. The first 103
values retain the original observation and frozen normalization exactly. The
64-value suffix is normalized in fixed physical units by `plant_privileged.py`;
its checkpoint normalizer is identity. Values are not clipped or normalized by
each sample's RMS, which would discard useful absolute magnitudes. Zero remains
physical zero, nominal plants remain representable, and out-of-range values stay
visible. The ordered schema and unit scales are saved with the experiment and
every model checkpoint.

| Part | Values | Meaning |
|---|---:|---|
| Applied rigid body / motors | 15 | Mass, arm length, principal inertia, motor speed bounds, time constant, three thrust-polynomial coefficients, kappa, per-axis rate limits |
| Aerodynamics | 30 | Linear/quadratic/rotor/angular drag, mean wind, gust amplitudes/frequencies, initial gust phase as sine/cosine, center-of-mass offset |
| Timing and actuator limits | 4 | Requested actuator delay, control timestep, applied per-motor thrust bounds |
| Causal wind/frame context | 15 | Current gust phase sine/cosine, current body-frame wind, first two columns of world-from-body rotation |

The 49 constants are computed from the **applied** environment parameters once
per reset. The remaining 15 values are refreshed at the current command time.
Gust constants alone omit the phase clock; world-frame wind alone is ambiguous
to a gate-relative actor. The extra context resolves both without future state,
future expert actions, seeds, or course identifiers. Motor speeds and previous
commands already appear in the original state/control inputs.

Characteristic scales keep normal inputs near order one: mass 0.73 kg, arm
0.17 m, inertia from that nominal geometry, motor bounds 150/3,000, tau 0.1 ms,
rates 6 rad/s, delays 20 ms, dt 10 ms. Thrust coefficients are scaled by their
contribution at motor speed 3,000 to a 25 N reference, avoiding division by a
possibly zero polynomial coefficient. Aero scales follow the established
randomization envelope; angles use sine/cosine. See the saved JSON for all units.

State-estimation and flight-plan randomization are disabled in this recipe.
Camera randomization is irrelevant to this truth-state actor. These are not
silently substituted with estimated values. A future experiment enabling noisy
state/route input needs its own observability audit. Actuator delay is exposed,
but this does not turn three-step history into an arbitrarily long action queue
or guarantee a perfectly Markov observation under new delay ranges.

## Model

Exactly one settings token, taken from the latest history row, is prepended:
`settings, state/control x 3, route x 6, action`. The causal sequence grows from
13 to 14 tokens. The projector follows the working state projector: linear
SwiGLU input and output projections, width 320, no input RMS normalization or
Fourier expansion. All original trainable parameters receive exactly the same
scratch initialization for the same seed; the added modules use an isolated RNG.

A zero-initialized 320-to-640 FiLM projection supplies scale/shift to the ordinary
tokens. The prefix also provides a direct attention path from initialization.
The original four transformer blocks and `modern_adaln_zero=false` remain
unchanged. If adaptive blocks are enabled in a separate ablation, their existing
conditioning sum includes the settings embedding. Parameters increase from
5,488,987 to 5,839,067 (about 6.4%).

Adaptive feature modulation is motivated by conditional-transformer work such
as [DiT](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf).
Its diffusion results do not establish the optimal conditioning scheme for
drone control; retaining the working backbone makes this a bounded first test.

## Collection, replay, and evaluation coherence

Both compact collection observations and validation observations contain the
same applied descriptor. It travels in the existing numeric history tensor,
so row selection, teacher labels, previous commands, auxiliary targets, replay
shards, and the permanent/online sampler retain their existing alignment. The
learner uses only the latest settings row for the single prefix. Replay remains
FP16 for history storage and converts through the existing BF16 update path;
physical-unit scaling occurs before FP16 storage, including tiny motor constants.

This first version intentionally uses the existing fixed-shape replay primitive,
not a new episode-ID join. At full capacity, the 64-value suffix costs about
2.02 GiB extra resident history storage. HDF5 compression benefits from repeated
constants; memory maps and the OS cache can use the completed 24 GiB WSL memory
configuration described below. Episode-table deduplication remains a possible
memory optimization, but is not required for correct collation.

The first four expert/coverage rounds and their transition remain synchronous.
Thereafter, one frozen pre-update actor collects the next window during current
updates and validation. Evaluation always reads the current post-update learner
while its optimizer is paused, never the lagged collection actor. It evaluates
the frozen `configs/eval/v6211_vision_selection25.json`: 25 courses, four
episodes each, every round. Speed is 16.5 m/s, one complete course from gate zero;
seeds match student distillation (2034091462 + SHA256(slot) offset + repeat).
The 16.5 m/s value is the same fixed policy conditioning command on every
selection course. It is not a claim that 16.5 m/s is time optimal on every
course, nor is it an MPCC action label. The policy still imitates each training
teacher at that training course's qualified command. A fixed held-out command
keeps checkpoint ranking comparable across rounds and avoids importing noisy
MPCC admission commands as policy calibration. The matched policy-command probe
in `docs/heldout_speed_command_probe.md` found negligible aggregate gain from
per-course qualified commands, while a pushed-command full suite reduced RL2
completion. Faster-command panels can be evaluated separately without changing
this frozen checkpoint selector.
Top-five checkpoint ranking uses `selection_suite_success`: average success
within each family, weighted by that family's fraction of the full real100-v2
population. Raw unweighted success is also logged. Checkpoints and
rollback decisions occur after that evaluation. Full per-course results are
computed regardless of the dashboard filter.

Adaptive sampling now learns priorities in all 15 real100 behavior families.
The frozen `real100-behavior-map.json` derives soft training-course affinities
from shared requirement cells, weighting rare cells and balancing cell kinds.
Those affinities project bounded, EMA-smoothed family priorities onto the
existing 65 training shards. Original shard identities, base sampling mass,
35/65 replay split and admission schedule remain intact. Replay losses remain
a secondary priority signal, projected from the existing source groups.

Gate-start and replay gate weights use observed conditional failure on matching
training requirements. Ordered chains use survival through the whole chain.
Cold-spawn gate zero, unobserved transitions, and final cyclic outgoing chords
do not become invented failure evidence. Unobserved gates retain a uniform
positive prior; observed rates use Jeffreys smoothing and an EMA. Numerical
gate indices never transfer between different layouts. This is a geometric
transfer heuristic, not a measured guarantee of skill transfer. With four
episodes per course, individual results are noisy; bounds, smoothing, and
coverage floors remain important. There is no separate old-course sampler probe.
The old periodic reporting suite is also disabled for this experiment.

## Logging and reproducibility

W&B receives compact scalar allowlists: aggregate validation success/crash,
selected survival points, action/dynamics losses, replay sizes, collection wait,
update/evaluation times, memory/swap, sampler entropy/range and pipeline versions.
Per-course and detailed family metrics are **local only**, in
`outputs/logs/starscream-v6.21.1.1-plant-selection25-dagger.full.events.jsonl` and ordinary
evaluation/checkpoint artifacts. The smaller `.events.jsonl` mirrors W&B.
Two-second device/memory telemetry is in `.system.jsonl`.

Recipe: `configs/exp/v6.21.1.1/plant_dagger.yaml`.
Launcher: `scripts/launchers/current/run_v62111_plant.sh`.
Schema, frozen normalization and parent recipe diff:
`outputs/course-pools/v62111-plant/`.
The launcher refuses another active trainer or reuse of an existing run, checks
for active teacher tuning, and verifies the qualified launch certificate at
`outputs/diagnostics/v62111-train65-teacher-v4/frozen/approved/launch-readiness.json`.
The approved teacher manifest, per-course comparison and selected validation
command labels are saved beside the certificate.

Validation includes exact shared-weight initialization, native parameter/reset
and compact/full observation parity, latest-only prefix selection, CUDA graph
parity, BF16 gradients, local-only detailed logging, and the existing replay,
sampling, update and async regression suites. The initial broad suite passed
111 tests with one missing historical fixture skipped. The isolated smoke uses
shortened warmup and update budgets to exercise async transitions; its metrics
are not a quality comparison. Its first revision correctly hit the DART schedule
guard after an inconsistent shortened warmup; the second revision fixes the
test schedule without changing the production recipe.

The corrected smoke completed 269,220 environment steps and 96 captured
optimizer updates over three windows. Losses were 1.0476, 0.7293 and 0.4465;
all three small validation batches had zero completion, so this is a functional
test only. Saved-shard inspection verified 167-wide FP16 histories, identical
episode constants throughout each sampled history, and varying plants across
episodes. Window 3's frozen actor exactly matched the round-1 checkpoint; final
evaluation/model versions agreed and no pending work remained. Detailed results
are in `outputs/dagger-throughput/plant-smoke-audit.json`.

`outputs/dagger-throughput/plant-reference-v1/` freezes real plant-aware replay
inputs, actor weights/outputs and 100 hierarchical sampling draws in weighted
and unweighted modes. Capture and exact verification passed. The original
`async-reference-v1` fixture also still passes unchanged.

The full experiment launched detached in the existing Docker container at
2026-09-24 00:33:40 UTC (September 23, 20:33:40 EDT):
[W&B run jdsjhdbc](https://wandb.ai/andrewolvera/starscream/runs/jdsjhdbc).
Startup evaluation completed and round 1 collection began. It was stopped at
00:38:00 UTC on user request, before any training window committed. Only round
zero was checkpointed. Status and console
are in `outputs/logs/starscream-v6.21.1.1-plant-dagger.status` and `.log`.
No production v6.21.1 checkpoint was overwritten. At the time of that stopped
run, no WSL restart or memory-limit change had yet been performed. The replacement
run namespace is
`starscream-v6.21.1.1-plant-selection25-dagger`. The subsequent WSL memory
change and restart are complete. The replacement was launched fresh on
2026-09-24 at 05:54 UTC as [W&B run ucprnrg0](https://wandb.ai/andrewolvera/starscream/runs/ucprnrg0),
without resuming the old validation baseline.

The revised suite passed 72 focused regression tests (one unavailable historical
fixture skipped), and the final teacher configuration passed 49 focused tests
(one unavailable historical fixture skipped). A native 100-episode evaluation
of the undertrained three-window smoke policy was rerun after the approved
teacher manifest and completed in 11.95 seconds including worker startup: 36,532
steps,
25 courses, 15 feedback families, 65 weighted training shards. Its success was
zero; this validates wiring and gives no mature-policy evaluation speed or quality
claim. All 25 commands matched the 16.5 m/s course labels, and 2,031 inference
batches checked the active speed target, normalized 167-value histories and
stable episode plant constants. Inputs, episode outputs, all metrics and sampler
state are saved in
`outputs/dagger-throughput/selection25-native-audit.json`.

An in-progress checkpoint at 63.13M steps was checked on 2,048 recent replay
states from all 65 training courses. Swapping static plant constants within
each course changed mean absolute action output by 0.278, establishing that the
model uses the settings token. A small closed-loop, paired 25-course check at
two episodes per course scored 29.5% weighted success with correct plant
constants and 23.75% with plausible constants from other episodes; mean gates
fell from 4.96 to 4.04. The difference is only three completed laps in this
50-episode probe, so it is directional evidence of useful conditioning rather
than a precision estimate. Reproducible scripts and results are in
`scripts/audits/probe_v62111_settings_shuffle.py`,
`scripts/audits/probe_v62111_settings_closed_loop.py`, and
`outputs/diagnostics/v62111-settings-*.json`.

## WSL memory change and restart handoff

The Windows host has 32 GiB RAM. The user created
`C:\Users\drewo\.wslconfig` and restarted the machine/WSL after a separate eigenpy
build exhausted the old memory/swap budget. The existing `starscream` container
was restarted successfully and has no Docker memory cap; no image rebuild was
needed. Verified configuration:

```ini
[wsl2]
memory=24GB
swap=8GB
```

This leaves roughly 8 GiB for Windows. Swap is a fallback, not a throughput gain.
After restart, Linux reported about 22 GiB available and zero swap use. With
eight tuning workers running it still reported about 18 GiB available, zero swap
use, and zero sampled swap-in/out. Windows vmmem working set alone is not a measure
of unavailable Linux memory: inspect `MemAvailable` and active paging as well.
Verify with `wsl -d Ubuntu -- docker exec starscream free -h` and `vmstat 1 2`.
Full replay residency still needs telemetry from the larger production run.
See Microsoft's [WSL configuration documentation](https://learn.microsoft.com/windows/wsl/wsl-config).

The replay implementation has no separate 16 GiB cap to raise. Larger WSL memory
lets existing allocations and file-backed replay/handoff pages stay resident;
Linux uses additional room for page cache automatically. Keep replay capacity
and the bounded 2 GiB async handoff unchanged for this experiment. Existing memory,
swap-I/O and pressure telemetry will show whether paging actually falls.

Cleanup found no remaining trainer/actor/evaluation workers or DAgger temporary
handoff directories. Container `OOMKilled` is false, cgroup OOM counters are zero,
and available Ubuntu kernel logs contained no OOM kill. The trainer's exit code
1 came from the requested interrupt, not evidence of OOM. The unrelated Megatron
image build was left untouched and subsequently completed successfully:
`megatron/sim:isaacsim5.1-lab2.3.2-cu128 Built`.
That cleanup describes the earlier stopped run. The subsequent WSL memory
change/restart was performed by the user, as recorded above.
