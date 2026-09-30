# v6.22 independent championship and challenge benchmarks

## Decision and scope

The generated portions of **real60** and **real100-hard** are independent.
Each retains eight explicitly shared public-reference adaptations: Swift,
CDRA, A2RL, and UTT03/04/05/06/08. The intended composition is 8+52 and 8+92.
The existing real50 remains unchanged for historical comparisons. These names
do not mean that every course is a reconstructed real venue.

## Validated local release — 2026-09-17

Both suites are frozen and load through the production suite loader with all
declared geometry fingerprints verified:

- `configs/eval/v6_22_real60.yaml`: 52 fresh generated courses + 8 shared references.
- `configs/eval/v6_22_real100_hard.yaml`: 92 independent fresh courses + the same 8 references.
- Adjacent `.manifest.json` files preserve provenance, admission contracts,
  accepted teacher settings, cell counts and the evaluation/exposure policies.
- `configs/exp/v6.22/behavior_requirements.json` exports 2,345 geometric cells
  from hard100 only, plus separately labeled executed-behavior cells.

All 152 distinct courses passed MPCC admission. Final release checks passed
static/semantic geometry, historical and cross-suite anti-clone limits,
handedness targets, reference-path arena clearance and serialization hashes.
The expanded candidate arena audit checked 420 eligible variants with zero
remaining clearance failures. The benchmark/generator/online-bank tests passed
54/54. No actor benchmark evaluations have been launched; orchestration is
handed off separately at the user's request.

Final **generated-only** geometry, recomputed after qualification:

| Measure | real60 (52) | real100-hard (92) |
|---|---:|---:|
| Distinct requirement cells | 1,278 | 2,151 |
| Cells with ≥3 independent course witnesses | 157 | 265 |
| ≥25 m approach to a ≤1.2 m center-height gate | 5 | 24 |
| ≥25 m approach with ≥120° chord turn | 3 | 14 |
| ≥2 m vertical change with ≥90° chord turn | 32 | 53 |
| Wrong-side approach transitions | 12 | 25 |

Counts include cyclic geometry and are not independent episode counts. The
hard suite expands the tails and compositions; this does not establish that
every marginal coordinate or every policy success rate is harder. Teacher
feasibility is validated, while model difficulty awaits frozen-checkpoint
evaluation. Source redistribution terms remain to be reviewed before external
publication; this is a validated local benchmark release, not a licensing claim.

`scripts/audits/build_v622_benchmarks.py` generates and qualifies candidates;
`scripts/audits/analyze_v622_benchmarks.py` measures their requirements.
Artifacts live in `outputs/course-pools/v622-benchmarks/`. A candidate ledger
is not a released benchmark. `scripts/audits/publish_v622_benchmarks.py` is the
release gate: it joins the independent-geometry selection with initial and
extended teacher evidence, rechecks geometry/leakage, and refuses to publish
final suites while qualification slots are missing. Use this publisher rather
than the initial builder's prototype `report --freeze` path.

Validation is under the existing headless Flightmare dynamics and directed
aperture-crossing contract, including ground-contact termination. It is not
physical flight validation or swept-rotor/gate-frame collision certification.
Public-track adaptations retain their source uncertainties; no official
leaderboard timing equivalence is claimed by these suites.

No policy training, reward edits, or changes to the active online sampler are
part of this benchmark build. All geometry and MPCC selection use course
requirements and teacher feasibility, not actor failures or success scores.

## What the v6.21 evidence supports

Sources inspected:

- `docs/v621_eval_set.md` and its suite builder;
- `docs/v621_a2rl_utt_support_diagnosis.md`;
- `docs/v622_pretrain_acceptance_plan.md`;
- `docs/v621_online_bank.md`, `starscream/ppo_online_bank.py`, and the v6.21 generator;
- `docs/v621_midtrain_expert_speed.md` and the teacher/speed-conditioning code;
- `docs/v621_rl73_posttraining_review.md` and the current `.7.4` configuration.

The retrospective associates newly occupied cells with transfer improvement:
26 courses with new support improved 9.98 points on average; 24 without it
improved 0.00. Nearest-chain distance alone correlated poorly with improvement.
This supports coverage and multiple independent witnesses as design tools;
it does not establish an optimal discretization or causal transfer law.

The old selection deliberately gave aperture and gate-center height zero
weight and collapsed incoming distances above 12 m. That is inappropriate for
the observed low-gate, 28–42 m UTT failures. The old generator also stopped at
10 gates and did not target A2RL's descending/descending/climbing sequence.

The online bank targets cells with 25–85% mean course competence and fewer
than three active witnesses. It retires solved generated courses to a floor,
rehearses anchors under drift, and screens proposals with nominal MPCC. This
is useful for reliability, but course success attributed to every cell is an
imprecise local competence estimate. The coarse cell vocabulary also cannot
target the regimes it does not describe. Speed is not an explicit cell target.

Speed conditioning is wired through collection and optimization. The reported
weakness is one command per course, with only 16.5/17/18 m/s used: course and
command were confounded. Multiple commands on each course, including a cap
below the curvature plateau, provide a meaningful contrast. A larger command
does not by itself create a faster teacher trajectory.

Recovery-induced generalization and recovery-induced pace dispersion remain
working hypotheses. The expert-speed midtraining experiment does not by itself
establish an optimal recovery schedule. Similarly, the current evidence does
not establish that one reward bug explains the pace problem: the mild time-cost
ablation improved completion while the tested speed-tracking variants did not
establish a better recipe.

## Requirement coordinates

`starscream/env/racing_manifold/benchmark_v22.py` adds a versioned vocabulary
without silently changing old checkpoints or bank cells:

| Cell group | Coordinates |
|---|---|
| Approach | incoming length <5 / 5–12 / 12–20 / 20–30 / 30–40 / ≥40 m; 30° turn bins; center height ≤0.8 / ≤1.2 / ≤2.5 / >2.5 m; smaller aperture dimension ≤1.2 / ≤1.65 / ≤2.2 / >2.2 m |
| Vertical | signed altitude-change magnitude <0.75 / 0.75–2 / 2–3 / ≥3 m; turn; gate height |
| Entry | wrong-side approach, explicit reverse-entry flag, gate-plane incidence, turn |
| Ordered chains | horizons 2, 3 and 4; signed turns, signed vertical magnitude, wrong-side approach |

These are factorized joint cells, not a full Cartesian product whose cells are
nearly all singletons. Witness counts are per independent course, not per gate.
Longer chains have lower selection weights. Coverage gain has diminishing
returns with witness count; selection does not maximize singletons alone.
No weight depends on training deficits or historical policy failures.

Center-chord turn angles are geometry proxies, not executed turn radius or
braking demand. Teacher traces separately measure entry/exit/minimum/peak
speed, braking p90 and heading change, joined to physical gate phase. This
distinction matters especially for wrong-side entries. Virtual UTT route
checkpoints must be identified separately from physical apertures when
interpreting these distributions. Cyclic geometry includes the closure;
canonical single-lap actor evaluation alone does not validate that closure.
The benchmark admission prefix mode flies the canonical prefix and then its
remaining suffix; it does not certify an additional lap. In those saved prefix
traces, `phase` is already the canonical gate index. Cold-reset trace phases
instead need the cold start-gate offset. Do not add the prefix start twice when
joining executed behavior to geometry. The nominal behavior audit starts at
gate zero and is unaffected by that distinction.

## Generation and balance

Nine strata receive near-equal course counts: flow, slalom, hairpin chain,
diving hairpin, long/low, long braking, ordered 3D, stacked reversal and
opposite-side go-around. This is a declared balanced design, not a claim to
have sampled an unbiased empirical championship population: there is no
sufficiently large surveyed population here from which to estimate one.

The ordinary generated courses use 5–12 gates; hard courses allow up to 18.
The hard long-approach proposals extend to 46 m and include narrower apertures
and stronger vertical/turn combinations. The shared Fury reference has 19
route checkpoints, of which many are virtual; gate counts must not conflate
physical obstacles with route bookkeeping. The benchmark is bounded by
raceable championship-like geometry, not arbitrary adversarial obstacles.

Each slot initially receives 60 deterministic proposals and reserves three
ranked alternatives. Geometry is independent of public-reference coordinates.
Historical train/validation/report courses, all real50 geometry and saved bank
YAMLs are protected. The first generation scanned 866 distinct protected
geometries. Same-count cyclic, reflection-, yaw-, translation- and scale-aware
clone distance must be at least 0.12. Zero distance is explicitly rejected;
the old builder's `distance or 9` fallback incorrectly bypassed exact clones.

The vectorized clone calculation is tested against the existing implementation.
An additional 64-point arc-length resampling check compares whole shapes across
gate counts, also removing cyclic phase, reflection, yaw, translation and scale.
Distances below 0.08 trigger review; the final independent portfolio requires
at least 0.06 against historical geometry and all reserved alternatives across
slots. The first pass found admissible alternatives for 141/144 slots; the
remaining three received nine new independent candidates. Local motif overlap
is intentional and is not eliminated by a whole-course anti-clone criterion.

Initial rank-zero static audit (before MPCC selection): pretraining has zero
≥25 m low-gate arrivals, generated real60 has 9, and generated real100-hard has
24. The generated sets contain 1,378 and 2,272 requirement cells respectively;
155 and 272 cells have at least three course witnesses. Those counts are
descriptive, not a guarantee of adequate transfer support. Qualification may
change which alternatives are selected, so final counts must be recomputed.

## Teacher validation and release gate

The existing 130 Hz dynamics/teacher pipeline is reused with a 6,000-step
episode limit. The first search compares 16.5 m/s baseline, a 24 m/s higher
acceleration/braking profile, and 12/8 m/s controlled profiles. Physical thrust
authority remains within the actor contract. Profiles are ranked by measured
usable lap time, not their nominal command.

Two canonical nominal episodes screen each profile. **Benchmark admission uses
the established r6 pipeline:** two canonical nominal, two canonical randomized,
two canonical DART episodes, and one randomized expert-prefix episode from
every physical gate. Every episode must finish, have no failed prefix, ≤6%
solver failures and ≤10% recovery steps. Slower screened profiles are tried
if the fastest fails. `qualify_v622_benchmark_protocol.py` pins this contract.

The first exploratory validation additionally required randomized and DART
success from every cold gate reset. That is a useful stronger stress test, but
it was stronger than the existing pipeline and biased admission toward cold
reset robustness. It remains separate diagnostic evidence. Complete passes
under that stronger contract are explicitly identified and reused as stronger
evidence; failures do not bypass the canonical/prefix admission checks. Raw
trajectories, failed trials, seeds and per-profile results remain available;
a nominal screen is never relabeled full qualification.

Configuration, resolved teacher settings and relevant source hashes identify
the qualification contract; geometry fingerprints identify the courses.
The initial four-profile search is bounded, not a proof of globally optimal
MPCC labels. The extended public-reference search tests twelve joint profiles:
three acceleration/braking envelopes, two racing-line aperture fractions, and
10/20 m/s caps. Full confirmation proceeds in measured nominal lap-time order
with fallback to the next candidate. Geometry, plant, reset noise and acceptance
thresholds remain fixed.

A2RL exhausted the initial profiles. A separately hashed refinement searched
4/6/8 m/s commands, 8/12/18 m/s² lateral envelopes and three aperture fractions.
The 6 m/s, 8 m/s², center-aperture profile passed every benchmark cohort,
including all 12 randomized prefix starts, with zero solver failures and
at most 1.39% recovery steps. All eight public references now have admission
evidence. This conservative A2RL witness proves pipeline feasibility, not a
maximum-speed frontier; label-speed optimization remains a separate task.

Public-reference geometry is immutable. Virtual route apertures extending
below ground and repeated crossings of the same obstacle need semantic
review, not automatic deletion or edits to official physical coordinates.
Their prior source provenance and reconstruction uncertainty must be retained.
The semantic review found no below-ground *physical* apertures in the eight
references. The generic Fury flags came from virtual planning windows and
intentional repeated obstacle crossings.

### Confirmed arena-boundary artifact

High Voltage's baseline reference path reaches y=37.88 and -9.89 m while its
old arena stops at 36 and -8 m. The failed flight gets clamped at the boundary
and falls. CDRA also has a reference-path excursion outside its original box;
A2RL lacks the required buffer. Versioned `_arena1` files expand only the
simulation envelope, preserving every physical gate position and orientation.
Old real50 files and prior evidence remain unchanged. Fresh validation after
this correction made CDRA and High Voltage pass even the stronger all-start
contract. This is evidence of an environment artifact, not behavioral-support
failure on those teacher trials.

The frozen v6.21 actor did **not** gain completion in the corresponding
32-episode matched-seed randomized comparison: CDRA changed 6/32 → 5/32,
A2RL remained 0/32, and High Voltage remained 0/32. This small cohort does
not establish a policy regression, but it does rule out claiming that the
bounds correction solved the observed policy-transfer problem in these runs.
The ablation verifies identical loaded physical gate arrays between each
old/new pair. Raw metrics: `arena-policy-ablation.json` in the benchmark output
directory. Teacher feasibility and learned behavioral support remain separate
questions.

The initial path-containment audit identified 46 of 405 reserved candidates
(three public references, 43 generated alternatives) needing expansion or a
larger buffer. The correction contains the planned path with a six-metre
planar maneuver margin. Corrected fingerprints require fresh qualification.

The family audit also found near-unidirectional flow generation. The portfolio
now fixes alternating dominant turn directions per family/slot before teacher
admission. Each candidate has only one eligible orientation; reflecting one
does not increase its independent-witness count. Both reflected coordinates
and proper gate frames are saved, bounds are reflected consistently, and the
new fingerprint receives new teacher evidence.

## Reproduction and artifacts

Run Python commands inside the existing `starscream` container at `/workspace`.
The initial proposal command requires a fresh output directory; it deliberately
refuses to overwrite an existing candidate ledger.

```sh
python scripts/audits/build_v622_benchmarks.py propose --proposals 60 --alternates 3
python scripts/audits/build_v622_benchmarks.py qualify --workers 12
python scripts/audits/select_v622_independent_candidates.py
python scripts/audits/repair_v622_independence.py
python scripts/audits/balance_v622_handedness.py
python scripts/audits/audit_v622_arena.py --repair
python scripts/audits/qualify_v622_benchmark_protocol.py --workers 12
python scripts/audits/refine_v622_teacher.py --course a2rl_s2_2026_source_consistent_v2_arena1 --workers 2
# Only after existing alternatives are exhausted, append unresolved-slot candidates:
python scripts/audits/extend_v622_candidates.py --per-slot 3 --round 1
python scripts/audits/qualify_v622_benchmark_protocol.py --workers 12
python scripts/audits/publish_v622_benchmarks.py
python scripts/audits/report_v622_admitted_behavior.py
python scripts/audits/export_v622_behavior_requirements.py
```

The geometry preference plan must be fixed before the full generated-course
qualification starts. Qualified slots are retained; only missing slots advance
to another independent candidate. If alternatives are exhausted, the progress
ledger remains explicitly incomplete and publication fails closed.

`distribution-audit.json` contains geometric distributions and witness counts.
`executed-behavior-audit.json` contains trace-measured speed/braking and
speed/radius cells, using both the common baseline and tuned teachers. Its
scope is successful nominal rank-zero proposals, clearly distinct from final
selection or randomized reliability. `geometry-review.json` retains the
different-count similarity flags and public-reference semantic review.
`admitted-behavior-audit.json` instead follows the accepted profile and its
trace provenance for each currently admitted slot. It reports completeness,
course-level cell witnesses and transition-weighted quantiles; the latter are
descriptive teacher-dependent behavior, not optimal physical requirements.
`completion-progress.json` and `reference-completion.json` record full
*cold-start stress* progress; `benchmark-progress.json` is the benchmark
admission ledger. Candidate atlas PDFs and preview PNGs are generated by
`render_v622_benchmarks.py`; pending courses are visibly labeled.

The initial nominal search completed with 137/152 courses screened successfully:
131/144 original generated choices and 6/8 public references. This is not the
full qualification result. The expanded search found clean CDRA profiles, which
the first search lacked. Final counts must be read from the qualification and
release ledgers, not inferred from this initial snapshot.

## Evaluation and contamination policy

Report all-course macro completion, generated-only completion, public-reference
completion, per-stratum completion and per-course intervals. Also report
conditional gate survival, failure phase, successful-lap median/p10/mean,
mean-to-fastest gap, and matched-start pace comparisons. Pace comparisons need
completion counts so a faster survivor mean cannot hide reliability loss.

The eight public references have already influenced training-distribution
design and real50 validation. They are not untouched zero-shot tests. Real60's
fresh generated slice can support a stronger geometry-holdout claim after it
is frozen and excluded from training and checkpoint selection. Its design is
still informed by earlier research; say so.

Only **real100-hard** may drive future requirement target counts. Those targets
should use independent-course witness floors and horizon-specific coverage.
Training generated from them is benchmark-informed; real100-hard subsequently
measures development generalization, not untouched-test generalization. Use
independent train/development geometry and retain the anti-clone protection
against both benchmark suites. Shared public-reference geometry or teacher
trajectories must not enter training under a stronger zero-shot claim.
The requirement export includes a separate executed speed/braking/radius cell
section from the accepted hard/public teachers. It verifies all 100 geometry
fingerprints against the frozen manifest and excludes real60 records. These
teacher-dependent coordinates complement rather than replace geometric cells.

For release, freeze geometry, manifests, teacher contracts, evaluation seeds,
source/reconstruction provenance and per-course qualification. Audit source
redistribution terms before external publication. Do not freeze a partially
qualified set just to hit the requested counts.

## Follow-on pretraining experiments

The dataset work does not require implementing the recovery schedule now.
A useful later experiment is family-conditioned competence with hysteresis:
reduce learner/recovery rollout sampling only after sustained high success
across every family, measure pace dispersion on independent development
courses, and restore recovery exposure after a reliability drop. Distinguish
expert action mixing, which states are visited, replay fractions, and solver
recovery mode; these are different interventions. Compare against a fixed
mixture under matched labels and compute rather than attributing all effects
to one knob. Tune teacher pace per course first, retain unsuccessful-trial
counts, and collect multiple speed commands per geometry.

A concrete starting ablation is learner-induced rollout fractions
0.65 → 0.45 → 0.25 → 0.10 when the weakest development-family success EMA
exceeds 0.80 / 0.88 / 0.94 for three consecutive checks, with at least 64 recent
episodes per family and both speed panels represented. Step back one level
if any family drops by five points from its stage-entry baseline. Compare the
final 10% recovery floor with a pure-expert terminal phase; neither is claimed
optimal. Do not use real60 for these triggers. Keep trajectory sampling,
expert action mixing and replay weighting separately logged.

For the speed knob, choose caps by *measured teacher response*, not just fixed
fractions of a nominal maximum. Require a low-command demonstration to be
meaningfully slower on the same geometry (for example ≥15% longer median lap
time under matched seeds), otherwise lower the cap and remeasure. A three-cap
grid with matched starts and a majority of frontier-speed labels is a useful
pilot. It avoids collecting several different high commands whose trajectories
are all pinned to the same curvature-limited speed.
