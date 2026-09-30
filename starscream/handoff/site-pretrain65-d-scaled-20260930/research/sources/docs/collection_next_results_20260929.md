# Collection-next screen: completed results

All 12 scratch arms completed 24 rounds and 15,024 optimizer updates each. They
share the corrected inputs, objective and replay baseline except for the
specified collection change (and the equal-axis objective arm). This is one
training seed on nine mini-slalom training courses, with a repeated five-course,
20-episode validation panel. The two repeat arms use a different training seed
from the previous A/D pilots. Environment steps and labels differ greatly.

Each cell below is **family-weighted eventual / clean-and-timely success**.
The final column reports the last six checkpoints on the same 20 evaluation
episodes, so those checkpoints are trend observations, not independent trials.

| Arm | Steps, millions | Final E / C | Last-six E / C | Final raw E / C |
|---|---:|---:|---:|---:|
| A repeat: per-step mix | 1.38 | 15% / 15% | 18% / 14% | 3/20 / 3/20 |
| D repeat: 2–4 gate learner | 1.72 | 53% / **30%** | 50% / 13% | 10/20 / 6/20 |
| Full learner rollout | 2.51 | 60% / 25% | **60%** / 9% | 10/20 / 5/20 |
| Learner 1–2 gates | 1.01 | 50% / 0% | 33% / 3% | 8/20 / 0/20 |
| Learner 4–6 gates | 2.24 | 65% / 10% | 50% / 7% | 12/20 / 2/20 |
| Learner 2 s window | 0.66 | 10% / 5% | 8% / 3% | 2/20 / 1/20 |
| Learner 4 s window | 0.97 | 35% / 15% | 33% / 12% | 7/20 / 3/20 |
| Full learner + 20% expert episodes | 1.94 | 58% / 5% | 39% / 7% | 10/20 / 1/20 |
| Gate segments + 20% expert episodes | 1.56 | 35% / 10% | 44% / 5% | 7/20 / 2/20 |
| Relaxed learner recovery, 4 s | 1.31 | 35% / 20% | 45% / 13% | 7/20 / 4/20 |
| Relaxed learner recovery, 8 s | 1.74 | **68%** / 10% | 51% / **15%** | 11/20 / 2/20 |
| D + equal action-axis weights | 1.74 | 58% / 0% | 40% / 8% | 10/20 / 0/20 |

The fresh 100-episode evaluation of the *previous* A, D and fixed-bounded
checkpoints remains separate evidence: A had 20 eventual / 14 clean-timely, D
45 / 13 and bounded 10 / 10. D's eventual gain persists in its second-seed
training run; the clean/timely ordering does not. A repeat's 15% final
clean/timely versus D repeat's 30% reverses the original 25% versus 20%
endpoint. D's 30% final is its late-window high; its six-checkpoint average is
13%, so the endpoint alone is too unstable to claim a decisive clean-flight
win. Both training seeds nevertheless favor learner-controlled collection for
eventual completion.

## What the collection changes suggest

Full learner rollouts collected 2.51 million steps and sustained about 60%
weighted eventual completion across the last six checkpoints, above D's 50%.
Their clean/timely average was 9%, below D's 13%. This separates coherent
learner execution from D's 2–4 accepted-gate truncation: longer learner
exposure supports survival and completion, with no demonstrated corresponding
improvement in expert-like first attempts.

Among gate lengths, 1–2 gates ended 870 segments and underfilled replay in
three rounds. Four to six gates collected 2.24 million steps and had high
eventual completion at the final checkpoint, but only 7% late-window
clean/timely. A two-second window underfilled seven rounds and produced little
learning. Four seconds filled replay but remained below D on eventual
completion. These windows change both trajectory duration and state exposure;
the short-arm deficits cannot be attributed only to their endpoint rule.

The 20% coherent-expert arms did execute teacher actions (199k active teacher
steps in full episodes; 97k in gate segments), but neither improved the
late-window clean/timely result over D. Adding expert episodes this way is not
yet a reliable replacement for first-attempt supervision.

Relaxed learner recovery has a more interesting tradeoff than the earlier C/E
arms. The 4-second arm recorded 145 completed versus 1,154 failed collection
recoveries and averaged 45% eventual / 13% clean-timely in the last six
checkpoints. The 8-second arm recorded 462 completed versus 1,080 failed,
averaging 51% eventual / 15% clean-timely. Unlike old C/E, 723 and 390
recovery timeouts actually fired in these arms. This shows the earlier 5 m
guard and invalid-query stopping made the old clock a poor test of recovery
duration. It does **not** isolate which relaxed guard caused the improvement.

The 8-second arm's final 68% eventual high is paired with only 10%
clean/timely. At its best-timely checkpoint (round 23), it was 55% eventual,
30% timely and 25% clean/timely. That earlier checkpoint is a candidate for
further evaluation, not a confirmed winner selected on this repeated panel.

Recovery occupancy is the main caution. Across active rounds 2–24, the
8-second arm retained about 263k recovery-tagged rows out of 575k online rows
(46%). The 4-second arm retained 217k of 575k (38%). Those are **retained
label** shares, not measured optimizer draws or autonomous recovery time. They
are large enough that the 8-second recipe should not be scaled unchanged for a
base model whose normal behavior must be timely. The completed recoveries are
useful; long recoveries should not gain more training influence simply by
lasting longer. D has no explicit recovery class in this telemetry, so a
direct class-share comparison would be misleading.

The equal-axis D arm averaged 8% clean/timely versus D's 13%, despite similar
steps and nominal collection. It changes the objective but has only one seed;
there is no reason from this screen to revert the reduced yaw weight.

## Decision boundary

Keep D as the conservative learner-control baseline and full learner rollouts
as the high-completion comparator. The relaxed 8-second formulation is worth a
fresh evaluation because it has the best late-window balance in this screen,
but its recovery label share is too high for immediate scale. The immediate
question is whether one can retain its rescue capability while limiting each
recovery encounter's action-training weight and preserving precursor decisions.
That objective/retention change was **not** tested by these 12 arms.

No new arm has been evaluated on a larger untouched course panel. The shared
five-course panel is especially weak evidence for the external hard families:
at the final checkpoint D repeat and recovery 8 s each completed 1/8 hard
starts; full learner completed 0/8. The earlier fresh real100 test of the
*previous* D checkpoint found 2/100 eventual and zero timely. More diverse
evaluation and a matched downstream RL budget are needed before deciding to
scale a base-model recipe.

Sources: `outputs/diagnostics/collection-next-20260929/aggregate.json` has all
round curves, best-checkpoint metrics, step/label totals, telemetry and
per-course results; `aggregate.csv` is the compact comparison.
`queue.status.json` and all 12 arm statuses report successful completion.
