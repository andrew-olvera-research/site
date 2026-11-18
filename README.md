# Andrew Olvera research portfolio

Static HTML on the existing GitHub Pages deployment, `main` / root. No application build is required.
Preview: `python -m http.server 8080 --directory .`.

## Current content

- `starscream/` is the preprint teaser (updated October 29, 2026): Architecture (why the planner/controller split, control-loop diagram), Methodology (five-stage training table), Evaluation (three-band geometric corpus and how the same analysis drives online course generation), and Results (band completion, per-course chart, ten held-out flights in the interactive 3D viewer, JEPA pace ablation). `charts.js` draws both charts from `assets/results/results-2026-10-29.json`, drives the flight picker (`assets/flights/*.json`, slimmed from the bundle's demo trajectories) and plays the hero video while on screen.
- `starscream/earlier-study.html` is kept but no longer linked from the current page. It retains the October 13 supervised-controller comparison with its original data, charts and historical conditions. Those results do not characterize the new planner.
- `microrts/` is the research report "Learning under fog of war": a four-arm fog-of-war critic comparison (fog vs. pretrained full-state critic, with and without learned-model rollouts), an intended-action diagnostic, and the Tells return-prediction result. Figures and downloadable records live in `microrts/assets/microrts-report/` (`figures/`, and `data/` with CSV/JSON files and its own `index.html`). The public code link is pinned to the historical September snapshot of AO-OpenResearch; repoint it only once the factorial code and results are published there.
- `microrts/tells.html` is the archived Tells project page (the historical opponent-belief study), labeled as an archive and linked from the report. Tells remains that study's name.
- `microrts/tools/` holds optional helpers excluded from the build: `make_figures.py` regenerates the three PNGs from the bundled data (needs Matplotlib); `export_evidence.py` re-exports the data from the original research checkout.
- Starscream is labeled "Preprint coming soon"; RLVR (`token-level-rewards/`) is labeled a report.
- `Andrew_Olvera_CV.pdf` is generated from the public editable `Andrew_Olvera_CV.tex` with `pdflatex`. Preserve unrelated approved facts when editing it. Starscream and Tells are provisional manuscript titles; RLVR is not a manuscript.
- `manippo/`, `microrts/tools/`, tests and this README remain excluded from the Jekyll build.

## Oct 29 evidence

Copied from `starscream-full-pretrain-geometry-bands-2026-10-29-v2`: ten MP4 flights and posters (`assets/flights/`) and a compact export (`assets/results/results-2026-10-29.json`) of the Val18-selected checkpoint (SHA-256 `a35f9874…f1cba`, 696.7M RL transitions): per-course clean counts for all 80 held-out courses, band totals, and the val-selected JEPA ablation's per-course paired lap-time fractions. Raw episodes, NPZ states, source code and provenance traces stay in the bundle. The October 27 motion64 media were removed; they came from a different checkpoint.

## Validation

Run `python tests/check_site.py` for local links, media/architecture hooks, editorial guards, historical data calculations, current public hashes, split/cohort separation and physical trajectory durations. Inspect current and historical Starscream pages, homepage, the MicroRTS report, archived Tells and RLVR at desktop/mobile sizes. Check video playback, both charts (hover and resize), sideways diagram scrolling on phones, reduced motion and the rendered CV.

The source bundle's `validate.py` was run separately to decode every GIF frame, probe each MP4 and check quaternion/rotation geometry, timing and independent audit records. It and private audit material are not public site assets.

## Earlier study data

`python tests/export_study.py --paper PATH_TO_FROZEN_PAPER_PACKAGE --site .` verifies all 11,340 episode rows against 18 source policy summaries before producing the compact comparison JSON. `starscream/assets/data/course-geometry.json` records the manifest paths and SHA-256 hashes it was summarized from; the September research handoff it came from is no longer stored in the site. The historic aliases pretrain65/eval100/eval60 and controlled-recovery panel retain their original definitions.
