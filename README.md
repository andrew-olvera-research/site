# Andrew Olvera research portfolio

Static HTML on the existing GitHub Pages deployment, `main` / root. No application build is required.
Preview: `python -m http.server 8080 --directory .`.

## Current content

- `starscream/` documents the October 27, 2026 motion64 planner, its training stages, current input contract, nominal plant and selected flights.
- `starscream/earlier-study.html` retains the October 13 supervised-controller comparison with its original data, charts and historical conditions. Those results do not characterize the new planner.
- Tells remains at `microrts/`. RLVR remains a site report at `token-level-rewards/`.
- `Andrew_Olvera_CV.pdf` is generated from the public editable `Andrew_Olvera_CV.tex` with `pdflatex`. Preserve unrelated approved facts when editing it. Starscream and Tells are provisional manuscript titles; RLVR is not a manuscript.
- `manippo/`, tests and this README remain excluded from the Jekyll build.

## Motion64 provenance

Only approved MP4/GIF/poster/trajectory assets and compact public specifications were copied from `starscream-motion64-simple3d-best-2026-10-27-v1`. The ZIP, raw provenance traces, audit files and evaluation records are not deployed. `starscream/assets/motion64/manifest.json` records the source bundle, ZIP hash, checkpoint hash/update/steps, cohort conditions, selected clip timings and SHA-256 values for the published media.

The main gallery contains eight predefined learned layouts under the deterministic mean policy. Two sampled development-transfer successes retain their 3/17 and 4/17 denominators in a separate section. Do not relabel these as zero-shot benchmark results or mix the checkpoint with historical policies. Generalization is a research goal, not an ablation conclusion.

The existing dependency-free `Scene3D` component is retained. Its gate geometry now prefers the supplied world-from-gate rotation matrix and rectangular opening, falling back to yaw/square geometry for older assets. Position/orientation samples, timing and paths are unchanged.

## Validation

Run `python tests/check_site.py` for local links, media/architecture hooks, editorial guards, historical data calculations, current public hashes, split/cohort separation and physical trajectory durations. Inspect current and historical Starscream pages, homepage, Tells and RLVR at desktop/mobile sizes. Check video playback, all eight 3D selections, diagram scrolling, reduced motion and the rendered CV.

The source bundle's `validate.py` was run separately to decode every GIF frame, probe each MP4 and check quaternion/rotation geometry, timing and independent audit records. It and private audit material are not public site assets.

## Earlier study data

`python tests/export_study.py --paper PATH_TO_FROZEN_PAPER_PACKAGE --site .` verifies all 11,340 episode rows against 18 source policy summaries before producing the compact comparison JSON. `python tests/summarize_geometry.py` regenerates course descriptors from existing manifests. The historic aliases pretrain65/eval100/eval60 and controlled-recovery panel retain their original definitions.
