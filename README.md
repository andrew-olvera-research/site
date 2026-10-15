# Andrew Olvera research portfolio

Static HTML served by the existing GitHub Pages deployment from `main` / root.
No application build or package installation is required. Preview with
`python -m http.server 8080 --directory .` and open http://localhost:8080/.

## Content and evidence

- Home links to Starscream, Tells (`microrts/`), and small-model RLVR (`token-level-rewards/`).
- Starscream leads with the three-seed development comparison from the October 13, 2026 focused review draft (method, evaluation, results sections). The compact public export retains per-seed counts, denominators, definitions, and source hashes; `study.js` renders the outcome charts from it. Curated recorded flights are explicitly separate from the three-seed aggregate. Old recipe/fix comparisons are no longer presented; source assets remain unchanged.
- Tells and RLVR summarize the evidence and figures already present in this repository; no publication or paper URL is asserted.
- `manippo/` is retained as source history and excluded from the public Jekyll build.
- Future results should be added only with a dated source, evaluation contract, and stated limitations. Do not mix results across checkpoints or claim an isolated architecture effect from recipe comparisons.

## Validation

Run `python tests/check_site.py` for local links, required visual hooks, retirement, and editorial guardrails. Manually inspect the three project pages and homepage at desktop and mobile widths; test charts, videos, and 3D viewers. Starscream core content is open; no disclosures hide its findings.

## Starscream evidence refresh

`python tests/export_study.py --paper PATH_TO_FROZEN_PAPER_PACKAGE --site .` verifies all 11,340 episode rows against 18 source policy summaries before producing the compact comparison JSON. Package-relative provenance and SHA-256 hashes are retained; no machine paths or training logs are published. The source-count CSV is an unchanged copy of the package table. `python tests/summarize_geometry.py` regenerates course descriptors from manifests already in this repository.

Public aliases are presentation-only: pretrain65 selects qualified train entries; eval100 maps to the hard-v2 100-course manifest; eval60 maps to the separate 60-course manifest. The controlled-recovery panel has 60 starts across 15 eval100 courses and is not eval60.
