# Andrew Olvera research portfolio

Static HTML served by the existing GitHub Pages deployment from `main` / root.
No application build or package installation is required. Preview with
`python -m http.server 8080 --directory .` and open http://localhost:8080/.

## Content and evidence

- Home links to Starscream, Tells (`microrts/`), and small-model RLVR (`token-level-rewards/`).
- Starscream leads with the three-seed development comparison from the October 13, 2026 focused review draft (method, evaluation, results sections). Its machine-readable summary records provenance and limitations. The September 29 checkpoint charts and flight assets remain under explicitly historical diagnostics.
- Tells and RLVR summarize the evidence and figures already present in this repository; no publication or paper URL is asserted.
- `manippo/` is retained as source history and excluded from the public Jekyll build.
- Future results should be added only with a dated source, evaluation contract, and stated limitations. Do not mix results across checkpoints or claim an isolated architecture effect from recipe comparisons.

## Validation

Run `python tests/check_site.py` for local links, required visual hooks, retirement, and editorial guardrails. Manually inspect the three project pages and homepage at desktop and mobile widths; test expanded charts, videos, and 3D viewers.
