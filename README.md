# andrew-olvera.github.io

Static personal site, deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`.

```
index.html              → /             landing (name + info + project list)
starscream/index.html   → /starscream/  Starscream research page (single file: HTML + CSS + JS)
starscream/assets/      → demo GIFs, figures, PDF
```

## Adding content

- **Demos:** clips live in `starscream/assets/research-gifs-v6211-update-fix-refined/` (mp4 + poster + gif per clip).
  The Results gallery is generated from the `CLIPS` array in the page's `<script>` — edit course names
  and order there. The page plays the MP4s and falls back to the GIF if an MP4 can't load.
- **Architecture:** the diagram is inline SVG in the Method section — edit node labels or replace it.
- **Placeholders:** search for `TODO` and `—` to find text/numbers to fill in.
- **New page:** create `<name>/index.html` and add a row to the Projects list in the root `index.html`.

## Local preview

```
python -m http.server 8080
```

Then open http://localhost:8080 and http://localhost:8080/starscream/.
