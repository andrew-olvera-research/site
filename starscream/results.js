/* Results + behavioural analysis: course explorers, recovery stats, detour strip chart, recovery table.
 * All numbers come from assets/rollouts/index.json (written by assets/rollouts/build.py).
 */
(async () => {
  const DIR = 'assets/rollouts/';
  const BLUE = '#1f5a96', ORANGE = '#c8641e';
  let index;
  try { index = await (await fetch(DIR + 'index.json')).json(); }
  catch (err) { console.error('rollouts', err); return; }

  const fastest = index.filter(m => m.tag === 'fastest');
  const tagged = index.filter(m => m.tag.startsWith('recovery'));
  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) k === 'class' ? el.className = v : el.setAttribute(k, v);
    el.append(...kids);
    return el;
  };
  const median = xs => { const s = [...xs].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

  /* ---------- explorer: one live viewer + a grid of still tiles ---------- */
  const lazy = new IntersectionObserver(entries => entries.forEach(e => {
    if (!e.isIntersecting) return;
    lazy.unobserve(e.target);
    Scene3D.fetch(e.target.dataset.src).then(d => new Scene3D(e.target, { still: true }).load(d));
  }), { rootMargin: '200px' });

  function explorer(el, items, sub) {
    const view = h('div', { class: 'scene3d' });
    const tiles = h('div', { class: 'tiles' });
    el.append(view, tiles);
    const viewer = new Scene3D(view);
    const buttons = items.map(m => {
      const still = h('div', { class: 'still', 'data-src': `${DIR}${m.slug}.json` });
      const b = h('button', { class: 'tile', type: 'button', 'aria-pressed': 'false' },
        still, h('b', {}, m.name), h('span', {}, sub(m)));
      b.onclick = () => select(m.slug);
      lazy.observe(still);
      tiles.append(b);
      return [m.slug, b];
    });
    function select(slug) {
      buttons.forEach(([s, b]) => b.setAttribute('aria-pressed', String(s === slug)));
      Scene3D.fetch(`${DIR}${slug}.json`).then(d => viewer.load(d));
    }
    select(items[0].slug);
    return { select, el };
  }

  const res = explorer(document.querySelector('.explorer[data-set=fastest]'), fastest,
    m => `${m.time.toFixed(1)} s · ${m.gates} gates`);
  const rec = explorer(document.querySelector('.explorer[data-set=recovery]'), tagged,
    m => `${m.recoveries.length} recover${m.recoveries.length === 1 ? 'y' : 'ies'} · ${m.time.toFixed(1)} s`);

  const show = (ex, slug) => { ex.select(slug); ex.el.scrollIntoView({ behavior: 'smooth', block: 'start' }); };

  /* ---------- headline stats ---------- */
  const rl = index.flatMap(m => m.recoveries.map(r => ({ ...r, m })));
  const slow = rl.map(r => r.min_speed);
  const stat = (v, label) => h('div', {}, h('b', {}, v), h('span', {}, label));
  document.getElementById('bstats').append(
    stat(String(rl.length), `recovery legs, all inside the ${tagged.length} tagged rollouts`),
    stat('0', `in the ${fastest.reduce((a, m) => a + m.legs.length, 0)} legs of the fastest set`),
    stat(`${median(rl.map(r => r.time)).toFixed(1)} s`, 'median recovery leg'),
    stat(`${median(rl.map(r => r.extra_path)).toFixed(0)} m`, 'median extra path per recovery'),
    stat(`${Math.min(...slow).toFixed(1)}–${Math.max(...slow).toFixed(1)} m/s`,
      `slowest point in a recovery leg; ${median(fastest.flatMap(m => m.legs.map(l => l.min_speed))).toFixed(1)} m/s typical in the fastest set`),
  );

  /* ---------- strip chart: detour ratio per leg (log x) ---------- */
  const fig = document.getElementById('strip');
  const W = 820, L = 150, R = 16, rowH = 64, top = 8;
  const rows = [
    ['Fastest set', fastest],
    ['Tagged rollouts', tagged],
  ];
  const H = top + rows.length * rowH + 44;
  const lo = Math.log(0.95), hi = Math.log(40);
  const x = v => L + (Math.log(Math.max(v, 0.95)) - lo) / (hi - lo) * (W - L - R);
  const NS = 'http://www.w3.org/2000/svg';
  const s = (tag, attrs, text) => {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (text != null) el.textContent = text;
    return el;
  };
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': 'Detour ratio of every gate-to-gate leg, fastest set versus tagged rollouts, log scale' });
  const axisY = top + rows.length * rowH;
  for (const v of [1, 2, 5, 10, 20]) {
    svg.append(s('line', { class: 'ax', x1: x(v), x2: x(v), y1: top, y2: axisY }));
    svg.append(s('text', { x: x(v), y: axisY + 18, 'text-anchor': 'middle' }, `${v}×`));
  }
  svg.append(s('text', { x: L + (W - L - R) / 2, y: axisY + 38, 'text-anchor': 'middle' }, 'detour ratio (path ÷ straight line), log scale'));
  const tip = h('div', { class: 'tip' });
  let seed = 1;
  const jitter = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
  rows.forEach(([label, set], i) => {
    const cy = top + i * rowH + rowH / 2;
    const n = set.reduce((a, m) => a + m.legs.length, 0);
    svg.append(s('text', { class: 'row-l', x: 0, y: cy - 2 }, label));
    svg.append(s('text', { x: 0, y: cy + 15 }, `${n} legs`));
    // clean legs first so recovery dots sit on top
    const pts = set.flatMap(m => m.legs.map(l => ({ m, l }))).sort((a, b) => a.l.recovery - b.l.recovery);
    for (const { m, l } of pts) {
      const c = s('circle', { cx: x(l.detour), cy: cy + jitter() * (rowH - 26), r: 4,
        fill: l.recovery ? ORANGE : BLUE, 'fill-opacity': l.recovery ? 1 : 0.55, stroke: '#fff', 'stroke-width': 2 });
      c.style.cursor = 'pointer';
      c.addEventListener('pointerenter', () => {
        tip.innerHTML = `<b>${m.name}</b>, gate ${l.gate}<br>detour ${l.detour.toFixed(2)}× · ${l.time.toFixed(2)} s${l.recovery ? ' · <b>recovery</b>' : ''}`;
        tip.style.display = 'block';
        const r = c.getBoundingClientRect(), f = fig.getBoundingClientRect();
        const left = Math.min(r.left - f.left + 10, f.width - tip.offsetWidth);
        tip.style.left = `${Math.max(0, left)}px`; tip.style.top = `${r.top - f.top - tip.offsetHeight - 6}px`;
      });
      c.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
      c.addEventListener('click', () => show(m.tag === 'fastest' ? res : rec, m.slug));
      svg.append(c);
    }
  });
  const legend = h('div', { class: 'legend' },
    h('span', {}, h('i', { style: `--c:${BLUE};opacity:.55` }), 'clean leg'),
    h('span', {}, h('i', { style: `--c:${ORANGE}` }), 'recovery leg'));
  fig.append(legend, h('div', { class: 'scroll-x' }, svg), tip);
  svg.style.minWidth = '560px';

  /* ---------- recovery table ---------- */
  const table = document.getElementById('rtable');
  const cols = [['Course'], ['Tag'], ['Gate', 1], ['Leg time (s)', 1], ['Detour', 1], ['Extra path (m)', 1], ['Slowest (m/s)', 1]];  // tag as exported
  table.append(h('thead', {}, h('tr', {}, ...cols.map(([c, n]) => h('th', n ? { class: 'n' } : {}, c)))));
  const tbody = h('tbody');
  for (const r of rl) {
    const tr = h('tr', { tabindex: '0' },
      h('td', {}, r.m.name), h('td', {}, h('span', { class: 'tag' }, r.m.tag)),
      ...[r.gate, r.time.toFixed(2), `${r.detour.toFixed(2)}×`, r.extra_path.toFixed(1), r.min_speed.toFixed(2)]
        .map(v => h('td', { class: 'n' }, String(v))));
    const go = () => show(rec, r.m.slug);
    tr.onclick = go;
    tr.onkeydown = e => { if (e.key === 'Enter') go(); };
    tbody.append(tr);
  }
  table.append(tbody);
  /* ---------- completion charts (numbers from the v6.21.1 evaluation brief) ---------- */
  function dotChart(fig, { rows, series, label }) {
    const W = 820, L = 170, R = 24, rowH = series.length > 1 ? 52 : 40, top = series.length > 1 ? 14 : 12;
    const H = top + rows.length * rowH + 30;
    const x = v => L + v / 100 * (W - L - R);
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': label });
    const axisY = top + rows.length * rowH;
    for (const v of [0, 25, 50, 75, 100]) {
      svg.append(s('line', { class: 'ax', x1: x(v), x2: x(v), y1: top, y2: axisY }));
      svg.append(s('text', { x: x(v), y: axisY + 18, 'text-anchor': 'middle' }, `${v}%`));
    }
    const tip = h('div', { class: 'tip' });
    const hover = (el, html) => {
      el.addEventListener('pointerenter', () => {
        tip.innerHTML = html; tip.style.display = 'block';
        const r = el.getBoundingClientRect(), f = fig.getBoundingClientRect();
        tip.style.left = `${Math.max(0, Math.min(r.left - f.left + 12, f.width - tip.offsetWidth))}px`;
        tip.style.top = `${r.top - f.top - tip.offsetHeight - 6}px`;
      });
      el.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
    };
    rows.forEach((row, i) => {
      const cy = top + i * rowH + rowH / 2;
      svg.append(s('text', { class: 'row-l', x: 0, y: cy + 4 }, row.name));
      const vals = row.values.filter(v => v);
      if (vals.length > 1) {
        const xs = vals.map(v => x(v.p));
        svg.append(s('line', { x1: Math.min(...xs), x2: Math.max(...xs), y1: cy, y2: cy, stroke: 'var(--line-2)', 'stroke-width': 2 }));
      }
      row.values.forEach((v, k) => {
        if (!v) return;
        const g = s('g', {});
        if (v.lo != null) {
          g.append(s('line', { x1: x(v.lo), x2: x(v.hi), y1: cy, y2: cy, stroke: BLUE, 'stroke-width': 2 }));
          for (const e of [v.lo, v.hi]) g.append(s('line', { x1: x(e), x2: x(e), y1: cy - 5, y2: cy + 5, stroke: BLUE, 'stroke-width': 2 }));
        }
        const hollow = series[k].hollow;
        g.append(s('circle', { cx: x(v.p), cy, r: 5.5, fill: hollow ? '#fff' : BLUE, stroke: hollow ? BLUE : '#fff',
          'stroke-width': 2 }));
        // generous invisible hit target
        g.append(s('circle', { cx: x(v.p), cy, r: 12, fill: 'transparent' }));
        const below = vals.length > 1 && k === 1;
        svg.append(g);
        svg.append(s('text', { class: 'val', x: x(v.p), y: below ? cy + 19 : cy - 10, 'text-anchor': 'middle' }, `${v.p.toFixed(1)}`));
        hover(g, `<b>${row.name}</b>${series[k].name ? `, ${series[k].name}` : ''}<br>${v.p.toFixed(1)}% completed${v.n ? ` · ${v.n}` : ''}` +
          (v.lo != null ? `<br>95% interval ${v.lo}–${v.hi}%` : ''));
      });
    });
    const legend = series.filter(x => x.name).length > 1 ? h('div', { class: 'legend' }, ...series.map(se =>
      h('span', {}, h('i', { style: se.hollow ? `border:2px solid ${BLUE};width:7px;height:7px` : `background:${BLUE}` }), se.name))) : '';
    fig.append(legend, h('div', { class: 'scroll-x' }, svg), tip);
    svg.style.minWidth = '520px';
  }

  dotChart(document.getElementById('suites'), {
    label: 'Completion rate by evaluation suite with 95% course-bootstrap intervals',
    series: [{ name: '' }],
    rows: [
      { name: 'Fresh real60', values: [{ p: 60.6, lo: 52.4, hi: 68.5, n: '252/416, 52 courses' }] },
      { name: 'Fresh hard-v2', values: [{ p: 45.5, lo: 39.4, hi: 51.6, n: '335/736, 92 courses' }] },
      { name: 'Exposed reference', values: [{ p: 23.4, n: '15/64, 8 courses' }] },
    ],
  });
  dotChart(document.getElementById('families'), {
    label: 'Completion rate by route family, fresh real60 (filled) and fresh hard-v2 (hollow)',
    series: [{ name: 'fresh real60' }, { name: 'fresh hard-v2', hollow: true }],
    rows: [
      { name: 'Go-around', values: [{ p: 97.5 }, { p: 83.3 }] },
      { name: 'Compound reversal', values: [null, { p: 85.0 }] },
      { name: 'Ordered 3D', values: [{ p: 85.4 }, { p: 77.5 }] },
      { name: 'Slalom', values: [{ p: 81.2 }, null] },
      { name: 'Long-low', values: [{ p: 18.8 }, { p: 6.2 }] },
    ],
  });
})();
