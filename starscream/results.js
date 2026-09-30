/* Starscream results page: charts, clip grid, 3D explorers and the recovery table.
 * Chart numbers come from assets/data/results.json (built by assets/data/build.py from the handoff);
 * rollout metrics come from assets/rollouts/index.json (built by assets/rollouts/build.py).
 */
(async () => {
  const RDIR = 'assets/rollouts/';
  const MEDIA = 'handoff/site-pretrain65-d-scaled-20260930/generated/';
  const C = { new: '#1f5a96', old: '#9aa1ab', c2: '#c8641e', c3: '#1a9a7a', light: '#6da7ec', rest: '#c3c8cf' };
  const NS = 'http://www.w3.org/2000/svg';

  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) k === 'class' ? el.className = v : el.setAttribute(k, v);
    el.append(...kids);
    return el;
  };
  const s = (tag, attrs = {}, text) => {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (text != null) el.textContent = text;
    return el;
  };
  const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
  const svgRoot = (W, H, label, minW = 520) => {
    const el = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': label });
    el.style.minWidth = `${minW}px`;
    return el;
  };
  const legend = items => h('div', { class: 'legend' }, ...items.map(([c, t, kind = '']) =>
    h('span', {}, h('i', { class: kind, style: `--c:${c}` }), t)));

  /* one tooltip per figure; marks call show(evt, html) */
  function tipper(fig) {
    const tip = h('div', { class: 'tip' });
    fig.append(tip);
    return {
      show(evt, html) {
        tip.innerHTML = html;
        tip.style.display = 'block';
        const f = fig.getBoundingClientRect();
        const x = evt.clientX - f.left, y = evt.clientY - f.top;
        tip.style.left = `${Math.max(0, Math.min(x + 12, f.width - tip.offsetWidth))}px`;
        tip.style.top = `${Math.max(0, y - tip.offsetHeight - 10)}px`;
      },
      hide() { tip.style.display = 'none'; },
    };
  }
  const hover = (el, t, html) => {
    el.addEventListener('pointermove', e => t.show(e, typeof html === 'function' ? html() : html));
    el.addEventListener('pointerleave', () => t.hide());
  };
  const mount = (id, ...kids) => { const f = document.getElementById(id); f.append(...kids); return f; };
  const scrollBox = svg => h('div', { class: 'scroll-x' }, svg);

  let D, index;
  try {
    [D, index] = await Promise.all([
      fetch('assets/data/results.json').then(r => r.json()),
      fetch(RDIR + 'index.json').then(r => r.json()),
    ]);
  } catch (err) { console.error('results data', err); return; }

  const FAM = {
    'behavior:compound_reversal': 'Compound reversal', 'behavior:long_low_braking': 'Long low braking',
    'behavior:radius_switch': 'Changing-radius turns', 'behavior:vertical_chain': 'Vertical chain',
    'behavior:wrong_side_incidence': 'Oblique approaches', diving_hairpin: 'Diving hairpin', flow: 'Flow',
    go_around: 'Go-around', hairpin_chain: 'Hairpin chain', long_braking: 'Long braking', long_low: 'Long low',
    ordered_3d: 'Ordered 3D', public_reference: 'Public reference', slalom: 'Slalom', stacked_reversal: 'Stacked reversal',
  };

  /* ================= Figure 1: suites, old -> new ================= */
  {
    const W = 820, L = 150, R = 60, rowH = 54, top = 10;
    const rows = D.suites.map(r => ({ ...r, label: r.suite }));
    const H = top + rows.length * rowH + 34;
    const x = v => L + v * (W - L - R);
    const svg = svgRoot(W, H, 'Eventual completion, previous base vs current base, per evaluation panel');
    const fig = document.getElementById('fig-suites');
    const t = tipper(fig);
    const axisY = top + rows.length * rowH;
    for (const v of [0, .25, .5, .75, 1]) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: x(v), x2: x(v), y1: top, y2: axisY }));
      svg.append(s('text', { x: x(v), y: axisY + 18, 'text-anchor': 'middle' }, `${v * 100}%`));
    }
    rows.forEach((r, i) => {
      const cy = top + i * rowH + rowH / 2;
      svg.append(s('text', { class: 'row-l', x: 0, y: cy + 4 }, r.label));
      svg.append(s('line', { x1: x(r.old), x2: x(r.new), y1: cy, y2: cy, stroke: C.new, 'stroke-width': 2, 'stroke-opacity': .35 }));
      svg.append(s('circle', { cx: x(r.old), cy, r: 6, fill: C.old, stroke: '#fff', 'stroke-width': 2 }));
      svg.append(s('circle', { cx: x(r.new), cy, r: 7, fill: C.new, stroke: '#fff', 'stroke-width': 2 }));
      svg.append(s('text', { class: 'val m', x: x(r.old), y: cy - 12, 'text-anchor': 'middle' }, pct(r.old)));
      svg.append(s('text', { class: 'val', x: x(r.new) + 12, y: cy + 4 }, pct(r.new)));
      const hit = s('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width: W, height: rowH });
      hover(hit, t, `<b>${r.label}</b><br><span class="k">previous</span> ${pct(r.old)} · ${r.old_starts} starts/course<br>` +
        `<span class="k">current</span> ${pct(r.new)} · ${r.new_starts} starts/course, ${r.new_total.toLocaleString()} total<br>` +
        `<span class="k">change</span> +${((r.new - r.old) * 100).toFixed(1)} points`);
      svg.append(hit);
    });
    fig.prepend(legend([[C.old, 'previous base (v6.21.1)'], [C.new, 'current base (scaled D, round 71)']]));
    fig.append(scrollBox(svg));
  }

  /* ================= Figure 2: ladder + outcome partition ================= */
  {
    const r = D.real100, n = r.episodes;
    const rows = [
      ['Eventual', 'all gates in order, no crash', r.eventual],
      ['Timely', 'and inside the course deadline', r.timely],
      ['Clean', 'and no missed gate', r.clean],
      ['Clean and timely', 'both', r.clean_timely],
    ];
    const W = 820, L = 190, R = 70, rowH = 40, top = 6;
    const H = top + rows.length * rowH + 30;
    const x = v => L + v * (W - L - R);
    const svg = svgRoot(W, H, 'real100-v2 success rate at each bar');
    const fig = document.getElementById('fig-ladder');
    const t = tipper(fig);
    const axisY = top + rows.length * rowH;
    for (const v of [0, .25, .5, .75, 1]) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: x(v), x2: x(v), y1: top, y2: axisY }));
      svg.append(s('text', { x: x(v), y: axisY + 18, 'text-anchor': 'middle' }, `${v * 100}%`));
    }
    rows.forEach(([lab, sub, v], i) => {
      const y = top + i * rowH + 8, bh = rowH - 16;
      svg.append(s('text', { class: 'row-l', x: 0, y: y + bh / 2 + 4 }, lab));
      const bar = s('rect', { x: x(0), y, width: x(v) - x(0), height: bh, rx: 3, fill: C.new, 'fill-opacity': 1 - i * .16 });
      svg.append(bar);
      svg.append(s('text', { class: 'val', x: x(v) + 8, y: y + bh / 2 + 4 }, pct(v)));
      const hit = s('rect', { class: 'hit', x: 0, y: y - 8, width: W, height: rowH });
      hover(hit, t, `<b>${lab}</b>: ${sub}<br>${Math.round(v * n).toLocaleString()} of ${n.toLocaleString()} starts · ${pct(v, 2)}`);
      svg.append(hit);
    });
    fig.append(scrollBox(svg));

    /* partition */
    const o = r.outcomes;
    const parts = [
      ['clean_timely', 'clean and timely', C.new],
      ['missed_timely', 'timely after a miss', C.light],
      ['clean_late', 'clean but late', C.c3],
      ['missed_late', 'late after a miss', C.c2],
      ['noncompletion', 'did not finish', C.rest],
    ];
    const W2 = 820, bh = 34, H2 = 70;
    const x2 = v => v / n * W2;
    const svg2 = svgRoot(W2, H2, 'Outcome partition of the 3,200 real100-v2 starts');
    const fig2 = document.getElementById('fig-outcomes');
    const t2 = tipper(fig2);
    let acc = 0;
    for (const [k, lab, col] of parts) {
      const v = o[k];
      const seg = s('rect', { x: x2(acc) + (acc ? 1 : 0), y: 4, width: Math.max(0, x2(v) - (acc ? 2 : 1)), height: bh, fill: col, rx: 2 });
      hover(seg, t2, `<b>${lab}</b><br>${v.toLocaleString()} starts · ${pct(v / n)}` +
        (k === 'noncompletion' ? `<br><span class="k">${r.termination.crash} crashes, ${r.termination.hard_cap} hit the 46 s cap</span>` : ''));
      svg2.append(seg);
      if (v / n > .06) svg2.append(s('text', { class: 'val', x: x2(acc) + 8, y: 4 + bh / 2 + 4, fill: k === 'clean_timely' ? '#fff' : '#1a1d23', style: k === 'clean_timely' ? 'fill:#fff' : '' }, pct(v / n)));
      acc += v;
    }
    svg2.append(s('text', { x: 0, y: H2 - 8 }, '0'));
    svg2.append(s('text', { x: W2, y: H2 - 8, 'text-anchor': 'end' }, `${n.toLocaleString()} starts`));
    fig2.append(legend(parts.map(([k, lab, col]) => [col, `${lab} (${o[k].toLocaleString()})`, 'sq'])), scrollBox(svg2));
  }

  /* ================= Figure 3: families ================= */
  {
    const rows = [...D.families].sort((a, b) => b.eventual - a.eventual || b.clean_timely - a.clean_timely);
    const W = 820, L = 200, R = 20, rowH = 30, top = 6;
    const H = top + rows.length * rowH + 30;
    const x = v => L + v * (W - L - R);
    const svg = svgRoot(W, H, 'Completion by route family: previous, current, and current clean-and-timely', 560);
    const fig = document.getElementById('fig-families');
    const t = tipper(fig);
    const axisY = top + rows.length * rowH;
    for (const v of [0, .25, .5, .75, 1]) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: x(v), x2: x(v), y1: top, y2: axisY }));
      svg.append(s('text', { x: x(v), y: axisY + 18, 'text-anchor': 'middle' }, `${v * 100}%`));
    }
    rows.forEach((f, i) => {
      const cy = top + i * rowH + rowH / 2;
      svg.append(s('text', { class: 'row-l', x: 0, y: cy + 4 }, FAM[f.family] || f.family));
      svg.append(s('text', { x: L - 10, y: cy + 4, 'text-anchor': 'end' }, `${f.courses}`));
      svg.append(s('line', { x1: x(f.old), x2: x(f.eventual), y1: cy, y2: cy, stroke: C.new, 'stroke-width': 2, 'stroke-opacity': .3 }));
      svg.append(s('circle', { cx: x(f.old), cy, r: 5, fill: C.old, stroke: '#fff', 'stroke-width': 2 }));
      svg.append(s('circle', { cx: x(f.clean_timely), cy, r: 5, fill: '#fff', stroke: C.c3, 'stroke-width': 2.2 }));
      svg.append(s('circle', { cx: x(f.eventual), cy, r: 6, fill: C.new, stroke: '#fff', 'stroke-width': 2 }));
      const hit = s('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width: W, height: rowH });
      hover(hit, t, `<b>${FAM[f.family] || f.family}</b> · ${f.courses} courses, ${f.episodes} starts<br>` +
        `<span class="k">previous completion</span> ${pct(f.old)}<br><span class="k">completion</span> ${pct(f.eventual)}<br>` +
        `<span class="k">timely</span> ${pct(f.timely)}<br><span class="k">clean and timely</span> ${pct(f.clean_timely)}` +
        (f.median_s ? `<br><span class="k">median successful lap</span> ${f.median_s.toFixed(2)} s` : ''));
      svg.append(hit);
    });
    svg.append(s('text', { x: L - 10, y: top - 0, 'text-anchor': 'end', style: 'font-size:11px' }, ''));
    fig.append(legend([[C.old, 'previous completion'], [C.new, 'current completion'], [C.c3, 'current clean and timely', 'ring']]),
      scrollBox(svg), h('p', { class: 'chart-note' }, 'The number beside each family is its course count.'));
  }

  /* ================= Figure 4: per-course scatter ================= */
  {
    const W = 560, L = 56, B = 46, T = 10, R = 16, P = W - L - R;
    const H = T + P + B;
    const x = v => L + v * P, y = v => T + (1 - v) * P;
    const svg = svgRoot(W, H, 'Per-course completion, previous vs current base', 360);
    const fig = document.getElementById('fig-courses');
    fig.style.maxWidth = '560px';
    fig.style.marginInline = 'auto';
    const t = tipper(fig);
    for (const v of [0, .25, .5, .75, 1]) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: x(v), x2: x(v), y1: T, y2: T + P }));
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: L, x2: L + P, y1: y(v), y2: y(v) }));
      svg.append(s('text', { x: x(v), y: T + P + 18, 'text-anchor': 'middle' }, `${v * 100}%`));
      svg.append(s('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end' }, `${v * 100}%`));
    }
    svg.append(s('line', { class: 'ref', x1: x(0), y1: y(0), x2: x(1), y2: y(1) }));
    svg.append(s('text', { x: L + P / 2, y: H - 6, 'text-anchor': 'middle' }, 'previous base completion (8 starts)'));
    svg.append(s('text', { x: 14, y: T + P / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${T + P / 2})` }, 'current base completion (32 starts)'));
    let seed = 7;
    const jit = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.028;
    const pts = [...D.courses].sort((a, b) => (a.family === 'public_reference') - (b.family === 'public_reference'));
    for (const c of pts) {
      const pub = c.family === 'public_reference';
      const cx = x(Math.min(1, Math.max(0, c.old + jit()))), cy = y(Math.min(1, Math.max(0, c.eventual + jit() * .5)));
      const dot = s('circle', { cx, cy, r: 5, fill: pub ? C.c2 : C.new, 'fill-opacity': pub ? .95 : .6, stroke: '#fff', 'stroke-width': 1.5 });
      hover(dot, t, `<b>${c.name}</b><br><span class="k">${FAM[c.family] || c.family}</span><br>` +
        `<span class="k">previous</span> ${pct(c.old, 0)} · <span class="k">current</span> ${pct(c.eventual, 1)}<br>` +
        `<span class="k">clean and timely</span> ${pct(c.clean_timely, 1)}` +
        (c.median_s ? `<br><span class="k">median lap</span> ${c.median_s.toFixed(2)} s (ref ${c.reference_s.toFixed(2)} s)` : ''));
      svg.append(dot);
    }
    fig.append(legend([[C.new, 'generated hard course'], [C.c2, 'public reference adaptation']]), h('div', { class: 'scroll-x' }, svg));
  }

  /* ================= Figure 5: learning curves (log x) ================= */
  {
    const d = D.learning;
    const Dn = d.scaled_d.filter(r => r[0] > 0), En = d.earlier.filter(r => r[0] > 0);
    const W = 820, L = 52, R = 110, T = 12, B = 44, Hp = 300;
    const H = T + Hp + B;
    const lo = Math.log10(5e5), hi = Math.log10(2e8);
    const x = v => L + (Math.log10(v) - lo) / (hi - lo) * (W - L - R);
    const y = v => T + (1 - v) * Hp;
    const svg = svgRoot(W, H, 'Selection-panel completion vs environment steps, scaled D vs earlier recipe, log scale');
    const fig = document.getElementById('fig-learning');
    const t = tipper(fig);
    for (const v of [0, .25, .5, .75, 1]) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(s('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end' }, `${v * 100}%`));
    }
    for (const [v, lab] of [[1e6, '1M'], [3e6, '3M'], [1e7, '10M'], [3e7, '30M'], [1e8, '100M']]) {
      svg.append(s('line', { class: 'ax', x1: x(v), x2: x(v), y1: T, y2: T + Hp }));
      svg.append(s('text', { x: x(v), y: T + Hp + 18, 'text-anchor': 'middle' }, lab));
    }
    svg.append(s('text', { x: L + (W - L - R) / 2, y: H - 6, 'text-anchor': 'middle' }, 'environment steps (log scale)'));
    svg.append(s('line', { class: 'ref', x1: L, x2: W - R, y1: y(.5), y2: y(.5) }));
    const line = (pts, col, w = 2) => s('polyline', { points: pts.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' '),
      fill: 'none', stroke: col, 'stroke-width': w, 'stroke-linejoin': 'round' });
    svg.append(line(En, C.old), line(Dn, C.new, 2.4));
    const first = (pts, th) => pts.find(r => r[1] >= th);
    const dx = first(Dn, .5), ex = first(En, .5);
    for (const [p, col, lab, anchor, dy] of [[dx, C.new, `${(dx[0] / 1e6).toFixed(1)}M`, 'end', -10], [ex, C.old, `${(ex[0] / 1e6).toFixed(1)}M`, 'end', 22]]) {
      svg.append(s('line', { x1: x(p[0]), x2: x(p[0]), y1: y(.5), y2: T + Hp, stroke: col, 'stroke-dasharray': '2 3' }));
      svg.append(s('circle', { cx: x(p[0]), cy: y(.5), r: 5, fill: col, stroke: '#fff', 'stroke-width': 2 }));
      svg.append(s('text', { class: 'val', x: x(p[0]) + (anchor === 'end' ? -8 : 8), y: y(.5) + dy, 'text-anchor': anchor }, `50% at ${lab}`));
    }
    const sel = Dn.reduce((a, r) => Math.abs(r[0] - d.selected_step) < Math.abs(a[0] - d.selected_step) ? r : a);
    const sx = x(sel[0]), sy = y(sel[1]);
    svg.append(s('path', { d: `M${sx} ${sy - 7}L${sx + 7} ${sy}L${sx} ${sy + 7}L${sx - 7} ${sy}Z`, fill: C.new, stroke: '#fff', 'stroke-width': 2 }));
    const endD = Dn[Dn.length - 1], endE = En[En.length - 1];
    svg.append(s('text', { class: 'row-l', x: x(endD[0]) + 8, y: y(endD[1]) - 8 }, 'Scaled D'));
    svg.append(s('text', { class: 'row-l', x: x(endE[0]) + 8, y: y(endE[1]) + 4 }, 'Earlier recipe'));
    svg.append(s('text', { x: x(endE[0]) + 8, y: y(endE[1]) + 19 }, '218 rounds'));
    svg.append(s('text', { x: x(endD[0]) + 8, y: y(endD[1]) + 7 }, `round ${Dn.length}`));
    /* crosshair */
    const cross = s('line', { x1: 0, x2: 0, y1: T, y2: T + Hp, stroke: '#1a1d23', 'stroke-opacity': .25, visibility: 'hidden' });
    const mD = s('circle', { r: 4, fill: C.new, visibility: 'hidden' }), mE = s('circle', { r: 4, fill: C.old, visibility: 'hidden' });
    svg.append(cross, mD, mE);
    const near = (pts, lx) => pts.reduce((a, r) => Math.abs(Math.log10(r[0]) - lx) < Math.abs(Math.log10(a[0]) - lx) ? r : a);
    const hit = s('rect', { class: 'hit', x: L, y: T, width: W - L - R, height: Hp });
    hit.addEventListener('pointermove', e => {
      const b = svg.getBoundingClientRect();
      const px = (e.clientX - b.left) * (W / b.width);
      const lx = lo + (px - L) / (W - L - R) * (hi - lo);
      const pd = near(Dn, lx), pe = near(En, lx);
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
      const inD = lx <= Math.log10(endD[0]) + .05;
      mD.setAttribute('visibility', inD ? 'visible' : 'hidden'); mD.setAttribute('cx', x(pd[0])); mD.setAttribute('cy', y(pd[1]));
      mE.setAttribute('visibility', 'visible'); mE.setAttribute('cx', x(pe[0])); mE.setAttribute('cy', y(pe[1]));
      t.show(e, `<b>${(10 ** lx / 1e6).toFixed(1)}M steps</b><br>` +
        (inD ? `<span class="k">Scaled D</span> ${pct(pd[1])} <span class="k">(round ${Dn.indexOf(pd) + 1})</span><br>` : '') +
        `<span class="k">Earlier</span> ${pct(pe[1])} <span class="k">(round ${En.indexOf(pe) + 1})</span>`);
    });
    hit.addEventListener('pointerleave', () => { t.hide(); [cross, mD, mE].forEach(el => el.setAttribute('visibility', 'hidden')); });
    svg.append(hit);
    fig.append(legend([[C.new, 'Scaled D (current recipe)', 'ln'], [C.old, 'Earlier recipe with settings token', 'ln'], [C.new, 'selected checkpoint (round 71)', 'sq']]), scrollBox(svg));
  }

  /* ================= Figure 6: quality by round ================= */
  {
    const rows = D.learning.scaled_d;
    const series = [[1, 'Eventual', C.new], [2, 'Timely', C.c3], [4, 'Clean and timely', C.c2]];
    const W = 820, L = 52, R = 130, T = 12, B = 44, Hp = 250;
    const H = T + Hp + B;
    const xmax = Math.ceil(rows[rows.length - 1][0] / 5e6) * 5e6;
    const x = v => L + v / xmax * (W - L - R), y = v => T + (1 - v) * Hp;
    const svg = svgRoot(W, H, 'Scaled D selection-panel success by environment steps: eventual, timely, clean and timely');
    const fig = document.getElementById('fig-quality');
    const t = tipper(fig);
    for (const v of [0, .25, .5, .75, 1]) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(s('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end' }, `${v * 100}%`));
    }
    for (let v = 0; v <= xmax; v += 5e6) svg.append(s('text', { x: x(v), y: T + Hp + 18, 'text-anchor': 'middle' }, `${v / 1e6}M`));
    svg.append(s('text', { x: L + (W - L - R) / 2, y: H - 6, 'text-anchor': 'middle' }, 'environment steps'));
    const sx = x(D.learning.selected_step);
    svg.append(s('line', { class: 'ref', x1: sx, x2: sx, y1: T, y2: T + Hp }));
    svg.append(s('text', { x: sx + 6, y: T + 12 }, 'selected, round 71'));
    const last = rows[rows.length - 1];
    const labelY = [];
    for (const [k, lab, col] of series) {
      svg.append(s('polyline', { points: rows.map(r => `${x(r[0]).toFixed(1)},${y(r[k]).toFixed(1)}`).join(' '),
        fill: 'none', stroke: col, 'stroke-width': 2, 'stroke-linejoin': 'round' }));
      let ly = y(last[k]) + 4;
      for (const o of labelY) if (Math.abs(ly - o) < 14) ly = o + 14;
      labelY.push(ly);
      svg.append(s('text', { class: 'row-l', x: x(last[0]) + 8, y: ly }, lab));
    }
    const cross = s('line', { y1: T, y2: T + Hp, stroke: '#1a1d23', 'stroke-opacity': .25, visibility: 'hidden' });
    const marks = series.map(([, , col]) => s('circle', { r: 4, fill: col, visibility: 'hidden' }));
    svg.append(cross, ...marks);
    const hit = s('rect', { class: 'hit', x: L, y: T, width: W - L - R, height: Hp });
    hit.addEventListener('pointermove', e => {
      const b = svg.getBoundingClientRect();
      const v = ((e.clientX - b.left) * (W / b.width) - L) / (W - L - R) * xmax;
      const i = rows.reduce((a, r, j) => Math.abs(r[0] - v) < Math.abs(rows[a][0] - v) ? j : a, 0);
      const r = rows[i];
      cross.setAttribute('x1', x(r[0])); cross.setAttribute('x2', x(r[0])); cross.setAttribute('visibility', 'visible');
      series.forEach(([k], j) => { marks[j].setAttribute('cx', x(r[0])); marks[j].setAttribute('cy', y(r[k])); marks[j].setAttribute('visibility', 'visible'); });
      t.show(e, `<b>Round ${i}</b> · ${(r[0] / 1e6).toFixed(1)}M steps<br>` +
        series.map(([k, lab]) => `<span class="k">${lab}</span> ${pct(r[k])}`).join('<br>'));
    });
    hit.addEventListener('pointerleave', () => { t.hide(); [cross, ...marks].forEach(el => el.setAttribute('visibility', 'hidden')); });
    svg.append(hit);
    fig.append(legend(series.map(([, lab, col]) => [col, lab, 'ln'])), scrollBox(svg));
  }

  /* ================= Figure 7: lap times ================= */
  function histogram(id, { labels, counts, denom, title, label, marks = [] }) {
    const W = 400, L = 40, R = 8, T = 22, B = 40, Hp = 170;
    const H = T + Hp + B;
    const max = Math.max(...counts), n = counts.length;
    const bw = (W - L - R) / n;
    const y = v => T + (1 - v / max) * Hp;
    const svg = svgRoot(W, H, label, 300);
    const fig = document.getElementById(id);
    const t = tipper(fig);
    svg.append(s('text', { class: 'ttl', x: 0, y: 12 }, title));
    const step = max > 1000 ? 500 : 200;
    for (let v = 0; v <= max; v += step) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(s('text', { x: L - 6, y: y(v) + 4, 'text-anchor': 'end' }, v));
    }
    counts.forEach((c, i) => {
      const bx = L + i * bw;
      if (c) svg.append(s('rect', { x: bx + 1, y: y(c), width: Math.max(1, bw - 2), height: T + Hp - y(c), fill: C.new, rx: 1.5 }));
      const hit = s('rect', { class: 'hit', x: bx, y: T, width: bw, height: Hp });
      hover(hit, t, `<b>${labels[i]}</b><br>${c.toLocaleString()} laps · ${pct(c / denom)}`);
      svg.append(hit);
    });
    for (const [i, lab] of marks) svg.append(s('text', { x: L + (i + .5) * bw, y: T + Hp + 16, 'text-anchor': 'middle' }, lab));
    fig.append(scrollBox(svg));
    return { svg, x: i => L + i * bw, T, Hp };
  }
  {
    const lh = D.laps.hist, e = lh.bin_edges_s;
    const cut = 22;  // show 0-44 s
    const hist = histogram('fig-laps', {
      labels: lh.counts.slice(0, cut).map((_, i) => `${e[i]}–${e[i + 1]} s`), counts: lh.counts.slice(0, cut),
      denom: lh.denominator, title: `Lap time, ${lh.denominator.toLocaleString()} completed laps`,
      label: 'Histogram of successful lap times', marks: [0, 5, 10, 15, 20].map(i => [i - .5, `${e[i]}s`]),
    });
    const a = D.laps.all;
    for (const [v, lab, right] of [[a.median, `median ${a.median.toFixed(1)}s`, false], [a.p90, `p90 ${a.p90.toFixed(1)}s`, true]]) {
      const px = hist.x(0) + v / 2 * (hist.x(1) - hist.x(0));
      hist.svg.append(s('line', { x1: px, x2: px, y1: hist.T + 20, y2: hist.T + hist.Hp, stroke: C.c2, 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }));
      hist.svg.append(s('text', { class: 'val', x: px + (right ? 4 : -4), y: hist.T + 30, 'text-anchor': right ? 'start' : 'end' }, lab));
    }
    const rh = D.laps.ratio_hist, re = rh.bin_edges;
    histogram('fig-ratio', {
      labels: rh.counts.map((_, i) => `${re[i]}–${re[i + 1]}× reference`), counts: rh.counts, denom: rh.denominator,
      title: 'Lap time ÷ expert reference', label: 'Histogram of lap time relative to reference',
      marks: rh.counts.map((_, i) => [i, `${re[i]}`]),
    });
  }
  {
    const bo = D.laps.by_outcome;
    const rows = [['clean_timely', 'Clean and timely', C.new], ['missed_timely', 'Timely after a miss', C.light],
      ['clean_late', 'Clean but late', C.c3], ['missed_late', 'Late after a miss', C.c2]];
    const W = 820, L = 190, R = 70, rowH = 34, top = 6, xmax = 30;
    const H = top + rows.length * rowH + 30;
    const x = v => L + Math.min(v, xmax) / xmax * (W - L - R);
    const svg = svgRoot(W, H, 'Lap time 10th to 90th percentile and median by outcome class');
    const fig = document.getElementById('fig-bytype');
    const t = tipper(fig);
    const axisY = top + rows.length * rowH;
    for (let v = 0; v <= xmax; v += 5) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: x(v), x2: x(v), y1: top, y2: axisY }));
      svg.append(s('text', { x: x(v), y: axisY + 18, 'text-anchor': 'middle' }, `${v} s`));
    }
    rows.forEach(([k, lab, col], i) => {
      const r = bo[k], cy = top + i * rowH + rowH / 2;
      svg.append(s('text', { class: 'row-l', x: 0, y: cy + 4 }, lab));
      svg.append(s('line', { x1: x(r.p10), x2: x(r.p90), y1: cy, y2: cy, stroke: col, 'stroke-width': 6, 'stroke-linecap': 'round', 'stroke-opacity': .45 }));
      svg.append(s('circle', { cx: x(r.median), cy, r: 6, fill: col, stroke: '#fff', 'stroke-width': 2 }));
      svg.append(s('text', { class: 'val m', x: x(r.p90) + 10, y: cy + 4 }, `n=${r.count}`));
      const hit = s('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width: W, height: rowH });
      hover(hit, t, `<b>${lab}</b> · ${r.count.toLocaleString()} laps<br><span class="k">p10</span> ${r.p10.toFixed(1)} s · ` +
        `<span class="k">median</span> ${r.median.toFixed(1)} s · <span class="k">p90</span> ${r.p90.toFixed(1)} s<br><span class="k">max</span> ${r.maximum.toFixed(1)} s`);
      svg.append(hit);
    });
    fig.append(scrollBox(svg));
  }

  /* ================= Figure 8: speed command ================= */
  {
    const rows = D.speed;
    const series = [['sr', 'Completion', C.new], ['clean_timely', 'Clean and timely', C.c2]];
    const W = 820, L = 52, R = 130, T = 12, B = 44, Hp = 220;
    const H = T + Hp + B;
    const x = v => L + (v - 6) / 22 * (W - L - R), y = v => T + (1 - v) * Hp;
    const svg = svgRoot(W, H, 'Completion and clean-and-timely rate against commanded speed');
    const fig = document.getElementById('fig-speed');
    const t = tipper(fig);
    for (const v of [0, .25, .5, .75, 1]) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(s('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end' }, `${v * 100}%`));
    }
    for (const r of rows) svg.append(s('text', { x: x(r.command), y: T + Hp + 18, 'text-anchor': 'middle' }, `${r.command}`));
    svg.append(s('text', { x: L + (W - L - R) / 2, y: H - 6, 'text-anchor': 'middle' }, 'commanded speed, m/s'));
    svg.append(s('line', { class: 'ref', x1: x(16.5), x2: x(16.5), y1: T, y2: T + Hp }));
    svg.append(s('text', { x: x(16.5) + 6, y: T + Hp - 8 }, 'nominal'));
    svg.append(s('text', { x: x(23.5), y: T + 14, 'text-anchor': 'middle' }, '21 and 26: beyond the training range'));
    for (const [k, lab, col] of series) {
      svg.append(s('polyline', { points: rows.map(r => `${x(r.command)},${y(r[k])}`).join(' '), fill: 'none', stroke: col, 'stroke-width': 2 }));
      for (const r of rows) svg.append(s('circle', { cx: x(r.command), cy: y(r[k]), r: 5, fill: col, stroke: '#fff', 'stroke-width': 2 }));
      const lr = rows[rows.length - 1];
      svg.append(s('text', { class: 'row-l', x: x(lr.command) + 10, y: y(lr[k]) + (k === 'sr' ? -4 : 12) }, lab));
    }
    rows.forEach(r => {
      const hit = s('rect', { class: 'hit', x: x(r.command) - 30, y: T, width: 60, height: Hp });
      hover(hit, t, `<b>${r.command} m/s</b> · ${r.n} starts<br><span class="k">completion</span> ${pct(r.sr)}<br>` +
        `<span class="k">timely</span> ${pct(r.timely)}<br><span class="k">clean and timely</span> ${pct(r.clean_timely)}<br>` +
        `<span class="k">median lap</span> ${r.median_s.toFixed(1)} s<br><span class="k">paired lap ratio vs 16.5</span> ${r.paired_median_lap_ratio.toFixed(3)} (${r.paired_completers} pairs)`);
      svg.append(hit);
    });
    fig.append(legend(series.map(([, lab, col]) => [col, lab])), scrollBox(svg));
  }

  /* ================= Figure 9: recovery tails ================= */
  {
    const tail = D.tails.recovery_over_s, n = D.real100.episodes;
    const ks = Object.keys(tail).map(Number).sort((a, b) => a - b);
    const W = 820, L = 52, R = 16, T = 12, B = 44, Hp = 200;
    const H = T + Hp + B;
    const max = 450, bw = (W - L - R) / ks.length;
    const y = v => T + (1 - v / max) * Hp;
    const svg = svgRoot(W, H, 'Completed laps with a recovery longer than each duration');
    const fig = document.getElementById('fig-tails');
    const t = tipper(fig);
    for (let v = 0; v <= max; v += 100) {
      svg.append(s('line', { class: v ? 'ax' : 'ax0', x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      svg.append(s('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end' }, v));
    }
    ks.forEach((k, i) => {
      const v = tail[k], bx = L + i * bw + bw * .22, w = bw * .56;
      svg.append(s('rect', { x: bx, y: y(v), width: w, height: T + Hp - y(v), fill: k >= 8 ? C.c2 : C.new, rx: 3 }));
      svg.append(s('text', { class: 'val', x: bx + w / 2, y: y(v) - 20, 'text-anchor': 'middle' }, v));
      svg.append(s('text', { class: 'val m', x: bx + w / 2, y: y(v) - 6, 'text-anchor': 'middle' }, pct(v / n, 2)));
      svg.append(s('text', { x: bx + w / 2, y: T + Hp + 18, 'text-anchor': 'middle' }, `> ${k} s`));
      const hit = s('rect', { class: 'hit', x: L + i * bw, y: T, width: bw, height: Hp });
      hover(hit, t, `<b>Recovery longer than ${k} s</b><br>${v} completed laps<br>${pct(v / n, 2)} of all starts · ${pct(v / D.real100.completed, 2)} of completions`);
      svg.append(hit);
    });
    svg.append(s('text', { x: L + (W - L - R) / 2, y: H - 6, 'text-anchor': 'middle' }, 'longest resolved recovery in the lap'));
    fig.append(scrollBox(svg));
  }

  /* ================= Clips ================= */
  {
    const pick = ['Compound reversal 020-05', 'Slalom 028-029', 'Ordered 3D 042-003', 'Long braking 032-010',
      'Changing-radius turns 029-21', 'MultiGP Nautilus', 'Stacked reversal 016-033', 'Long low 058-007', 'Diving hairpin 039-011'];
    const grid = document.getElementById('clips');
    for (const name of pick) {
      const m = index.find(r => r.tag === 'clean' && r.name === name);
      if (!m) continue;
      const v = h('video', { muted: '', loop: '', playsinline: '', preload: 'none', poster: MEDIA + m.poster },
        h('source', { src: MEDIA + m.mp4, type: 'video/mp4' }));
      v.muted = true;
      grid.append(h('figure', {}, h('div', { class: 'frame' }, v),
        h('figcaption', {}, m.name, h('span', {}, `${m.time.toFixed(2)} s · ${m.gates} gates · course ${Math.round(m.course_sr * 32)}/32`))));
      window.__watchVideo && window.__watchVideo(v);
    }
  }

  /* ================= 3D explorers ================= */
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
      const still = h('div', { class: 'still', 'data-src': `${RDIR}${m.slug}.json` });
      const b = h('button', { class: 'tile', type: 'button', 'aria-pressed': 'false' },
        still, h('b', {}, m.name), h('span', {}, sub(m)));
      b.onclick = () => select(m.slug);
      lazy.observe(still);
      tiles.append(b);
      return [m.slug, b];
    });
    function select(slug) {
      buttons.forEach(([s2, b]) => b.setAttribute('aria-pressed', String(s2 === slug)));
      Scene3D.fetch(`${RDIR}${slug}.json`).then(d => viewer.load(d));
    }
    select(items[0].slug);
    return { select, el };
  }

  const clean = index.filter(m => m.tag === 'clean' && !m.name.startsWith('Flow'));
  const TAG = { recovery: 'timely recovery', late_recovery: 'late recovery', failure: 'crash' };
  const order = ['Go-around 017-040', 'Slalom 055-040', 'A2RL S2 2026'];
  const rec = index.filter(m => m.tag !== 'clean')
    .sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name) || (a.tag === 'failure') - (b.tag === 'failure'));
  explorer(document.querySelector('.explorer[data-set=clean]'), clean,
    m => `${m.time.toFixed(1)} s · ${m.gates} gates`);
  const recEx = explorer(document.querySelector('.explorer[data-set=recovery]'), rec,
    m => `${TAG[m.tag]} · ${m.time.toFixed(1)} s`);

  /* ================= recovery table ================= */
  {
    const table = document.getElementById('rtable');
    const cols = [['Episode'], ['Outcome'], ['Gate', 1], ['Miss to pass (s)', 1], ['Detour', 1], ['Extra path (m)', 1], ['Slowest (m/s)', 1], ['Course', 1]];
    table.append(h('thead', {}, h('tr', {}, ...cols.map(([c, n]) => h('th', n ? { class: 'n' } : {}, c)))));
    const tbody = h('tbody');
    for (const m of rec) for (const r of m.recoveries) {
      const tr = h('tr', { tabindex: '0' },
        h('td', {}, m.name), h('td', {}, h('span', { class: 'tag' }, r.passed ? TAG[m.tag] : 'crashed before retaking')),
        ...[r.gate, r.passed ? r.miss_to_pass.toFixed(2) : '–', r.passed ? `${r.detour.toFixed(2)}×` : '–',
          r.passed ? r.extra_path.toFixed(1) : '–', r.min_speed.toFixed(2), `${Math.round(m.course_sr * 32)}/32`]
          .map(v => h('td', { class: 'n' }, String(v))));
      const go = () => { recEx.select(m.slug); recEx.el.scrollIntoView({ behavior: 'smooth', block: 'start' }); };
      tr.onclick = go;
      tr.onkeydown = e => { if (e.key === 'Enter') go(); };
      tbody.append(tr);
    }
    table.append(tbody);
  }
})();
