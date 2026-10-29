/* Results charts and in-view video playback for the Starscream page.
 * Data: assets/results/results-2026-10-29.json (exported from the Oct 29 evaluation bundle). */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const C = { ink: '#1a1d23', ink2: '#3b414b', muted: '#58616f', line: '#e5e7eb', line2: '#d1d5db', dot: '#1f5a96' };
  const BAND = { id: 'In-distribution', near: 'Near', outer: 'Outer' };
  const FAMILY = f => f.replace('_', ' ');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const text = (parent, x, y, str, attrs = {}) => {
    const t = el('text', { x, y, fill: C.muted, 'font-size': 12, ...attrs }, parent);
    t.textContent = str;
    return t;
  };

  function tooltip(host) {
    const tip = document.createElement('div');
    tip.className = 'tip';
    host.appendChild(tip);
    return {
      show(html, x, y) {
        tip.innerHTML = html; tip.style.display = 'block';
        const w = tip.offsetWidth, hw = host.clientWidth;
        tip.style.left = Math.min(Math.max(0, x - w / 2), hw - w) + 'px';
        tip.style.top = (y - tip.offsetHeight - 12) + 'px';
      },
      hide() { tip.style.display = 'none'; },
    };
  }

  function bindDot(dot, tip, html, cx, cy) {
    dot.setAttribute('tabindex', '0');
    dot.setAttribute('role', 'img');
    dot.setAttribute('aria-label', html.replace(/<[^>]+>/g, ' '));
    const on = () => tip.show(html, cx, cy), off = () => tip.hide();
    dot.addEventListener('mouseenter', on); dot.addEventListener('focus', on);
    dot.addEventListener('mouseleave', off); dot.addEventListener('blur', off);
  }

  // Clean laps (of 9) for each of the 80 held-out courses against geometric distance.
  function corpusChart(host, courses) {
    const W = host.clientWidth, narrow = W < 560;
    const m = { l: narrow ? 34 : 44, r: 12, t: 40, b: 46 }, H = narrow ? 260 : 300;
    const x0 = 0.25, x1 = 1.4, X = d => m.l + (d - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = c => m.t + (1 - c / 9) * (H - m.t - m.b);
    host.querySelector('svg')?.remove();
    const svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    host.prepend(svg);
    const bands = [['id', x0, 1], ['near', 1, 1.25], ['outer', 1.25, x1]];
    const shade = { id: '#ffffff', near: '#f3f5f8', outer: '#e9edf2' };
    for (const [b, a, z] of bands) {
      el('rect', { x: X(a), y: m.t - 32, width: X(z) - X(a), height: H - m.b - m.t + 32, fill: shade[b] }, svg);
      text(svg, X(a) + 6, m.t - 16, BAND[b], { fill: C.ink2, 'font-weight': 600, 'font-size': narrow ? 11 : 12 });
    }
    for (const c of [0, 3, 6, 9]) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(c), y2: Y(c), stroke: c ? C.line : C.line2 }, svg);
      text(svg, m.l - 8, Y(c) + 4, String(c), { 'text-anchor': 'end' });
    }
    for (const d of [0.25, 0.5, 0.75, 1, 1.25]) text(svg, X(d), H - m.b + 18, d.toFixed(2), { 'text-anchor': 'middle' });
    text(svg, (m.l + W - m.r) / 2, H - 8, narrow ? 'Distance from training (1.0 = envelope edge)' : 'Distance from nearest training course (1.0 = edge of training envelope)', { 'text-anchor': 'middle', 'font-size': narrow ? 11 : 12 });
    const tip = tooltip(host);
    for (const c of courses) {
      const cx = X(c.distance), cy = Y(c.clean);
      const dot = el('circle', { cx, cy, r: 5, fill: C.dot, stroke: '#fff', 'stroke-width': 2, class: 'dot' }, svg);
      bindDot(dot, tip, `<b>${c.course.toUpperCase()}</b> · ${FAMILY(c.family)} · ${c.gates} gates<br>${c.clean}/9 clean · distance ${c.distance.toFixed(2)}`, cx, cy);
    }
  }

  // Per-course lap-time penalty without JEPA alignment, same course and noise, both laps clean.
  function jepaChart(host, jepa) {
    const W = host.clientWidth, narrow = W < 560;
    const m = { l: narrow ? 78 : 118, r: narrow ? 16 : 150, t: 14, b: 44 }, row = 62, H = m.t + row * 3 + m.b;
    const x0 = -0.2, x1 = 0.5, X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    host.querySelector('svg')?.remove();
    const svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    host.prepend(svg);
    for (const v of [-0.2, 0, 0.2, 0.4]) {
      el('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: v === 0 ? C.muted : C.line }, svg);
      text(svg, X(v), H - m.b + 18, (v > 0 ? '+' : '') + Math.round(v * 100) + '%', { 'text-anchor': 'middle' });
    }
    text(svg, (m.l + W - m.r) / 2, H - 8, 'Lap time without JEPA, relative to the full stack', { 'text-anchor': 'middle', 'font-size': narrow ? 11 : 12 });
    const tip = tooltip(host);
    ['id', 'near', 'outer'].forEach((b, i) => {
      const cy = m.t + row * i + row / 2, d = jepa[b];
      text(svg, 0, cy - 4, narrow ? BAND[b].replace('In-distribution', 'In-dist.') : BAND[b], { fill: C.ink, 'font-weight': 600 });
      text(svg, 0, cy + 12, `${d.per_course.length} courses`, { 'font-size': 11 });
      const sorted = [...d.per_course].sort((a, z) => a.slower_fraction - z.slower_fraction);
      sorted.forEach((c, k) => {
        const cx = X(c.slower_fraction), off = ((k % 3) - 1) * 9;
        const dot = el('circle', { cx, cy: cy + off, r: 4.5, fill: C.dot, 'fill-opacity': 0.85, stroke: '#fff', 'stroke-width': 1.5, class: 'dot' }, svg);
        bindDot(dot, tip, `<b>${c.course.toUpperCase()}</b><br>${c.slower_fraction > 0 ? '+' : ''}${(c.slower_fraction * 100).toFixed(1)}% lap time without JEPA`, cx, cy + off);
      });
      const mx = X(d.slower_fraction);
      el('line', { x1: mx, x2: mx, y1: cy - 18, y2: cy + 18, stroke: C.ink, 'stroke-width': 2 }, svg);
      if (narrow) text(svg, mx + 4, cy - 22, `avg +${(d.slower_fraction * 100).toFixed(0)}%`, { fill: C.ink, 'font-weight': 600, 'font-size': 11 });
      else {
        text(svg, W - m.r + 16, cy - 2, `avg +${(d.slower_fraction * 100).toFixed(0)}%`, { fill: C.ink, 'font-weight': 600 });
        text(svg, W - m.r + 16, cy + 14, `+${d.extra_seconds.toFixed(2)} s per lap`, { 'font-size': 11.5 });
      }
    });
  }

  async function charts() {
    const corpus = document.getElementById('corpus-chart'), jepa = document.getElementById('jepa-chart');
    if (!corpus && !jepa) return;
    try {
      const r = await fetch('assets/results/results-2026-10-29.json');
      if (!r.ok) throw Error('results unavailable');
      const data = await r.json();
      const draw = () => { if (corpus) corpusChart(corpus, data.courses); if (jepa) jepaChart(jepa, data.jepa); };
      draw();
      let w = window.innerWidth;
      window.addEventListener('resize', () => { if (window.innerWidth !== w) { w = window.innerWidth; draw(); } });
    } catch (e) {
      for (const h of [corpus, jepa]) if (h) h.textContent = 'Chart data could not load.';
      console.error(e);
    }
  }

  // Play flight videos while they are on screen; leave them paused for reduced motion.
  function inView() {
    if (reduced || !('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver(entries => {
      for (const e of entries) {
        const v = e.target;
        if (e.isIntersecting && !v.dataset.userPaused) v.play().catch(() => {});
        else if (!e.isIntersecting && !v.paused) { v.dataset.autoPause = '1'; v.pause(); }
      }
    }, { threshold: 0.5 });
    document.querySelectorAll('video[data-inview]').forEach(v => {
      // A pause the viewer makes sticks; one we make on scroll-out does not.
      v.addEventListener('pause', () => { if (v.dataset.autoPause) delete v.dataset.autoPause; else v.dataset.userPaused = '1'; });
      v.addEventListener('play', () => { delete v.dataset.userPaused; });
      io.observe(v);
    });
  }

  // Interactive replay of the held-out flights (recorded 130 Hz states, drawn with Scene3D).
  function flights() {
    const host = document.getElementById('flight-viewer');
    if (!host || !window.Scene3D) return;
    const status = document.getElementById('replay-status');
    const tiles = [...document.querySelectorAll('[data-flight]')];
    const fail = e => { status.textContent = 'The 3D viewer could not load. Each flight is still available as a video.'; console.error(e); };
    const viewer = new Scene3D(host);
    let request = 0;
    async function select(btn) {
      const current = ++request, id = btn.dataset.flight;
      status.textContent = 'Loading flight…';
      const data = await Scene3D.fetch(`assets/flights/${id}.json`);
      if (current !== request) return;
      const [info] = btn.querySelectorAll(':scope > span:last-of-type');
      data.name = id.toUpperCase();
      viewer.load(data);
      document.getElementById('course-title').textContent = id.toUpperCase();
      document.getElementById('course-info').textContent = info.textContent;
      tiles.forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
      status.textContent = '';
    }
    tiles.forEach(b => b.addEventListener('click', () => select(b).catch(fail)));
    select(tiles[0]).catch(fail);
  }

  charts();
  flights();
  inView();
})();
