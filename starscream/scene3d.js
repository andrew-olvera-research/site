/* Scene3D: a tiny dependency-free 3D line renderer for Starscream rollouts.
 *
 * Draws in the same minimalist style as the clips: grey gate outlines, a blue
 * trail that grows over time, and a small quadrotor glyph. Drag to orbit,
 * scroll (after clicking the scene) or pinch to zoom, double-click to reset.
 *
 * Usage: <div class="scene3d" data-scene3d="assets/scene3d/demo.json"></div>
 * Data schema: see assets/scene3d/make_demo.py (metres, seconds, z up, quat = [w,x,y,z]).
 */
(() => {
  const STYLE = {
    bg: '#ffffff',
    gate: '#6b6b6b',
    trail: '#1f5a96',
    shadow: '#e3e6ea',
    grid: '#f0f1f3',
    ink: '#1a1d23',
    muted: '#6b7280',
    font: '13px Inter, system-ui, sans-serif',
  };
  const DRONE_ARM = 0.45;   // drawn arm length (m); exaggerated so the glyph reads at course scale
  const HOLD_END = 1.2;     // seconds to hold the finished trail before looping
  const IDLE_SPIN = 0.06;   // rad/s auto-orbit until the viewer takes control

  // ---------- math ----------
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = a => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
  const lerp = (a, b, f) => a.map((v, i) => v + (b[i] - v) * f);
  function qrot(q, v) {  // rotate v by unit quaternion q = [w,x,y,z]
    const [w, x, y, z] = q, u = [x, y, z];
    const t = cross(u, v).map(c => 2 * c);
    const c2 = cross(u, t);
    return [v[0] + w * t[0] + c2[0], v[1] + w * t[1] + c2[1], v[2] + w * t[2] + c2[2]];
  }
  function qslerp(a, b, f) {  // cheap nlerp is plenty at 20 Hz
    const s = dot(a.slice(1), b.slice(1)) + a[0] * b[0] < 0 ? -1 : 1;
    const q = a.map((v, i) => v + (s * b[i] - v) * f);
    const n = Math.hypot(...q);
    return q.map(v => v / n);
  }

  function gateCorners(g) {
    const h = g.size / 2, c = Math.cos(g.yaw), s = Math.sin(g.yaw);
    const side = [-s * h, c * h, 0];  // horizontal, perpendicular to the gate normal
    const up = [0, 0, h];
    return [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) =>
      [g.pos[0] + a * side[0] + b * up[0], g.pos[1] + a * side[1] + b * up[1], g.pos[2] + a * side[2] + b * up[2]]);
  }

  class Scene3D {
    constructor(el, data) {
      this.el = el;
      this.data = data;
      this.traj = data.trajectory;
      this.T = (this.traj.pos.length - 1) * this.traj.dt;
      this.gates = data.gates.map(gateCorners);

      // bounds -> orbit target + default distance
      const pts = [...this.traj.pos, ...this.gates.flat()];
      const lo = [0, 1, 2].map(i => Math.min(...pts.map(p => p[i])));
      const hi = [0, 1, 2].map(i => Math.max(...pts.map(p => p[i])));
      this.center = lo.map((v, i) => (v + hi[i]) / 2);
      this.radius = Math.hypot(...sub(hi, lo)) / 2;
      this.ground = Math.min(0, lo[2]);
      this.home = { az: -2.2, el: 0.62, dist: this.radius * 2.2, target: this.center.slice() };
      this.view = { ...this.home, target: this.home.target.slice() };

      this.t = 0;
      this.playing = true;
      this.touched = false;  // stop idle spin once the viewer interacts
      this.visible = false;

      this.build();
      this.bind();
      this.resize();
      requestAnimationFrame(this.frame.bind(this));
    }

    build() {
      this.el.innerHTML = '';
      this.canvas = document.createElement('canvas');
      this.canvas.setAttribute('aria-label', `${this.data.name}: interactive 3D view. Drag to orbit.`);
      this.canvas.tabIndex = 0;
      this.ctx = this.canvas.getContext('2d');

      const bar = document.createElement('div');
      bar.className = 'scene3d-bar';
      this.btn = document.createElement('button');
      this.btn.type = 'button';
      this.scrub = document.createElement('input');
      Object.assign(this.scrub, { type: 'range', min: 0, max: 1000, value: 0 });
      this.scrub.setAttribute('aria-label', 'Time');
      const reset = document.createElement('button');
      reset.type = 'button';
      reset.textContent = 'Reset view';
      reset.onclick = () => { this.view = { ...this.home, target: this.home.target.slice() }; this.dirty = true; };
      bar.append(this.btn, this.scrub, reset);
      this.el.append(this.canvas, bar);
      this.syncBtn();
    }

    syncBtn() { this.btn.textContent = this.playing ? 'Pause' : 'Play'; }

    bind() {
      const c = this.canvas, ptrs = new Map();
      let pinch = 0, armed = false;
      const take = () => { this.touched = true; armed = true; this.dirty = true; };

      c.addEventListener('pointerdown', e => {
        c.setPointerCapture(e.pointerId);
        ptrs.set(e.pointerId, [e.clientX, e.clientY]);
        take();
      });
      c.addEventListener('pointermove', e => {
        if (!ptrs.has(e.pointerId)) return;
        const [px, py] = ptrs.get(e.pointerId);
        ptrs.set(e.pointerId, [e.clientX, e.clientY]);
        if (ptrs.size === 2) {
          const [a, b] = [...ptrs.values()];
          const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
          if (pinch) this.zoom(pinch / d);
          pinch = d;
          return;
        }
        const dx = e.clientX - px, dy = e.clientY - py;
        if (e.shiftKey || e.buttons === 2) this.pan(dx, dy);
        else {
          this.view.az -= dx * 0.008;
          this.view.el = Math.max(-0.1, Math.min(1.5, this.view.el + dy * 0.008));
        }
        this.dirty = true;
      });
      const up = e => { ptrs.delete(e.pointerId); pinch = 0; };
      c.addEventListener('pointerup', up);
      c.addEventListener('pointercancel', up);
      c.addEventListener('contextmenu', e => e.preventDefault());
      // Wheel only zooms once the scene has been clicked, so page scrolling isn't hijacked.
      c.addEventListener('wheel', e => {
        if (!armed) return;
        e.preventDefault();
        this.zoom(Math.exp(e.deltaY * 0.0015));
      }, { passive: false });
      c.addEventListener('pointerleave', () => { if (!ptrs.size) armed = false; });
      c.addEventListener('dblclick', () => { this.view = { ...this.home, target: this.home.target.slice() }; this.dirty = true; });
      c.addEventListener('keydown', e => {
        if (e.key === ' ') { e.preventDefault(); this.toggle(); }
        if (e.key === 'ArrowLeft') { this.view.az += 0.1; take(); }
        if (e.key === 'ArrowRight') { this.view.az -= 0.1; take(); }
      });

      this.btn.onclick = () => this.toggle();
      this.scrub.addEventListener('input', () => {
        this.t = (this.scrub.value / 1000) * this.T;
        this.playing = false; this.syncBtn(); this.dirty = true;
      });

      new ResizeObserver(() => this.resize()).observe(this.el);
      new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; }, { threshold: 0.1 }).observe(this.el);
    }

    toggle() {
      if (!this.playing && this.t >= this.T) this.t = 0;
      this.playing = !this.playing; this.syncBtn(); this.dirty = true;
    }

    zoom(f) {
      this.view.dist = Math.max(this.radius * 0.6, Math.min(this.radius * 8, this.view.dist * f));
      this.dirty = true;
    }

    pan(dx, dy) {
      const { right, up } = this.basis();
      const k = this.view.dist / this.focal;
      this.view.target = this.view.target.map((v, i) => v - (right[i] * dx - up[i] * dy) * k);
    }

    resize() {
      const r = this.canvas.getBoundingClientRect();
      const dpr = Math.min(devicePixelRatio || 1, 2);
      this.w = r.width; this.h = r.height;
      this.canvas.width = Math.round(r.width * dpr);
      this.canvas.height = Math.round(r.height * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.focal = Math.min(this.w, this.h * 1.6) * 0.9;
      this.dirty = true;
    }

    // ---------- camera ----------
    basis() {
      const { az, el, dist, target } = this.view;
      const eye = [target[0] + dist * Math.cos(el) * Math.cos(az),
                   target[1] + dist * Math.cos(el) * Math.sin(az),
                   target[2] + dist * Math.sin(el)];
      const fwd = unit(sub(target, eye));
      const right = unit(cross(fwd, [0, 0, 1]));
      const up = cross(right, fwd);
      return { eye, fwd, right, up };
    }

    project(p) {
      const d = sub(p, this.cam.eye);
      const z = dot(d, this.cam.fwd);
      if (z < 0.05) return null;
      return [this.w / 2 + (dot(d, this.cam.right) / z) * this.focal,
              this.h / 2 - (dot(d, this.cam.up) / z) * this.focal];
    }

    polyline(pts, color, width, closed = false) {
      const ctx = this.ctx;
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath();
      let pen = false;
      for (const p of closed ? [...pts, pts[0]] : pts) {
        const s = this.project(p);
        if (!s) { pen = false; continue; }
        pen ? ctx.lineTo(s[0], s[1]) : ctx.moveTo(s[0], s[1]);
        pen = true;
      }
      ctx.stroke();
    }

    // ---------- state at time t ----------
    sample(t) {
      const { pos, quat, dt } = this.traj;
      const x = Math.min(t / dt, pos.length - 1), i = Math.floor(x), f = x - i;
      const j = Math.min(i + 1, pos.length - 1);
      return { i, pos: lerp(pos[i], pos[j], f), quat: qslerp(quat[i], quat[j], f) };
    }

    // ---------- draw ----------
    draw() {
      const ctx = this.ctx, { w, h } = this;
      this.cam = this.basis();
      ctx.fillStyle = STYLE.bg;
      ctx.fillRect(0, 0, w, h);
      ctx.lineJoin = ctx.lineCap = 'round';

      // faint ground grid
      const g = this.ground, R = Math.ceil(this.radius * 1.15 / 2) * 2, cx = Math.round(this.center[0]), cy = Math.round(this.center[1]);
      for (let k = -R; k <= R; k += 2) {
        this.polyline([[cx + k, cy - R, g], [cx + k, cy + R, g]], STYLE.grid, 1);
        this.polyline([[cx - R, cy + k, g], [cx + R, cy + k, g]], STYLE.grid, 1);
      }

      const s = this.sample(this.t);
      const trail = [...this.traj.pos.slice(0, s.i + 1), s.pos];

      // ground shadow of the trail: cheap depth cue
      this.polyline(trail.map(p => [p[0], p[1], g]), STYLE.shadow, 1.4);

      for (const c of this.gates) this.polyline(c, STYLE.gate, 1.3, true);
      this.polyline(trail, STYLE.trail, 1.8);
      this.drawDrone(s);

      ctx.fillStyle = STYLE.ink; ctx.font = STYLE.font; ctx.textBaseline = 'top';
      ctx.fillText(this.data.name, 14, 12);
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${Math.min(this.t, this.T).toFixed(2)} s`, 14, h - 12);
      if (!this.touched) {
        ctx.fillStyle = STYLE.muted; ctx.textAlign = 'right';
        ctx.fillText('drag to orbit', w - 14, h - 12);
        ctx.textAlign = 'left';
      }
    }

    drawDrone({ pos, quat }) {
      const ctx = this.ctx, a = DRONE_ARM, r = a * 0.42;
      const body = v => { const q = qrot(quat, v); return [pos[0] + q[0], pos[1] + q[1], pos[2] + q[2]]; };
      const tips = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([x, y]) => [x * a * Math.SQRT1_2, y * a * Math.SQRT1_2, 0]);
      ctx.fillStyle = STYLE.bg;
      this.polyline([body(tips[0]), body(tips[2])], STYLE.ink, 1.5);
      this.polyline([body(tips[1]), body(tips[3])], STYLE.ink, 1.5);
      for (const t of tips) {
        const ring = [];
        for (let k = 0; k < 16; k++) {
          const th = (k / 16) * Math.PI * 2;
          ring.push(body([t[0] + r * Math.cos(th), t[1] + r * Math.sin(th), 0]));
        }
        this.polyline(ring, STYLE.ink, 1.1, true);
      }
      // short thrust tick so attitude reads at a glance
      this.polyline([pos, body([0, 0, a * 0.7])], STYLE.ink, 1.1);
    }

    frame(now) {
      const dt = this.last ? Math.min((now - this.last) / 1000, 0.1) : 0;
      this.last = now;
      if (this.visible) {
        if (this.playing) {
          this.t += dt;
          if (this.t > this.T + HOLD_END) this.t = 0;
          this.scrub.value = Math.round(Math.min(this.t / this.T, 1) * 1000);
          this.dirty = true;
        }
        if (!this.touched) { this.view.az += IDLE_SPIN * dt; this.dirty = true; }
        if (this.dirty) { this.draw(); this.dirty = false; }
      }
      requestAnimationFrame(this.frame.bind(this));
    }
  }

  async function mount(el) {
    try {
      const res = await fetch(el.dataset.scene3d);
      if (!res.ok) throw new Error(res.status);
      el.scene3d = new Scene3D(el, await res.json());
    } catch (err) {
      el.textContent = 'Could not load 3D scene.';
      console.error('scene3d', err);
    }
  }

  window.Scene3D = Scene3D;
  document.querySelectorAll('[data-scene3d]').forEach(mount);
})();
