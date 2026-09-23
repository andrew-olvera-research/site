"""Writes demo.json: a small placeholder course + a smooth flight through it.

Placeholder only, so the 3D viewer can be styled before real rollouts are exported.
Schema (all metres / seconds / radians, z up):
  name        str
  gates       [{pos: [x,y,z], yaw: rad, size: m}]   gate normal = (cos yaw, sin yaw, 0), size = inner side length
  trajectory  {dt: s, pos: [[x,y,z], ...], quat: [[w,x,y,z], ...], passed: [int, ...]}
Pure stdlib: python make_demo.py
"""
import json, math

GATES = [(0, 0, 2.0), (5, 2.5, 3.0), (8, 8, 4.0), (2.5, 10.5, 3.0), (-4, 8, 1.8), (-6.5, 2, 3.2), (-3, -3.5, 2.5)]
SPEED = 7.0   # mean speed, m/s
DT = 0.05
G = 9.81

def add(a, b): return [x + y for x, y in zip(a, b)]
def sub(a, b): return [x - y for x, y in zip(a, b)]
def mul(a, s): return [x * s for x in a]
def norm(a): return math.sqrt(sum(x * x for x in a))
def unit(a): n = norm(a) or 1.0; return [x / n for x in a]
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def catmull(p0, p1, p2, p3, t):
    # uniform Catmull-Rom, closed loop
    return [0.5 * ((2*b) + (-a+c)*t + (2*a-5*b+4*c-d)*t*t + (-a+3*b-3*c+d)*t**3)
            for a, b, c, d in zip(p0, p1, p2, p3)]

n = len(GATES)
dense = []  # (point, segment index)
for i in range(n):
    p = [GATES[(i + k - 1) % n] for k in range(4)]
    for s in range(400):
        dense.append((catmull(*p, s / 400), i))
dense.append((dense[0][0], n))

# resample by arc length at constant speed -> positions every DT
arc = [0.0]
for (a, _), (b, _) in zip(dense, dense[1:]):
    arc.append(arc[-1] + norm(sub(b, a)))
total = arc[-1]
steps = int(total / SPEED / DT)
pos, seg, j = [], [], 0
for k in range(steps + 1):
    s = total * k / steps
    while j < len(arc) - 2 and arc[j + 1] < s: j += 1
    f = (s - arc[j]) / max(arc[j + 1] - arc[j], 1e-9)
    pos.append(add(dense[j][0], mul(sub(dense[j + 1][0], dense[j][0]), f)))
    seg.append(dense[j][1])

def at(i): return pos[max(0, min(len(pos) - 1, i))]

quat, passed = [], []
for i in range(len(pos)):
    vel = mul(sub(at(i + 1), at(i - 1)), 1 / (2 * DT))
    acc = mul(add(sub(at(i + 1), mul(at(i), 2)), at(i - 1)), 1 / DT**2)
    zb = unit(add(acc, [0, 0, G]))                  # thrust axis
    xc = unit([vel[0], vel[1], 0])                  # heading = velocity direction
    yb = unit(cross(zb, xc)); xb = cross(yb, zb)
    m = [[xb[0], yb[0], zb[0]], [xb[1], yb[1], zb[1]], [xb[2], yb[2], zb[2]]]
    w = math.sqrt(max(0, 1 + m[0][0] + m[1][1] + m[2][2])) / 2
    quat.append([round(v, 4) for v in (w, (m[2][1]-m[1][2]) / (4*w), (m[0][2]-m[2][0]) / (4*w), (m[1][0]-m[0][1]) / (4*w))])
    passed.append(min(seg[i] + (1 if i == len(pos) - 1 else 0), n))

gates = []
for i, g in enumerate(GATES):
    t = sub(GATES[(i + 1) % n], GATES[(i - 1) % n])  # Catmull-Rom tangent at the knot
    gates.append({"pos": list(g), "yaw": round(math.atan2(t[1], t[0]), 4), "size": 1.5})

out = {"name": "Demo course", "gates": gates,
       "trajectory": {"dt": DT, "pos": [[round(v, 3) for v in p] for p in pos], "quat": quat, "passed": passed}}
with open(__file__.replace("make_demo.py", "demo.json"), "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"{len(pos)} frames, {total:.1f} m, {len(pos) * DT:.1f} s")
