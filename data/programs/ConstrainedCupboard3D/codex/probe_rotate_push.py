"""Probe yaw-rotated versions of the known floor-contact pose."""
import argparse
import numpy as np
from env_client import make_env

Q = np.array([-3.0175, -2.3524, -1.1879, -.314, -1.5966, .6707, -2.7101])


def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi


ap = argparse.ArgumentParser()
ap.add_argument("yaw", type=float)
ap.add_argument("--q1", type=float, default=0.)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--travel", type=float, default=.7)
ap.add_argument("--dx", type=float, default=0.)
ap.add_argument("--dy", type=float, default=0.)
ap.add_argument("--track", action="store_true")
ap.add_argument("--count", type=int, default=None)
args = ap.parse_args()
env = make_env()
options = None if args.count is None else {"object_count": args.count}
s, _ = env.reset(seed=args.seed, options=options)
r_total = 0.0
r_best = -1e9
r = s.get_object_from_name("robot"); rod = s.get_object_from_name("cuboid_0")
def v(o, f): return float(s.get(o, f))
def pos(o): return np.array([v(o, f) for f in ("x", "y", "z")])
def go(base, yaw, q, grip=1.):
    global s, r_total, r_best
    a = np.zeros(11, np.float32)
    b = np.array([v(r, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")])
    e = base-b[:2]; a[:2] = np.clip(e/.87, -.1, .1)
    a[2] = np.clip(wrap(yaw-b[2]), -.1, .1)
    qc = np.array([v(r, f"pos_arm_joint{i}") for i in range(1, 8)])
    a[3:10] = np.clip(.5*wrap(q-qc), -.1, .1); a[-1] = grip
    s, rew, term, trunc, inf = env.step(a)
    r_total += float(rew)
    r_best = max(r_best, float(rew))
    return rew
p0 = pos(rod); q = Q.copy(); q[0] = wrap(q[0] + args.q1)
# Known yaw-zero base-to-contact vector is rod-base=(.8,-.1). Rotate it.
c, sn = np.cos(args.yaw), np.sin(args.yaw)
off = np.array([c*.8 + sn*.1, sn*.8 - c*.1])
contact = p0[:2] - off + np.array([args.dx, args.dy])
# Configure arm while safely displaced opposite desired world-x push.
safe = contact - np.array([.6, 0.])
for _ in range(120): go(safe, args.yaw, q)
for _ in range(35): go(contact, args.yaw, q)
pt = pos(rod)
for _ in range(45):
    target = contact + np.array([args.travel, 0.])
    if args.track:
        target[1] = pos(rod)[1] - off[1] + args.dy
    go(target, args.yaw, q)
pf = pos(rod)
print("yaw", args.yaw, "q1", args.q1, "p0", np.round(p0,4),
      "touch", np.round(pt,4), "final", np.round(pf,4),
      "touch_d", np.round(pt-p0,4), "push_d", np.round(pf-pt,4),
      "reward", round(r_total, 4), "best", round(r_best, 4),
      "base", np.round([v(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")],4))
env.close()
