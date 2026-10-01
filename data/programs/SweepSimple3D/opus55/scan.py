"""Carry a held cube low over the floor along a lawnmower path; log reward/termination.
usage: python scan.py SEED HEADING_DEG X0 X1 Y0 Y1 [lines=x|y] [spacing] [z] [tag]
(X/Y ranges are for the held cube / tip position in world frame)."""
import sys, json, numpy as np
from env_client import make_env
import kin, rutil
from rutil import Bot

seed = int(sys.argv[1]); hd = np.deg2rad(float(sys.argv[2]))
x0, x1, y0, y1 = map(float, sys.argv[3:7])
lines = sys.argv[7] if len(sys.argv) > 7 else 'y'
sp = float(sys.argv[8]) if len(sys.argv) > 8 else 0.12
zc = float(sys.argv[9]) if len(sys.argv) > 9 else 0.025
tag = sys.argv[10] if len(sys.argv) > 10 else 'scan'
OFF = 0.55
VMAX = 0.08  # m/step
log = []
env = make_env(); obs, info = env.reset(seed=seed); b = Bot(env, obs)
cube = 'cube_0'
state = {'term': False, 'trunc': False}

def step(a):
    a = np.asarray(a, np.float32)
    b.obs, r, te, tr, inf = b.env.step(a)
    b.nsteps += 1
    c = b.obj(cube); bs = b.base()
    log.append([b.nsteps, *np.round(bs, 3).tolist(), *np.round(c, 3).tolist(), r, te, tr])
    if r != -1.0 or te or tr or set(inf) - {'object_count'}:
        print('SIGNAL', b.nsteps, 'r', r, 'term', te, 'trunc', tr, 'cube', np.round(c, 3), 'base', np.round(bs, 3), inf, flush=True)
    state['term'] |= te; state['trunc'] |= tr
    return r
b.step = step

ok = rutil.pick(b, cube)
print('picked', ok, b.nsteps, np.round(b.obj(cube), 3), flush=True)
if not ok:
    sys.exit(1)

def wp_list():
    pts = []
    if lines == 'y':  # lines of constant x, sweeping along y
        xs = np.arange(x0, x1 + 1e-6, sp)
        for i, x in enumerate(xs):
            ys = (y0, y1) if i % 2 == 0 else (y1, y0)
            pts += [(x, ys[0]), (x, ys[1])]
    else:
        ys = np.arange(y0, y1 + 1e-6, sp)
        for i, y in enumerate(ys):
            xs = (x0, x1) if i % 2 == 0 else (x1, x0)
            pts += [(xs[0], y), (xs[1], y)]
    return pts

u = np.array([np.cos(hd), np.sin(hd)])
def base_for(tip):
    return np.array(tip) - OFF * u

# rotate base to heading with cube lifted, then go to the first waypoint base pose and lower
pts = wp_list()
p0 = base_for(pts[0])
b.goto(base_t=[b.base()[0], b.base()[1], hd], tol=0.01, max_steps=80)
q_hold, okk, e = kin.ik_multi(np.array([0, 0, 0.0]), b.q(), [OFF, 0, zc], 'down', yaw=None)
q_high, _, _ = kin.ik_multi(np.array([0, 0, 0.0]), q_hold, [OFF, 0, 0.12], 'down', yaw=None)
b.goto(q_t=q_high, tol=0.01, max_steps=100)

def drive(target, tol=0.02, maxst=200):
    """drive base (heading hd) toward target xy with capped speed; returns reached."""
    best = 1e9; stall = 0
    for k in range(maxst):
        bs = b.base(); d = np.array(target) - bs[:2]; dist = np.linalg.norm(d)
        if dist < tol:
            return True
        v = d if dist < VMAX else d / dist * VMAX
        a = np.zeros(11, np.float32)
        a[0:2] = np.clip(v / 0.87, -0.1, 0.1)
        a[2] = np.clip(((hd - bs[2] + np.pi) % (2 * np.pi)) - np.pi, -0.1, 0.1)
        dq = kin.wrap_to(q_cur[0], b.q()) - b.q(); a[3:10] = np.clip(1.5 * dq, -0.1, 0.1)
        a[10] = 1.0
        b.step(a)
        if state['term'] or state['trunc']:
            return False
        if dist < best - 0.005:
            best = dist; stall = 0
        else:
            stall += 1
            if stall > 15:
                print('stuck at base', np.round(b.base(), 3), 'target', np.round(target, 3), flush=True)
                return False
    return False

q_cur = [q_high]
drive(p0, tol=0.02)
q_cur[0] = q_hold
b.goto(q_t=q_hold, tol=0.01, max_steps=100)
print('lowered', np.round(b.obj(cube), 3), 'tip', np.round(b.tip()[0], 3), 'step', b.nsteps, flush=True)
for i in range(1, len(pts)):
    a0, a1 = np.array(pts[i - 1]), np.array(pts[i])
    n = max(1, int(np.ceil(np.linalg.norm(a1 - a0) / 0.1)))
    for t in np.linspace(0, 1, n + 1)[1:]:
        drive(base_for(a0 + t * (a1 - a0)), tol=0.03, maxst=40)
        if state['term'] or state['trunc']:
            break
    if state['term'] or state['trunc']:
        break
c = b.obj(cube)
print('done step', b.nsteps, 'cube', np.round(c, 3), 'term', state, flush=True)
json.dump(log, open('logs/%s_s%d.json' % (tag, seed), 'w'))
