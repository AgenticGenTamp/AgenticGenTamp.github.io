"""Pick cube_0 and carry it (tip at z) through a list of tip waypoints with fixed base heading.
usage: carry.py SEED HEADING_DEG Z VMAX 'x,y;x,y;...' [release]  -> prints termination/reward events."""
import sys, numpy as np
from env_client import make_env
import kin, rutil
from rutil import Bot
seed = int(sys.argv[1]); hd = np.deg2rad(float(sys.argv[2])); zc = float(sys.argv[3]); VMAX = float(sys.argv[4])
pts = [tuple(map(float, p.split(','))) for p in sys.argv[5].split(';')]
release = len(sys.argv) > 6
OFF = 0.55
env = make_env(); obs, _ = env.reset(seed=seed); b = Bot(env, obs)
names = sorted(n for n in obs.get_object_names() if n.startswith('cube'))
def step(a):
    b.obs, r, te, tr, inf = b.env.step(np.asarray(a, np.float32)); b.nsteps += 1
    if b.nsteps % 10 == 0 or te or r != -1:
        print(b.nsteps, 'r', r, 'te', te, 'base', np.round(b.base(), 3), 'cube', np.round(b.obj('cube_0'), 3), flush=True)
    if te: sys.exit()
    return r
b.step = step
rutil.pick(b, 'cube_0')
u = np.array([np.cos(hd), np.sin(hd)])
b.goto(base_t=[*b.base()[:2], hd], tol=0.01, max_steps=80)
q_hold, _, _ = kin.ik_multi(np.zeros(3), b.q(), [OFF, 0, zc], 'down')
q_high, _, _ = kin.ik_multi(np.zeros(3), q_hold, [OFF, 0, 0.15], 'down')
b.goto(q_t=q_high, tol=0.01, max_steps=100)
b.goto(base_t=[*(np.array(pts[0]) - OFF * u), hd], tol=0.01, max_steps=200)
b.goto(q_t=q_hold, tol=0.005, max_steps=100)
print('start', b.nsteps, np.round(b.obj('cube_0'), 3), flush=True)
for i in range(1, len(pts)):
    a0, a1 = np.array(pts[i - 1]), np.array(pts[i])
    n = int(np.ceil(np.linalg.norm(a1 - a0) / VMAX))
    for t in np.linspace(0, 1, n + 1)[1:]:
        tgt = a0 + t * (a1 - a0) - OFF * u
        for _ in range(3):
            b.act(q_t=q_hold, base_t=[*tgt, hd], grip=1.0)
            if np.linalg.norm(b.base()[:2] - tgt) < 0.01: break
    print('wp', pts[i], 'cube', np.round(b.obj('cube_0'), 3), b.nsteps, flush=True)
if release:
    b.wait(8, grip=0.0); b.goto(q_t=q_high, max_steps=60); b.wait(30)
print('end', b.nsteps, np.round(b.obj('cube_0'), 3))
