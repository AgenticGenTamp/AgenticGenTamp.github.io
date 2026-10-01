"""5-cube seed: place every cube into the island-north region; report termination."""
import sys, numpy as np
from env_client import make_env
import rutil
from rutil import Bot
seed = int(sys.argv[1])
env = make_env(); obs, _ = env.reset(seed=seed); b = Bot(env, obs)
names = sorted(n for n in obs.get_object_names() if n.startswith('cube'))
def step(a):
    b.obs, r, te, tr, inf = b.env.step(np.asarray(a, np.float32)); b.nsteps += 1
    if te or r != -1:
        print('SIG', b.nsteps, r, te, inf, {n: np.round(b.obj(n), 3).tolist() for n in names}, flush=True)
    if te: sys.exit()
    return r
b.step = step
spots = [(0.25, 1.15), (0.45, 1.15), (0.65, 1.15), (0.35, 1.32), (0.6, 1.32)]
for n, xy in zip(names, spots):
    c = b.obj(n)
    b.goto(base_t=[c[0], 1.6, -np.pi / 2], tol=0.02, max_steps=200)
    ok = rutil.pick(b, n)
    b.goto(base_t=[xy[0], 1.75, -np.pi / 2], tol=0.02, max_steps=200)
    f = rutil.place(b, n, xy)
    b.goto(base_t=[1.6, 1.75, -np.pi / 2], tol=0.02, max_steps=200)
    print(n, 'pick', ok, 'placed', np.round(f, 3), 'step', b.nsteps, flush=True)
print('final', {n: np.round(b.obj(n), 3).tolist() for n in names})
b.wait(30)
print('no termination')
