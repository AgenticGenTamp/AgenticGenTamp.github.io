import sys, numpy as np
from env_client import make_env
from rutil import Bot
env = make_env(); obs, _ = env.reset(seed=1); b = Bot(env, obs)
orig = b.step
def step(a):
    b.obs, r, te, tr, inf = b.env.step(np.asarray(a, np.float32)); b.nsteps += 1
    if te or r != -1: print('TERM', b.nsteps, r, np.round(b.base(), 3), np.round(b.obj('cube_0'), 3), flush=True); sys.exit()
    return r
b.step = step
for p in [(1.2, 1.3), (0.8, 1.3), (0.5, 1.3), (0.3, 1.3), (0.2, 1.3), (0.1, 1.3), (0.0, 1.3), (-0.5, 1.3)]:
    b.goto(base_t=[p[0], p[1], 0.0], tol=0.01, max_steps=100)
    print('at', np.round(b.base(), 3), b.nsteps, flush=True)
