import sys, numpy as np
from env_client import make_env
import rutil
from rutil import Bot
seed = int(sys.argv[1]); x, y = float(sys.argv[2]), float(sys.argv[3])
env = make_env(); obs, _ = env.reset(seed=seed); b = Bot(env, obs)
def step(a):
    b.obs, r, te, tr, inf = b.env.step(np.asarray(a, np.float32)); b.nsteps += 1
    c = b.obj('cube_0')
    if te or r != -1: print('SIG', b.nsteps, r, te, np.round(b.base(), 3), np.round(c, 3), flush=True)
    if te: sys.exit()
    return r
b.step = step
print('pick', rutil.pick(b, 'cube_0'), b.nsteps)
print('placed', np.round(rutil.place(b, 'cube_0', (x, y)), 3), b.nsteps, np.round(b.base(), 3))
b.wait(20)
