import sys
import numpy as np
from env_client import make_env
import toss as T
w = float(sys.argv[1]); seed = int(sys.argv[2])
env = make_env()
obs, _ = env.reset(seed=seed)
n = 'cube_0'; bp = T.objpos(obs, 'bin_0')
goal = np.array([bp[0] + T.AIM_BEYOND - np.interp(w, T.CAL_W, T.CAL_D), bp[1], 0.0])
def q(o, nm):
    C = o.get_object_from_name(nm); return np.array([o.get(C, f) for f in ['qw','qx','qy','qz']])
def gen(obs):
    obs = yield from T.pick(obs, n)
    obs = yield from T.throw(obs, n, w=w, base_goal=goal)
    for _ in range(12):
        obs = yield T.arm_act(obs, T.rq(obs), T.OPEN)
g = gen(obs); a = next(g); t = 0
while True:
    obs, r, te, tr, _ = env.step(a); t += 1
    c = T.objpos(obs, n)
    if c[2] < 0.3 and c[0] > 2:
        b = T.objpos(obs, 'bin_0')
        print(t, te, round(r, 3), 'rel', (c - b).round(3), 'cq', q(obs, n).round(3), 'bin', b.round(3), 'bq', q(obs, 'bin_0').round(3), 'cv', T.objvel(obs, n).round(2))
    try: a = g.send(obs)
    except StopIteration: break
