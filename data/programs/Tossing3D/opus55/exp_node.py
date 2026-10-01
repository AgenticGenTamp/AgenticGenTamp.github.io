import sys, json
import numpy as np
from env_client import make_env
import toss as T

w = float(sys.argv[1]); seeds = [int(s) for s in sys.argv[2:]]
env = make_env()
for seed in seeds:
    obs, _ = env.reset(seed=seed)
    n = 'cube_0'
    bp = T.objpos(obs, 'bin_0')
    D = np.interp(w, T.CAL_W, T.CAL_D)
    goal = np.array([bp[0] + T.AIM_BEYOND - D, bp[1], 0.0])
    if not (0.1 < goal[0] < 0.9):
        print('skip', seed, goal); continue
    def gen(obs):
        obs = yield from T.pick(obs, n)
        obs = yield from T.throw(obs, n, w=w, base_goal=goal)
        for _ in range(15):
            obs = yield T.arm_act(obs, T.rq(obs), T.OPEN)
    g = gen(obs); a = next(g)
    first = None; t = 0; rec = []
    while True:
        obs, r, te, tr, _ = env.step(a); t += 1
        c = T.objpos(obs, n); b = T.objpos(obs, 'bin_0')
        rec.append((t, te, c.round(3).tolist(), b.round(3).tolist()))
        if te and first is None:
            first = rec[-1]
        try: a = g.send(obs)
        except StopIteration: break
    # landing: first rec with cube z<0.25 and x near bin
    land = [x for x in rec if x[2][2] < 0.22 and abs(x[2][0]-bp[0]) < 0.4][:2]
    print('w %.3f seed %d bin %s term %s first %s land %s final %s' % (w, seed, bp[:2].round(3).tolist(), first is not None, first, land, rec[-1][2:]), flush=True)
env.close()
