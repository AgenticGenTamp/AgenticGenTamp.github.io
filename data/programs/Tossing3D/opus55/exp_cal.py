import sys
import numpy as np
from env_client import make_env
import toss as T
import os
T.Q7_WIND = float(os.environ.get("Q7", 0.0))
w = float(sys.argv[1]); bx = float(sys.argv[2]); seeds = [int(s) for s in sys.argv[3:]]
env = make_env()
for seed in seeds:
    obs, _ = env.reset(seed=seed)
    n = 'cube_0'; log = []
    goal = np.array([bx, 0.0, 0.0])
    def gen(obs):
        obs = yield from T.pick(obs, n)
        obs = yield from T.throw(obs, n, w=w, base_goal=goal, log=log)
    g = gen(obs); a = next(g); t = 0
    while True:
        obs, r, te, tr, _ = env.step(a); t += 1
        try: a = g.send(obs)
        except StopIteration: break
    pre = log[0]; tr_ = np.array(log[1:])
    i = np.argmax(tr_[:, 2] < 0.06)
    print('CAL w %.3f seed %d D %.4f lat %.4f off %s steps %d relp %s relv %s' % (w, seed, tr_[i, 0] - bx, tr_[i, 1], (pre[10:13]*1000).round(2), t, tr_[2, :3].round(3), tr_[2, 3:].round(3)), flush=True)
env.close()
