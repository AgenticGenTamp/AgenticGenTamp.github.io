import sys
import numpy as np
from env_client import make_env
import toss as T
aim = float(sys.argv[1]); seeds = [int(s) for s in sys.argv[2:]]
env = make_env()
for seed in seeds:
    obs, _ = env.reset(seed=seed)
    n = 'cube_0'; bp = T.objpos(obs, 'bin_0'); log = []
    def gen(obs):
        obs = yield from T.pick(obs, n)
        obs = yield from T.throw(obs, n, aim=aim, log=log)
        for _ in range(10):
            obs = yield T.arm_act(obs, T.rq(obs), T.OPEN)
    g = gen(obs); a = next(g); t = 0; first = None
    while True:
        obs, r, te, tr, _ = env.step(a); t += 1
        if te and first is None: first = t
        try: a = g.send(obs)
        except StopIteration: break
    tr_ = np.array(log[1:]); i = np.argmax(tr_[:, 2] < 0.22)
    print('AIM %.3f seed %d term %s step %s bin %s landrel %s base %s' % (aim, seed, first is not None, first, bp[:2].round(3), (tr_[i, :3] - bp).round(3), T.rb(obs).round(3)), flush=True)
env.close()
