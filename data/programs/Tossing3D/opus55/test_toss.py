import sys, os
import numpy as np
from env_client import make_env
import toss as T
if os.environ.get("AIM"): T.AIM_BEYOND = float(os.environ["AIM"])

seeds = [int(s) for s in sys.argv[1:]] or list(range(5))
env = make_env()
res = []
for seed in seeds:
    nobj = os.environ.get('NOBJ')
    obs, _ = env.reset(seed=seed, options={'object_count': int(nobj)} if nobj else None)
    cubes = T.cube_names(obs)
    binp = T.objpos(obs, 'bin_0')
    g = T.toss_all(obs)
    a = next(g)
    R = 0.0
    for t in range(1000):
        obs, r, te, tr, _ = env.step(a)
        R += r
        if te or tr:
            break
        a = g.send(obs)
    fin = {n: T.objpos(obs, n).round(3).tolist() for n in cubes}
    print('seed', seed, 'bin', binp.round(3).tolist(), 'steps', t + 1, 'term', te, 'trunc', tr,
          'last_r', round(r, 3), 'ret', round(R, 2), 'cubes', fin, 'bin_end', T.objpos(obs, 'bin_0').round(3).tolist(), 'base', T.rb(obs).round(3).tolist(), flush=True)
    res.append(te)
print('SUCCESS', sum(res), '/', len(res))
env.close()
