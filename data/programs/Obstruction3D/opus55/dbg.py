import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from kin import fk
seed=int(sys.argv[1]); N=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(N):
    a = ap.get_action(obs)
    obs, r, te, tr, info = env.step(a)
    n = ap.task['name'] if ap.task else None
    tool = fk(ap.base, ap.q)[:3,3]
    top = ap.objs[n][0][2]+ap.objs[n][1][2] if n else 0
    print(t, ap.phase, n, 'tool', tool.round(3), 'obj', ap.objs[n][0].round(3) if n else '', 'top', round(top,3), 'grip', a[10], 'g', ap.grasped, 'dq', np.abs(a[3:10]).max().round(3))
    if te: print('DONE'); break
