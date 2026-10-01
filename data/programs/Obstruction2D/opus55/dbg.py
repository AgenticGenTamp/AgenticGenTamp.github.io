import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed, count = int(sys.argv[1]), int(sys.argv[2])
env = make_env()
obs, info = env.reset(seed=seed, options=({'object_count': count} if count>0 else None))
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
ap._parse(obs)
print('surface', ap.surface); print('block', ap.block)
for n,b in ap.obst.items(): print(n, b)
lastkind=None
for t in range(int(sys.argv[3]) if len(sys.argv)>3 else 1000):
    a = ap.get_action(obs)
    k = (ap.target_name, ap.plan[0]['kind'] if ap.plan else None)
    if k!=lastkind: print(t, k, {kk:round(v,3) for kk,v in ap.robot.items() if kk in 'x y arm vac'.split()}, ap.plan[0] if ap.plan else None); lastkind=k
    obs, r, term, trunc, info = env.step(a)
    if term: print('done', t+1); break
