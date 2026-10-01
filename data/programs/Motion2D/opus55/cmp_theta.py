import sys, math
import numpy as np
from env_client import make_env
from approach import GeneratedApproach, AXES
env = make_env()
gains=[]
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap._parse(obs)
    r=ap.robot; st=(float(obs.get(r,'x')),float(obs.get(r,'y'))); th=float(obs.get(r,'theta'))
    ap.arm=0.1; ap.margin=0.004
    costs=[]
    for c in [th]+list(AXES):
        p=ap._plan_single_m(st,c)
        if p is None: costs.append(None); continue
        pts=p[0][1]; costs.append(sum(max(abs(a[0]-b[0]),abs(a[1]-b[1])) for a,b in zip(pts,pts[1:]))/0.05)
    valid=[c for c in costs[1:] if c is not None]
    if costs[0] is not None and valid:
        gains.append(costs[0]-min(valid))
        if costs[0]-min(valid)>1: print(seed, [None if c is None else round(c,1) for c in costs])
print('mean gain', np.mean(gains), 'n>1', sum(g>1 for g in gains), len(gains))
