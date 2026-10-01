import sys, numpy as np
from env_client import make_env
import approach
from approach import GeneratedApproach
dz=float(sys.argv[1]); go=float(sys.argv[2]); seeds=range(int(sys.argv[3]), int(sys.argv[4]))
approach.GRASP_DZ=dz; approach.G_OPEN=go; approach.OPEN_GAP=0.085*(1-go)
succ=0; tot=0
env = make_env()
for seed in seeds:
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    prev=None; att=0
    for t in range(400):
        a = ap.get_action(obs)
        obs, r, term, trunc, info = env.step(a)
        if prev=='lift' and ap.phase!='lift':
            tot+=1; succ+= ap.phase=='transport'; att+=1
            if att>=3: break
        prev=ap.phase
        if term: break
print(dz, go, succ, tot, flush=True)
