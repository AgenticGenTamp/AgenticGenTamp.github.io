import sys, time, numpy as np
from env_client import make_env
import approach
K=float(sys.argv[1]); approach.K_ANG=K; approach.LOOKAHEAD=int(sys.argv[3]) if len(sys.argv)>3 else 8; approach.LOCAL_FAMILY=(sys.argv[4]=="1") if len(sys.argv)>4 else True
seeds=[int(x) for x in sys.argv[2].split(',')] if ',' in sys.argv[2] else range(*map(int,sys.argv[2].split(':')))
import time
env=make_env(); res=[]; steps=[]; tmax=0
for sd in seeds:
    obs,info=env.reset(seed=sd); ap=approach.GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    ok=False; t0=time.time()
    for t in range(1000):
        obs,r,term,trunc,info=env.step(ap.get_action(obs))
        if term: ok=True; break
    res.append(ok); steps.append(t+1); tmax=max(tmax,time.time()-t0)
    if not ok: print('fail',sd)
print('K',K,'succ',np.mean(res),'meansteps',np.mean(steps),'tmax',round(tmax,1))
