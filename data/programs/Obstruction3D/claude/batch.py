import sys, numpy as np, time
from env_client import make_env
from approach import GeneratedApproach
lo,hi=int(sys.argv[1]),int(sys.argv[2])
env=make_env()
fails=[]; steps=[]
for seed in range(lo,hi):
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    t0=time.time(); ap.reset(obs,info)
    term=False
    for i in range(env.max_steps):
        a=ap.get_action(obs)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    dt=time.time()-t0
    if not term: fails.append((seed,info.get('object_count'),round(dt,1)))
    else: steps.append(i+1)
    if dt>25: print("SLOW",seed,round(dt,1),flush=True)
print("range",lo,hi,"fails",fails,"mean steps",round(np.mean(steps),1) if steps else None,"n",len(steps),flush=True)
