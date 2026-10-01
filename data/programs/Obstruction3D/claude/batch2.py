import sys, numpy as np, time
from env_client import make_env
from approach import GeneratedApproach
n=int(sys.argv[1]); lo,hi=int(sys.argv[2]),int(sys.argv[3])
env=make_env()
fails=[]; steps=[]
for seed in range(lo,hi):
    obs,info=env.reset(seed=seed, options={"object_count":n})
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    t0=time.time(); ap.reset(obs,info)
    term=False
    for i in range(env.max_steps):
        a=ap.get_action(obs)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    if not term: fails.append(seed)
    else: steps.append(i+1)
print("n",n,"range",lo,hi,"fails",fails,"mean",round(np.mean(steps),1) if steps else None,"max",max(steps) if steps else None,flush=True)
