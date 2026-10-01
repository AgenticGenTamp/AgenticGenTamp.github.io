import sys, math, numpy as np
from env_client import make_env
from approach import GeneratedApproach, _cheb
n=int(sys.argv[1]); off=int(sys.argv[2])
env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space, {})
diffs=[]
for seed in range(off,off+n):
    obs,info=env.reset(seed=seed); ap.reset(obs,info)
    L=sum(_cheb(*ap.path[i],*ap.path[i+1]) for i in range(len(ap.path)-1))
    lb=math.ceil(L/0.05-1e-9)
    tot=0; term=False
    for i in range(env.max_steps):
        obs,rew,term,tr,info=env.step(ap.get_action(obs)); tot+=1
        if term or tr: break
    diffs.append(tot-lb)
    if tot-lb>3: print("seed",seed,"steps",tot,"planlb",lb,"diff",tot-lb,flush=True)
print("range",off,off+n,"mean_excess",round(float(np.mean(diffs)),2),"max",max(diffs))
env.close()
