import numpy as np, sys, time
from env_client import make_env
from approach import GeneratedApproach
n=int(sys.argv[1]); seeds=[int(x) for x in sys.argv[2:]] or list(range(5))
env=make_env(); tot=[]
for s in seeds:
    obs,info=env.reset(seed=s, options={"object_count":n})
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    t0=time.time(); k=0; term=False
    while k<env.max_steps:
        a=ap.get_action(obs); obs,r,term,trunc,info=env.step(a); k+=1
        if term or trunc: break
    tot.append((s,n,k,term,round(time.time()-t0,1),ap.phase)); print(tot[-1])
print("solved",sum(1 for t in tot if t[3]),"/",len(tot),"mean",np.mean([t[2] for t in tot]))
env.close()
