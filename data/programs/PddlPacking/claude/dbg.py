import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, Robot
s=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
R=Robot(obs); print("blocks",{k:np.round(v[:3],3) for k,v in R.blocks.items()})
last=None; n=0
while n<400:
    a=ap.get_action(obs)
    ph=ap.phase
    if ph!=last:
        R=Robot(obs)
        print(n,"->",ph,"tgt",ap.target,"cell",getattr(ap,'cell',None),"base",np.round(R.base,2),"tool",np.round(R.tool()[0],3),"hold",R.holding)
        last=ph
    obs,r,term,trunc,info=env.step(a); n+=1
    if term: print("TERM at",n); break
else: print("no term")
R=Robot(obs); print("final blocks",{k:np.round(v[:3],3) for k,v in R.blocks.items()})
env.close()
