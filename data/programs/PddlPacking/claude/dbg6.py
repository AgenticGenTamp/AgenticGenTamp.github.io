import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, Robot
s=int(sys.argv[1]); n=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=s, options={"object_count":n})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
R=Robot(obs); print("blocks",{k:np.round(v[:3],3) for k,v in R.blocks.items()}); print("cells",np.round(ap.cells,3))
last=None; k=0
while k<300:
    a=ap.get_action(obs)
    if ap.phase!=last:
        R=Robot(obs)
        print(k,"->",ap.phase,"tgt",ap.target,"cell",np.round(ap.cell,3) if ap.cell else None,"base",np.round(R.base,2),"tool",np.round(R.tool()[0],3),"hold",R.holding)
        last=ap.phase
    obs,r,term,trunc,info=env.step(a); k+=1
    if term: print("TERM",k);break
env.close()
