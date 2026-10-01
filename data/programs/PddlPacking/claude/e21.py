import numpy as np
from env_client import make_env
from approach import GeneratedApproach, Robot
env=make_env(); obs,info=env.reset(seed=0)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
R=Robot(obs)
for n,b in R.blocks.items():
    print(n, np.round(b[:3],3))
    pl=ap.plan_pick(R,n)
    print("  plan",None if pl is None else (np.round(pl[0],2),np.round(pl[1],3)))
    for base in [(-0.43,0.0,0.0),(-0.66,0.0,0.0),(-0.66,-0.2,0.0)]:
        gp=np.array([b[0],b[1],b[2]-0.05*-1])
        print("   base",base,"pre",ap.prefilter(base,gp),"ik",ap.ik_at(base,gp,np.eye(3)*0+np.column_stack([[0,0,-1],[0,1,0],[1,0,0]])) is not None)
env.close()
