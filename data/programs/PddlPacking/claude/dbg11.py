import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, Robot
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
R=Robot(obs)
for n,b in R.blocks.items():
    print(n,np.round(b[:3],3),"onplate",ap.on_plate(R,b))
    pl=ap.plan_pick(R,n)
    print("   plan:", "NONE" if pl is None else (np.round(pl[0],2), np.round(pl[1],3), np.round(pl[2][:,0],2)))
env.close()
