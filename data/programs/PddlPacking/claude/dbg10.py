import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, Robot
env=make_env(); obs,info=env.reset(seed=19)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
R=Robot(obs); print("init blocks",{k:np.round(v[:7],3) for k,v in R.blocks.items()})
for k in range(20):
    a=ap.get_action(obs)
    R=Robot(obs)
    tb=R.blocks.get(ap.target)
    print(k,ap.phase,"tool",np.round(R.tool()[0],3),"toolx",np.round(R.tool()[1][:,0],3),"blkq",np.round(tb[3:7],3) if tb is not None else "")
    obs,r,t,tr,i=env.step(a)
env.close()
