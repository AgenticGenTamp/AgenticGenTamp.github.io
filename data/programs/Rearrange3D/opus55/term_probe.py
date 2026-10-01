import numpy as np, sys
import approach
from env_client import make_env
from approach import GeneratedApproach
approach.PLACE_DZ=float(sys.argv[2])
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for t in range(500):
    a=ap.get_action(obs)
    obs,r,te,tr,info=env.step(a)
    if te or t%5==0:
        dd=np.linalg.norm(obs[16:18]-obs[0:2]); dc=np.linalg.norm(obs[32:34]-obs[0:2])
        if min(dd,dc)<0.14: print(t,te,'grip',a[10],'dd %.3f dc %.3f zd %.3f zc %.3f'%(dd,dc,obs[18],obs[34]), 'gr', obs[103].round(3))
    if te: break
