import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=3,suppress=True,linewidth=200)
seed=int(sys.argv[1]); N=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for t in range(N):
    obs,r,te,tr,info=env.step(ap.get_action(obs))
    dd=np.linalg.norm(obs[16:18]-obs[0:2]); dc=np.linalg.norm(obs[32:34]-obs[0:2])
    if dd<float(sys.argv[3]): print(t,round(dd,3),round(dc,3),obs[16:23].round(4),obs[32:39].round(3))
    if te: print('TERM',t); break
