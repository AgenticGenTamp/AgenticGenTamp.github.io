import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=250)
for idx,val in [(0,0.1),(0,-0.1),(1,0.1),(1,-0.1),(2,0.1)]:
    env=make_env(); obs,_=env.reset(seed=0); b0=obs[125:128].copy()
    for k in range(20):
        a=np.zeros(11); a[idx]=val
        obs,r,t,tr,_=env.step(a)
    print("act",idx,val,"base",np.round(obs[125:128],3),"d",np.round(obs[125:128]-b0,3),"r",r)
    env.close()
