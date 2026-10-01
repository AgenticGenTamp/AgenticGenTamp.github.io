import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=1,options={'object_count':10})
a=np.zeros(11)
for t in range(1,8001):
    obs,r,te,tr,info=env.step(a)
    if r!=-1.0 or te or tr or t%1000==0: print(t,r,te,tr,info,flush=True)
    if te or tr: break
