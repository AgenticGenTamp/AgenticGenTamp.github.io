import sys; sys.path.insert(0,'.')
from env_client import make_env
import numpy as np
env=make_env(); obs,info=env.reset(seed=0)
for t in range(1,1300):
    obs,r,te,tr,info=env.step(np.zeros(11))
    if r!=-1.0 or te or tr: print(t,r,te,tr,info); 
    if te or tr: break
print('end',t)
