import numpy as np
from env_client import make_env
from servo import *
env=make_env(); obs,i=env.reset(seed=0)
for v in [-1.0,-0.6,-0.9,1.0,0.6]:
    a=np.zeros(11,dtype=np.float32); a[10]=v
    for k in range(3):
        obs,r,t,tr,inf=env.step(a)
    print("v",v,rinfo(obs))
# does action out of bounds error?
try:
    a=np.zeros(11,dtype=np.float32); a[10]=-2.0
    obs,r,t,tr,inf=env.step(a); print("oob ok",rinfo(obs))
except Exception as e: print("oob err",e)
