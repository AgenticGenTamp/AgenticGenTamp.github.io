import numpy as np, time
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env=make_env()
obs,info=env.reset(seed=0)
o=np.asarray(obs)
print("q0",o[96:103])
a=np.zeros(11,dtype=np.float32); a[3]=0.1
t0=time.time()
for i in range(10):
    obs,r,te,tr,inf=env.step(a)
print("after10 j1",np.asarray(obs)[96:103], "t",time.time()-t0)
a=np.zeros(11,dtype=np.float32); a[0]=0.1
for i in range(10):
    obs,r,te,tr,inf=env.step(a)
print("base",np.asarray(obs)[93:96])
env.close()
