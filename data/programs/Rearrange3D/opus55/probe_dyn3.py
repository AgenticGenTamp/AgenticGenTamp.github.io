import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
# frame test: move back first to get clearance, then yaw 0.8
obs,_=env.reset(seed=0)
for k in range(5): obs,*_=env.step(A(b=(-0.05,0,0)))
for k in range(8): obs,*_=env.step(A(b=(0,0,0.1)))
for k in range(2): obs,*_=env.step(A())
p0=R(obs)[:3].copy(); print('pose',p0)
obs,*_=env.step(A(b=(0.05,0,0))); print('+x after yaw d1',R(obs)[:3]-p0)
p0=R(obs)[:3].copy()
obs,*_=env.step(A(b=(0,0.05,0))); print('+y after yaw d1',R(obs)[:3]-p0)
# drive toward table
obs,_=env.reset(seed=0)
for k in range(25):
    obs,r,te,tr,info=env.step(A(b=(0.05,0,0)))
    print(k,R(obs)[:3],R(obs)[11:14],r) if k%2==0 or k>10 else None
