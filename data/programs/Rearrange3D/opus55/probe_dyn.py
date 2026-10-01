import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
obs,info=env.reset(seed=0)
print('init base',R(obs)[:3],'arm',R(obs)[3:10],'grip',R(obs)[10])
# 1. base +x 0.05 one step then zeros
for k in range(4):
    a=A(b=(0.05,0,0)) if k==0 else A()
    obs,r,te,tr,info=env.step(a); print('bx step',k,R(obs)[:3],r,te,tr)
print('info',info)
# rotate yaw
for k in range(8):
    obs,*_=env.step(A(b=(0,0,0.1)))
for k in range(3): obs,*_=env.step(A())
p0=R(obs)[:3].copy(); print('after yaw',p0)
obs,*_=env.step(A(b=(0.05,0,0))); print('cmd +x after yaw ->',R(obs)[:3]-p0)
for k in range(3): obs,*_=env.step(A())
print('settled delta',R(obs)[:3]-p0)
p0=R(obs)[:3].copy()
obs,*_=env.step(A(b=(0,0.05,0))); 
for k in range(3): obs,*_=env.step(A())
print('cmd +y after yaw settled delta',R(obs)[:3]-p0)
# continuous base motion tracking
p0=R(obs)[:3].copy()
for k in range(5):
    obs,*_=env.step(A(b=(0.1,0,0))); print(' cont x',k,R(obs)[:3]-p0)
