import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
for c in [0.01,0.02,0.05,0.08,0.1]:
    obs,_=env.reset(seed=0); p0=R(obs)[:3].copy()
    obs,*_=env.step(A(b=(-c,0,0))); d1=R(obs)[0]-p0[0]
    for k in range(3): obs,*_=env.step(A(b=(-c,0,0)))
    print('cmd -x',c,'d1',round(d1,4),'avg over 4',round((R(obs)[0]-p0[0])/4,4))
for c in [0.02,0.05,0.1]:
    obs,_=env.reset(seed=0); p0=R(obs)[:3].copy()
    obs,*_=env.step(A(b=(0,0,c))); print('yaw',c,R(obs)[2]-p0[2])
# stuck then retreat
obs,_=env.reset(seed=0)
for k in range(10): obs,*_=env.step(A(b=(0.05,0,0)))
p=R(obs)[:3].copy(); print('stuck',p)
for k in range(3):
    obs,*_=env.step(A(b=(-0.05,0,0))); print('retreat',k,R(obs)[:3])
# seeds: initial base pose variance
for s in range(5):
    obs,_=env.reset(seed=s); print('seed',s,R(obs)[:3], 'arm', R(obs)[3:11])
