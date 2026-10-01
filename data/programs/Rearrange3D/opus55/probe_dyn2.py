import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
for cmd in [(0.05,0,0),(-0.05,0,0),(0,0.05,0),(0,-0.05,0),(0,0,0.1),(0,0,-0.1),(0.1,0,0),(0.1,0.1,0.1)]:
    obs,_=env.reset(seed=0); p0=R(obs)[:3].copy()
    out=[]
    for k in range(3):
        obs,*_=env.step(A(b=cmd) if k==0 else A()); out.append(R(obs)[:3]-p0)
    print(cmd,'d1',out[0],'d3',out[2],'vel',R(obs)[11:14])
# yaw step-by-step
obs,_=env.reset(seed=0); p0=R(obs)[:3].copy()
for k in range(8):
    obs,*_=env.step(A(b=(0,0,0.1))); print('yaw',k,R(obs)[:3]-p0)
