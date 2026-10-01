import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
for j in [0,3]:
    obs,_=env.reset(seed=0); q0=R(obs)[3:10].copy()
    arm=[0]*7; arm[j]=0.05
    s=[]
    for k in range(10):
        obs,*_=env.step(A(arm=arm)); s.append(R(obs)[3+j]-q0[j])
    print('j%d repeated 0.05:'%(j+1),np.round(s,4),'vel',R(obs)[14+j])
    s=[]
    for k in range(10):
        obs,*_=env.step(A()); s.append(R(obs)[3+j]-q0[j])
    print('  then zeros:',np.round(s,4))
# single pulse then long zeros
obs,_=env.reset(seed=0); q0=R(obs)[3:10].copy()
obs,*_=env.step(A(arm=[0.1,0,0,0,0,0,0])); s=[R(obs)[3]-q0[0]]
for k in range(10): obs,*_=env.step(A()); s.append(R(obs)[3]-q0[0])
print('j1 pulse 0.1:',np.round(s,4))
