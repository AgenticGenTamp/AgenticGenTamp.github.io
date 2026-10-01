import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
obs,_=env.reset(seed=0)
s=[]
for k in range(12): obs,r,*_=env.step(A(g=1.0)); s.append(R(obs)[10])
print('close g=1:',np.round(s,4))
s=[]
for k in range(12): obs,r,*_=env.step(A(g=0.0)); s.append(R(obs)[10])
print('open g=0:',np.round(s,4))
for gv in [0.25,0.5,0.75]:
    obs,_=env.reset(seed=0); s=[]
    for k in range(10): obs,*_=env.step(A(g=gv)); s.append(R(obs)[10])
    print('g=%.2f:'%gv,np.round(s,4))
    s=[]
