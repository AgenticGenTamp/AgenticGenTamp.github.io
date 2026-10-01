import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
obs,_=env.reset(seed=0); print('furn',obs[48:93])
# one-step tracking per joint, +0.05 then zeros
for j in range(7):
    for c in [0.05,-0.05,0.1]:
        obs,_=env.reset(seed=0); q0=R(obs)[3:10].copy()
        arm=[0]*7; arm[j]=c
        obs,*_=env.step(A(arm=arm)); d1=R(obs)[3:10]-q0
        obs,*_=env.step(A()); obs,*_=env.step(A()); d3=R(obs)[3:10]-q0
        print('j%d cmd %.2f d1 %.4f d3 %.4f other_max %.4f'%(j+1,c,d1[j],d3[j],np.abs(np.delete(d3,j)).max()))
