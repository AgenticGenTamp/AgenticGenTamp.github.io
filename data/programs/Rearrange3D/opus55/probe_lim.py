import numpy as np, time
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
t=time.time()
for j in range(7):
  for sg in [1,-1]:
    obs,_=env.reset(seed=0); q0=R(obs)[3:10].copy()
    arm=[0]*7; arm[j]=0.1*sg
    tr=[]
    for k in range(160):
        obs,r,te,trn,info=env.step(A(arm=arm)); tr.append(R(obs)[3+j])
        if te or trn: print('TERM',k,te,trn); break
    tr=np.array(tr)
    print('j%d dir %+d start %.3f min %.3f max %.3f final %.3f  at step40 %.3f 80 %.3f 120 %.3f'%(j+1,sg,q0[j],tr.min(),tr.max(),tr[-1],tr[39],tr[79],tr[119]))
print('time',time.time()-t)
