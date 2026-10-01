import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env = make_env()
for val in [0.1,0.05,0.02,0.01,-0.1]:
    obs,_=env.reset(seed=0); o0=np.asarray(obs)[96:103].copy()
    a=np.zeros(11,dtype=np.float32); a[3+0]=val
    tr=[]
    for t in range(40):
        obs,r,te,trn,info=env.step(a); tr.append(np.asarray(obs)[96])
    tr=np.array(tr)
    d=np.diff(np.concatenate([[o0[0]],tr]))
    print(f"cmd={val}: total40={tr[-1]-o0[0]:.4f} mean/step={(tr[-1]-o0[0])/40:.5f} ratio={(tr[-1]-o0[0])/40/val:.3f} last10mean={d[-10:].mean():.5f}")
# hold then release: does joint drift back?
obs,_=env.reset(seed=0); o0=np.asarray(obs)[96:103].copy()
a=np.zeros(11,dtype=np.float32); a[3]=0.1
for t in range(20): obs,_,_,_,_=env.step(a)
print("after 20 push:", np.asarray(obs)[96])
z=np.zeros(11,dtype=np.float32)
for t in range(20): obs,_,_,_,_=env.step(z)
print("after 20 zero:", np.asarray(obs)[96], "(holds?)")
env.close()
