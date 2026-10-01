import numpy as np, time
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env = make_env()
obs,_=env.reset(seed=0); o=np.asarray(obs)
print("init joints", o[96:103], "grip", o[103])
print("init vel 104:115", o[104:115])

for j in range(7):
    obs,_=env.reset(seed=0); o0=np.asarray(obs)[96:103].copy()
    a=np.zeros(11,dtype=np.float32); a[3+j]=0.1
    tr=[]
    for t in range(20):
        obs,r,te,trn,info=env.step(a); tr.append(np.asarray(obs)[96:103].copy())
    tr=np.array(tr)
    d=np.diff(np.vstack([o0,tr]),axis=0)[:,j]
    cross=np.abs(tr[-1]-o0); cross[j]=0
    print(f"joint{j+1} start={o0[j]:.4f} per-step d[:5]={d[:5]} total={tr[-1][j]-o0[j]:.4f} maxcrosstalk={cross.max():.4f}")
env.close()
