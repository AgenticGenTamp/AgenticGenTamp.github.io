import numpy as np
from env_client import make_env
np.set_printoptions(precision=3,suppress=True)
env=make_env()
for cmd in [(0.1,0,0),(0,0.1,0),(0,0,0.1),(0.05,0,0)]:
    obs,info=env.reset(seed=0)
    a=np.zeros(11,dtype=np.float32); a[0:3]=cmd
    tr=[np.asarray(obs)[93:96].copy()]
    for i in range(40):
        obs,r,te,trc,inf=env.step(a); tr.append(np.asarray(obs)[93:96].copy())
    tr=np.array(tr)
    print("cmd",cmd)
    for k in [1,2,5,10,20,40]: print("   ",k,np.round(tr[k],4), "d",np.round(tr[k]-tr[k-1],4))
env.close()
