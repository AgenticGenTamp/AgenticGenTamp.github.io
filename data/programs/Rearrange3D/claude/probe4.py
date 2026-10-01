import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True)
env=make_env()
for dim in [3,4,5,6,7,8,9]:
    obs,info=env.reset(seed=0)
    a=np.zeros(11,dtype=np.float32); a[dim]=0.1
    qs=[np.asarray(obs)[96:103].copy()]
    for k in range(30):
        obs,r,te,tr,inf=env.step(a)
        qs.append(np.asarray(obs)[96:103].copy())
    qs=np.array(qs)
    print("dim",dim)
    for k in [1,5,10,20,30]:
        print("  ",k, qs[k])
env.close()
