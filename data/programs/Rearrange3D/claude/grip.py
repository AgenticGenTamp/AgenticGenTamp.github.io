import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
for val in [1.0,0.0,0.5]:
    a=np.zeros(11,dtype=np.float32); a[10]=val
    vals=[]
    for i in range(25):
        obs,r,te,tr,inf=env.step(a); vals.append(round(float(np.asarray(obs)[103]),3))
    print(val, vals[:8], vals[-3:])
env.close()
