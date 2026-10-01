import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True)
env=make_env()
obs,_=env.reset(seed=0); o=np.asarray(obs).copy()
z=np.zeros(11,dtype=np.float32)
a=np.zeros(11,dtype=np.float32); a[3:10]=0.05
for t in range(40): obs,_,_,_,_=env.step(a)
o2=np.asarray(obs)
ch=np.where(np.abs(o2-o)>1e-4)[0]
print("changed idx:", ch.tolist())
env.close()
