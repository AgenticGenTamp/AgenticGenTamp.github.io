import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env=make_env()
obs,info=env.reset(seed=0)
o0=np.asarray(obs).copy()
# sweep joint2 (idx 97) to bring arm forward/down, watch objects
a=np.zeros(11,dtype=np.float32)
rews=set()
for k in range(60):
    a[:]=0
    a[4]=0.1   # joint2
    obs,r,te,tr,inf=env.step(a); rews.add(round(float(r),4))
o=np.asarray(obs)
print("q",o[96:103])
print("obj deltas", np.abs(o[[0,1,2,16,17,18,32,33,34]]-o0[[0,1,2,16,17,18,32,33,34]]))
print("rews",rews)
for k in range(60):
    a[:]=0; a[4]=-0.1
    obs,r,te,tr,inf=env.step(a); rews.add(round(float(r),4))
o=np.asarray(obs)
print("q back",o[96:103], "rews",rews)
print("obj deltas", np.abs(o[[0,1,2,16,17,18,32,33,34]]-o0[[0,1,2,16,17,18,32,33,34]]))
env.close()
