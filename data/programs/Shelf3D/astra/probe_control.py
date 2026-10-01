import numpy as np
from env_client import make_env
from probe_dynamics import vals
for axis in [0,3,4,10]:
 e=make_env();s,_=e.reset(seed=0)
 for step in range(8):
  a=np.zeros(11,np.float32)
  if step<3:a[axis]=1 if axis==10 else .1
  s,r,t,tr,i=e.step(a)
  print(axis,step,vals(s)['robot'],flush=True)
 e.close()
