import numpy as np
from env_client import make_env
from probe_dynamics import vals
for axis in [3,4,6,8]:
 e=make_env();s,_=e.reset(seed=0)
 for step in range(25):
  a=np.zeros(11,np.float32)
  if step<20:a[axis]=.1
  s,r,t,tr,i=e.step(a)
  if step in [9,19,24]:print(axis,step,vals(s),flush=True)
 e.close()
