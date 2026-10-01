from env_client import make_env
import numpy as np
E=make_env()
s,_=E.reset(seed=0)
r=s.get_object_from_name('robot')
for dim,sign in [(0,-1),(1,-1),(1,1),(0,1)]:
 for _ in range(100):
  a=np.zeros(5,dtype=np.float32); a[dim]=sign*.05
  s,*_=E.step(a)
 print(dim,sign,{f:s.get(r,f) for f in ('x','y','theta','arm_joint')})
E.close()
