import sys
import numpy as np
from env_client import make_env

seed=int(sys.argv[1]); seq=[]
for spec in sys.argv[2:]:
    axis,val,n=spec.split(','); seq.append((int(axis),float(val),int(n)))
env=make_env(); s,info=env.reset(seed=seed); rob=s.get_object_from_name('robot')
print('start',info,[float(s.get(rob,f)) for f in ('pos_base_x','pos_base_y','pos_base_rot')],flush=True)
k=0
for axis,val,n in seq:
  a=np.zeros(11,np.float32); a[axis]=val
  for _ in range(n):
    s,r,t,tr,inf=env.step(a); k+=1
    if k%10==0 or t or tr:
      print(k,r,t,tr,[round(float(s.get(rob,f)),3) for f in ('pos_base_x','pos_base_y','pos_base_rot')],flush=True)
    if t or tr: env.close(); raise SystemExit
env.close()
