from env_client import make_env
import numpy as np
for seed in range(3):
 e=make_env();s,i=e.reset(seed=seed)
 print('seed',seed,'info',i)
 for n in sorted(s.get_object_names()):
  o=s.get_object_from_name(n)
  if n!='robot':print(n,[round(s.get(o,f),3) for f in ['x','y','z']])
 for k in range(40):
  a=np.zeros(18);a[0]=.1;s,r,t,tr,i=e.step(a)
  if k%10==9:
   o=s.get_object_from_name('robot');print(k,[round(s.get(o,f),3) for f in ['pos_base_x','pos_base_y']],r)
 e.close()
