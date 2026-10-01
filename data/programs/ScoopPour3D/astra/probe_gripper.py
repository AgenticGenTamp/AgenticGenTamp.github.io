from env_client import make_env
import numpy as np
for g in [0.,1.]:
 e=make_env();s,info=e.reset(seed=0);scoop=s.get_object_from_name('scoop_0');rob=s.get_object_from_name('robot')
 p0=np.array([s.get(scoop,f) for f in ['x','y','z']])
 for i in range(20):
  a=np.zeros(11,dtype=np.float32);a[10]=g
  if i>=3: a[4]=-.1
  s,r,t,tr,info=e.step(a)
  if i in [0,2,8,19]:
   print('g',g,'i',i,'scoop',np.round([s.get(scoop,f) for f in ['x','y','z']],4),'q2',round(s.get(rob,'pos_arm_joint2'),4),'r',r,flush=True)
 e.close()
