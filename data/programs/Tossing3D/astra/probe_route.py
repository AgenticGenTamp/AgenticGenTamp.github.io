from env_client import make_env
import numpy as np
for y in [3.,4.,5.]:
 e=make_env();s,i=e.reset(seed=0);rob=s.get_object_from_name('robot')
 for target in [(0,y),(3,y),(3,2)]:
  for k in range(80):
   p=np.array([s.get(rob,'pos_base_x'),s.get(rob,'pos_base_y')]);a=np.zeros(18);a[:2]=np.clip(np.array(target)-p,-.1,.1);s,r,t,tr,i=e.step(a)
  print('route',y,'target',target,'actual',p,flush=True)
 e.close()
