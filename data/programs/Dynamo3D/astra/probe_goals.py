from env_client import make_env
import numpy as np
import time
with make_env() as e:
 s,info=e.reset(seed=0)
 rbt=s.get_object_from_name('robot')
 for target in [(3,0),(3,2),(3,3),(3,4),(4,4),(4,3),(4,2),(2,4),(2,3),(2,2)]:
  for k in range(100):
   p=np.array([s.get(rbt,'pos_base_x'),s.get(rbt,'pos_base_y')]); d=np.array(target)-p
   a=np.zeros(11); a[:2]=np.clip(d,-.1,.1)
   s,r,t,tr,info=e.step(a)
   if r!=-1 or t or tr: print('SPECIAL',target,k,p,r,t,tr,info,flush=True)
   if t or tr: exit()
   if np.linalg.norm(d)<.03: break
  print('TARGET',target,'pos',p,'k',k,flush=True)
