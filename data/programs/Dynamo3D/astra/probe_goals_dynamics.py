import numpy as np
from env_client import make_env
E=make_env();s,info=E.reset(seed=2);t=E.observation_space.get_type('mujoco_tidybot_robot');r=next(iter(s.get_objects(t)))
step=0
for target in [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3),(4,3),(4,4),(3,4),(2,4)]:
 prev=None; stalled=0
 for i in range(80):
  p=np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y')]);d=np.array(target)-p
  if np.linalg.norm(d)<.035: break
  a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(d/.87,-.1,.1)
  s,re,te,tr,info=E.step(a);step+=1
  new=np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y')])
  if np.linalg.norm(new-p)<.01:stalled+=1
  if re!=-1:print('REWARD',step,new.tolist(),re,te,tr,info,flush=True)
  if te or tr:print('DONE',step,new.tolist(),re,te,tr,info,flush=True);E.close();raise SystemExit
 print('TARGET',target,'REACHED',new.tolist(),'TOTAL',step,'STALLS',stalled,flush=True)
E.close()
