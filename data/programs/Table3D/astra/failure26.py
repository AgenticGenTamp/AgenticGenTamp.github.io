from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();P=GeneratedApproach(E.action_space,E.observation_space,{})
for idx in range(10):
 s,inf=E.reset(seed=26,options={'object_count':10});P.reset(s,inf)
 cs=sorted([c for c in s.get_objects(P.ct) if c.name!='table'],key=lambda c:s.get(c,'pose_x'))
 if idx==0:print('CUBES',[(c.name,[round(s.get(c,'pose_'+f),4) for f in 'xyz']) for c in cs],flush=True)
 P.c=cs[idx];init=[s.get(P.c,'pose_'+f) for f in 'xyz']
 for i in range(40):
  old=np.array([s.get(P.r,f) for f in P.fields]);a=P.get_action(s);s,re,te,tr,info=E.step(a)
  now=np.array([s.get(P.r,f) for f in P.fields]);ga=s.get(P.r,'grasp_active')
  if idx==0:print('STEP',i+1,'old',old.round(3).tolist(),'action',a.round(3).tolist(),'new',now.round(3).tolist(),'grasp',ga,'term',te,flush=True)
  if te:break
 print('RESULT',idx,P.c.name,init,i+1,te,'grasp',ga,flush=True)
E.close()
