import numpy as np
from env_client import make_env
for seed,route in [(1,[(0,-1),(1,-1),(1,0)]),(1,[(2,-1),(2,0),(1,0)]),(1,[(1,1),(1,0)]),(2,[(0,-1),(1,-1),(1,0)]),(3,[(2,-1),(2,0),(1,0)]),(4,[(1,1),(1,0)])]:
 E=make_env();s,info=E.reset(seed=seed,options={'object_count':1});r=next(iter(s.get_objects(E.observation_space.get_type('mujoco_tidybot_robot'))));cs=list(s.get_objects(E.observation_space.get_type('mujoco_movable_object')))
 start=[s.get(r,'pos_base_x'),s.get(r,'pos_base_y')];chairs=[[round(s.get(c,f),3) for f in ['x','y']]for c in cs]
 step=0;done=False
 for target in route:
  for i in range(100):
   p=np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y')]);d=np.array(target)-p
   if np.linalg.norm(d)<.025:break
   a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(d/.87,-.1,.1)
   s,re,te,tr,info=E.step(a);step+=1
   if te or tr:
    end=np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y')]);near=min(np.linalg.norm(end-[s.get(c,'x'),s.get(c,'y')])for c in cs)
    print('DONE',seed,route,'start',start,'chairs',chairs,'n',step,'p',end.tolist(),'dgoal',np.linalg.norm(end-[1,0]),'dchair',near,'reward',re,te,tr,flush=True);done=True;break
  if done:break
 if not done:print('NOTDONE',seed,route,'chairs',chairs,'p',[s.get(r,'pos_base_x'),s.get(r,'pos_base_y')],flush=True)
 E.close()
