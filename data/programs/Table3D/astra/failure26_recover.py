from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();P=GeneratedApproach(E.action_space,E.observation_space,{})
s,info=E.reset(seed=26,options={'object_count':10});P.reset(s,info)
for i in range(12):s,*_=E.step(P.get_action(s))
for i in range(5):
 a=np.zeros(11);a[-1]=-1
 if s.get(P.r,'grasp_active'):a[4]=-.4
 else:
  q2=.434994191;q4=-2.3
  target=np.array([s.get(P.c,'pose_x')-.44153585,s.get(P.c,'pose_y')-.00135+.02,0,0,q2,-np.pi,q4,0,q2-q4-np.pi,np.pi/2]);cur=np.array([s.get(P.r,f) for f in P.fields]);a[:10]=np.clip(target-cur,-.4,.4)
 s,re,te,tr,info=E.step(a)
 print('RECOVER',i+1,'grasp',s.get(P.r,'grasp_active'),'term',te,flush=True)
 if te:break
E.close()
