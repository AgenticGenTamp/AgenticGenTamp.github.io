from env_client import make_env
from approach import GeneratedApproach
import numpy as np
for x,y,z in [(.2,0,.123),(.205,0,.125),(.4,0,.123),(.3,.15,.123),(.3,-.15,.123),(.2,-.15,.123)]:
 e=make_env();s,info=e.reset(seed=1,options={'object_count':1});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);p.vertical=True;p.grasp_q=np.array([0.,.61910148195,-np.pi,-1.6,0.,-.92249117164,np.pi/2]);same=0
 for i in range(55):
  a=p.get_action(s)
  if hasattr(p,'drop'):p.drop=np.array([x,y,z])
  ns,r,te,tr,inf=e.step(a);same=same+1 if np.max(np.abs(p.config(ns)-p.config(s)))<1e-6 else 0;s=ns
  if te or same>4:break
 print((x,y,z),te,p.phase,p.xyz(s,s.get_object_from_name('part0')),s.get(p.robot,'grasp_active'),flush=True);e.close()
