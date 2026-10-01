from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();P=GeneratedApproach(E.action_space,E.observation_space,{})
for dx in [0,.005,-.005,-.01]:
 for dy in [.005,.01,.015,.02,-.01,-.02]:
  s,inf=E.reset(seed=26,options={'object_count':10});P.reset(s,inf)
  for i in range(12):
   a=P.get_action(s)
   if not s.get(P.r,'grasp_active'):
    # Replace base target with constant displacement from measured nominal.
    a[0]=np.clip(s.get(P.c,'pose_x')-.44153585+dx-s.get(P.r,'pos_base_x'),-.4,.4)
    a[1]=np.clip(s.get(P.c,'pose_y')-.00135+dy-s.get(P.r,'pos_base_y'),-.4,.4)
   s,re,te,tr,info=E.step(a)
   if te:break
  print(dx,dy,te,i+1,s.get(P.r,'grasp_active'),flush=True)
E.close()
