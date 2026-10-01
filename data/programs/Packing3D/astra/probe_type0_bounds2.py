from env_client import make_env
from approach import GeneratedApproach
import numpy as np
for xy in [(.3,.09),(.3,-.11),(.3,-.12)]:
 e=make_env();s,info=e.reset(seed=1,options={'object_count':1});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);same=0
 for i in range(45):
  a=p.get_action(s)
  if hasattr(p,'drop'):p.drop=np.array([*xy,.098])
  ns,r,te,tr,inf=e.step(a);same=same+1 if np.max(np.abs(p.config(ns)-p.config(s)))<1e-6 else 0;s=ns
  if te or same>4:break
 print(xy,te,p.phase,p.xyz(s,s.get_object_from_name('part0')),flush=True);e.close()
