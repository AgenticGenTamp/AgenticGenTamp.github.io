import numpy as np
from env_client import make_env
from approach import GeneratedApproach
for seed in [0,1]:
 for x,y in [(0.30,0),(0.27,0),(0.27,-.06),(.28,.03),(.25,-.05)]:
  e=make_env();s,info=e.reset(seed=seed,options={'object_count':1});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
  for i in range(45):
   a=p.get_action(s)
   if p.phase in ['lift','transfer','lower']:p.drop=np.array([x,y,.1])
   ns,r,te,tr,inf=e.step(a);s=ns
   if te:break
  print(seed,x,y,'result',te,p.phase,p.xyz(s,s.get_object_from_name('part0')).round(4),flush=True)
  e.close()
