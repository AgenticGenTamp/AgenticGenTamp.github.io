import numpy as np
from env_client import make_env
from approach import GeneratedApproach
for z in [.098,.10,.103,.105]:
 e=make_env();s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
 for i in range(35):
  a=p.get_action(s)
  if p.phase in ['lift','transfer','lower']:p.drop=np.array([.30,-.06,z])
  ns,r,te,tr,inf=e.step(a);s=ns
  if p.phase=='release':
   o=s.get_object_from_name('part0');print('z',z,'held',s.get(p.robot,'grasp_active'),'pos',p.xyz(s,o),'te',te,flush=True);break
 else: print('STUCK',z,p.phase,p.xyz(s,p.target),flush=True)
 e.close()
