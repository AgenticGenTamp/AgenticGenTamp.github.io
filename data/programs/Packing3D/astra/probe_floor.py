import numpy as np
from env_client import make_env
from approach import GeneratedApproach
e=make_env();s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
for i in range(65):
 a=p.get_action(s)
 if p.phase in ['lift','transfer','lower']:p.drop=np.array([.30,0,.095])
 ns,r,te,tr,inf=e.step(a)
 print(i,p.phase,'act',np.round(a,3),'part',p.xyz(ns,p.target).round(5) if p.target else None,'held',ns.get(p.robot,'grasp_active'),'q',p.config(ns)[[0,1,4,8]].round(4),te,flush=True)
 s=ns
 if te:break
e.close()
