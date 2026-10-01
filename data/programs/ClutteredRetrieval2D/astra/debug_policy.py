from env_client import make_env
from approach import GeneratedApproach
import sys,time,numpy as np
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
for k in range(200):
 a=p.get_action(s);s,_,done,_,_=e.step(a)
 if k%5==0:print(k,p.phase,p.chosen,'q',np.round(p.q,3),'action',np.round(a,3),'path',len(p.path),'held',p.held,'plansec',round(p.planning_time,2),flush=True)
 if done:print('SUCCESS',k);break
e.close()
