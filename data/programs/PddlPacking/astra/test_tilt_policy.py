from env_client import make_env
from candidate_tilt_policy import GeneratedApproach
import time,sys,numpy as np
count=int(sys.argv[1]); seeds=[int(x) for x in sys.argv[2:]] or list(range(31))
for seed in seeds:
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':count} if count else None);p=GeneratedApproach(e.action_space,e.observation_space,{}) ;p.reset(s,info);start=time.time()
 for i in range(e.max_steps):
  a=p.get_action(s);s,r,t,tr,inf=e.step(a)
  if t or tr or p.stuck>8:break
 print('RESULT',seed,count,len(s.get_objects(p.bt)),i+1,t,tr,round(time.time()-start,2),p.stage,p.target.name if p.target else None,len(p.done),p.stuck,flush=True);e.close()
