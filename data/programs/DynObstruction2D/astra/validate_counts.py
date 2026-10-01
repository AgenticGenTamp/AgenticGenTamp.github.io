from env_client import make_env
from approach import GeneratedApproach
import time
for count in [0,3,5,8,12]:
 for seed in range(5):
  e=make_env();s,info=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{})
  p.reset(s,info);start=time.time()
  for k in range(e.max_steps):
   s,r,t,tr,i=e.step(p.get_action(s))
   if t or tr:break
  target=s.get_object_from_name('target_block');surf=s.get_object_from_name('target_surface')
  print(count,seed,info,'result',t,k+1,round(time.time()-start,2),'phase',p.phase,'target',[round(s.get(target,f),3) for f in ['x','y','theta','width','height']],'goal',[round(s.get(surf,f),3) for f in ['x','width']],flush=True)
  e.close()
