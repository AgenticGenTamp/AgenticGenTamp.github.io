from env_client import make_env
from approach import GeneratedApproach,wrap
import numpy as np
import time,json
results=[]
revision="current_direction_preserving_arena_bounds"
for seed in [41,59,62,63,68]+list(range(80,120)):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);t=time.time();hist=[s.tolist()];term=False
 for k in range(e.max_steps):
  u=p.get_action(s);s,r,term,trunc,i=e.step(u);hist.append(s.tolist())
  if term or trunc:break
 elapsed=time.time()-t
 item={'revision':revision,'seed':seed,'success':bool(term),'steps':k+1,'error':(s[29:31]-s[:2]).tolist()+[wrap(s[31]-s[2])],'runtime':elapsed};results.append(item)
 if not term:
  with open('validation_failure_%d.json'%seed,'w') as f:json.dump(hist,f)
  print('FAIL',item,flush=True)
 if seed%10==9:print('progress',seed,'successes',sum(v['success'] for v in results),'max_steps',max(v['steps'] for v in results),'max_time',max(v['runtime'] for v in results),flush=True)
 with open('validation_probe_updated_summary.json','w') as f:json.dump(results,f)
 e.close()
print('DONE',len(results),sum(v['success'] for v in results),'mean_steps',np.mean([v['steps'] for v in results]),flush=True)
