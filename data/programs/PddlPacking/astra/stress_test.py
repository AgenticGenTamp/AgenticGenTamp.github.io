from env_client import make_env
from stress_policy import GeneratedApproach
import time,json,numpy as np

results=[]
for seed,count in [(s,None) for s in range(180,401)]:
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':count} if count else None)
 p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();t=False
 for i in range(e.max_steps):
  a=p.get_action(s);s,r,t,tr,inf=e.step(a)
  if t or tr or p.stuck>20:break
 result={'seed':seed,'count':info['object_count'],'steps':i+1,'ok':bool(t),'runtime':round(time.time()-start,3)}
 results.append(result)
 print('RESULT',json.dumps(result),flush=True)
 if not t:
  print('FAIL',json.dumps(dict(result,stage=p.stage,target=p.target.name if p.target else None,done=list(p.done),conf=p.conf(s).tolist(),goal=p.queue[0][0].tolist() if p.queue else None,grip=p.queue[0][1] if p.queue else None,blocks={b.name:p.pos(s,b).tolist() for b in s.get_objects(p.bt)})),flush=True)
 e.close()
print('SUMMARY',json.dumps({'total':len(results),'successes':sum(r['ok'] for r in results),'mean_steps':np.mean([r['steps'] for r in results]),'runtime':sum(r['runtime'] for r in results),'max_runtime':max(r['runtime'] for r in results)}),flush=True)
