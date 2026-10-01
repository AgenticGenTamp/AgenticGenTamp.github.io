from env_client import make_env
from approach import GeneratedApproach
import time,json
cases=[(seed,count) for count in [3,4] for seed in range(30,80)]
results=[]
errors={}
for seed,count in cases:
 if errors.get(count,0)>=2:continue
 e=make_env();start=time.monotonic()
 try:
  s,info=e.reset(seed=seed,options={'object_count':count} if count is not None else None)
  p=GeneratedApproach(e.action_space,e.observation_space,{})
  p.reset(s,info);t=False;tr=False
  for k in range(e.max_steps):
   a=p.get_action(s);s,r,t,tr,info=e.step(a)
   if t or tr:break
  result={'seed':seed,'count':count,'success':t,'steps':k+1,'time':round(time.monotonic()-start,3)}
  results.append(result)
  if not t:
   result['phase']=p.phase
   result['policy']={key:(val.name if hasattr(val,'name') else val) for key,val in p.__dict__.items() if key not in ['action_space','types']}
   result['state']={n:{f:s.get(s.get_object_from_name(n),f) for f in (['x','y','theta','arm_joint'] if n=='robot' else ['x','y','width','height'])} for n in s.get_object_names()}
   print('FAIL',json.dumps(result),flush=True)
  if len(results)%10==0:print('PROGRESS',len(results),'successes',sum(r['success'] for r in results),flush=True)
 except Exception as ex:
  errors[count]=errors.get(count,0)+1
  print('ERROR',seed,count,repr(ex),flush=True)
 finally:e.close()
print('AGGREGATE',json.dumps({'count':len(results),'successes':sum(r['success'] for r in results),'avgsteps':sum(r['steps'] for r in results)/len(results),'time':sum(r['time'] for r in results),'failures':[(r['seed'],r['count']) for r in results if not r['success']]}),flush=True)
