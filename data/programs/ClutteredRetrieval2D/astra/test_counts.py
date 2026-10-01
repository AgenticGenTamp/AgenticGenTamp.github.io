from env_client import make_env
from approach import GeneratedApproach
import time,json,sys
for count in ([25] if len(sys.argv)>1 else [0,1,15,25]):
 for seed in ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else range(5)):
  e=make_env();start=time.monotonic();done=False;k=-1;compute=0.
  try:
   s,i=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);hist=[]
   for k in range(e.max_steps):
    if time.monotonic()-start>180 or compute>45:break
    tick=time.monotonic();a=p.get_action(s);compute+=time.monotonic()-tick;rpc=time.monotonic();s,r,done,trunc,inf=e.step(a);p.deadline+=time.monotonic()-rpc
    if k%100==0:hist.append({'step':k,'phase':p.phase,'chosen':p.chosen,'q':p.q.tolist(),'failures':dict(p.failures)})
    if done or trunc:break
   result={'seed':seed,'count':count,'info':i,'steps':k+1,'success':done,'compute':round(compute,3),'planning_time':round(p.planning_time,3),'seconds':round(time.monotonic()-start,3)}
   if not done:result.update(phase=p.phase,chosen=p.chosen,history=hist)
  except Exception as ex:result={'seed':seed,'count':count,'error':str(ex),'seconds':round(time.monotonic()-start,3)}
  e.close()
  with open('counts.jsonl','a') as f:f.write(json.dumps(result)+'\n')
  print(json.dumps(result),flush=True)
