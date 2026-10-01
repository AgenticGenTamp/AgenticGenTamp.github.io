import time
from env_client import make_env
from approach import GeneratedApproach
for count in [2,3]:
 for seed in range(5):
  e=make_env()
  try:
   s,info=e.reset(seed=seed,options={'object_count':count})
   a=GeneratedApproach(e.action_space,e.observation_space,{})
   start=time.monotonic();a.reset(s,info);term=trunc=False
   for step in range(200):
    s,_,term,trunc,_=e.step(a.get_action(s))
    if term or trunc:break
   print('COUNT_RESULT',count,seed,info,step+1,term,trunc,round(time.monotonic()-start,2),flush=True)
   if not term:
    print('FAIL_GOALS',a.goal,[(o.name,s.get(o,'received_image')) for o in a.objs],flush=True)
  except Exception as exc:
   print('COUNT_EXCEPTION',count,seed,type(exc).__name__,str(exc),flush=True)
  finally:e.close()
