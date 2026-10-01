from env_client import make_env
from approach import GeneratedApproach
import numpy as np,time,json

e=make_env();policy=GeneratedApproach(e.action_space,e.observation_space,{})
try:
 for seed,count in [(987,1),(999,2),(1001,2)]:
  s,info=e.reset(seed=seed,options={'object_count':count});start=time.perf_counter();policy.reset(s,info);policy_seconds=time.perf_counter()-start
  for k in range(e.max_steps):
   t0=time.perf_counter();a=policy.get_action(s);policy_seconds+=time.perf_counter()-t0
   assert a.shape==e.action_space.shape and np.all(np.isfinite(a))
   assert np.all(a>=e.action_space.low) and np.all(a<=e.action_space.high)
   s,r,terminated,truncated,info=e.step(a)
   if terminated or truncated:break
  print(json.dumps({'seed':seed,'count':count,'success':terminated,'steps':k+1,'policy_seconds':policy_seconds,'total_seconds':time.perf_counter()-start}),flush=True)
finally:e.close()
