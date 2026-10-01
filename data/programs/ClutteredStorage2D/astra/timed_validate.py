import time,sys,json
from env_client import make_env
from approach import GeneratedApproach
for seed in map(int,sys.argv[1:] or [73]):
 e=make_env();s,info=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info)
 start=time.perf_counter();policy=0;longest=0;done=False
 for i in range(e.max_steps):
  t=time.perf_counter();ac=a.get_action(s);dt=time.perf_counter()-t;policy+=dt;longest=max(longest,dt)
  s,r,done,trunc,_=e.step(ac)
  if done or trunc:break
 print(json.dumps(dict(seed=seed,count=info['object_count'],steps=i+1,success=done,wall=round(time.perf_counter()-start,3),policy=round(policy,3),max_action=round(longest,3),phase=a.phase,failures=a.failures)),flush=True);e.close()
