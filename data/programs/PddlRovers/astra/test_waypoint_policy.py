import time
from env_client import make_env
from waypoint_policy import GeneratedApproach
for seed in range(30):
 e=make_env();s,info=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{})
 start=time.monotonic();a.reset(s,info)
 for step in range(180):
  s,_,term,trunc,_=e.step(a.get_action(s))
  if term or trunc:break
 print('OPT_RESULT',seed,step+1,term,round(time.monotonic()-start,2),a.schedule,flush=True)
 e.close()
