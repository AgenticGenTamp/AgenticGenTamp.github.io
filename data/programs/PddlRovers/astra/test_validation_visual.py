import sys,time
from env_client import make_env
from approach import GeneratedApproach
for seed in map(int,sys.argv[1:] or range(10)):
 e=make_env();s,info=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{})
 start=time.monotonic();a.reset(s,info)
 for step in range(min(e.max_steps,300)):
  act=a.get_action(s);s,reward,term,trunc,inf=e.step(act)
  if '--trace' in []:pass
  if step%100==99:print('progress',seed,step+1,a.goal,[a.xy(s,r).round(2).tolist() for r in a.rovers],flush=True)
  if term or trunc:break
 print('RESULT',seed,info,step+1,term,trunc,'seconds',round(time.monotonic()-start,2),flush=True)
 if not term:
  print('goals',a.goal)
  for t in ['rover','objective','sample']:
   print(t,[(o.name,[round(s.get(o,f),3) for f in e.observation_space.type_features[e.observation_space.get_type(t)]]) for o in s.get_objects(e.observation_space.get_type(t))])
 e.close()
