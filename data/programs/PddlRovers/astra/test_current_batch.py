import sys,time,json
from env_client import make_env
from approach import GeneratedApproach
for seed in range(int(sys.argv[1]),int(sys.argv[2])):
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':4} if seed%4==0 else None)
 a=GeneratedApproach(e.action_space,e.observation_space,{});start=time.monotonic();a.reset(s,info)
 for step in range(200):
  s,_,term,trunc,_=e.step(a.get_action(s))
  if term or trunc:break
 print('RESULT',seed,info['object_count'],step+1,term,round(time.monotonic()-start,3),flush=True)
 if not term:
  print('FAILURE',a.goal,[(o.name,[s.get(o,f) for f in e.observation_space.type_features[o.type]]) for t in e.observation_space.types for o in s.get_objects(t)],flush=True)
 e.close()
