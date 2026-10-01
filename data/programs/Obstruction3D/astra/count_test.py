import time,json
from env_client import make_env
from approach import GeneratedApproach
for count in [5,8,12]:
 for seed in [0]:
  e=make_env();s,info=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{})
  p.reset(s,info);start=time.monotonic();t=False
  for step in range(e.max_steps):
   s,r,t,tr,info=e.step(p.get_action(s))
   if t or tr:break
   if time.monotonic()-start>58:break
  result={'count':count,'seed':seed,'success':t,'steps':step+1,'seconds':round(time.monotonic()-start,3),'stage':p.stage,'index':p.index,'retry':p.retry}
  if not t:
   result['snapshot']={o.name:{f:round(float(s.get(o,f)),6) for f in e.observation_space.type_features[o.type]} for ty in e.observation_space.types for o in s.get_objects(ty)}
   result['goal']=p.goal.tolist() if p.goal is not None else None
  print(json.dumps(result),flush=True);e.close()
