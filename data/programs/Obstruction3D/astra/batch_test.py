from env_client import make_env
from approach import GeneratedApproach
import sys,time,json
seeds=[int(x) for x in sys.argv[1:]] or list(range(20))
for seed in seeds:
 E=make_env();s,info=E.reset(seed=seed);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info);start=time.time();term=False
 for t in range(E.max_steps):
  a=p.get_action(s);s,r,term,trunc,inf=E.step(a)
  if term or trunc:break
 result={'seed':seed,'n':info.get('object_count'),'steps':t+1,'success':term,'time':round(time.time()-start,2),'stage':p.stage,'index':p.index,'retry':p.retry}
 print(json.dumps(result),flush=True)
 if not term:
  with open('failure_'+str(seed)+'.json','w') as f:json.dump({'result':result,'objects':{o.name:{v:s.get(o,v) for v in E.observation_space.type_features[o.type]} for typ in E.observation_space.types for o in s.get_objects(typ)},'goal':None if p.goal is None else p.goal.tolist()},f)
 E.close()
