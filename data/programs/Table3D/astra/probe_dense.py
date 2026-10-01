from env_client import make_env
from approach import GeneratedApproach
import time,json
E=make_env();P=GeneratedApproach(E.action_space,E.observation_space,{})
results=[]
for count,seeds in [(10,range(100,300)),(6,range(20)),(8,range(20))]:
 for seed in seeds:
  s,info=E.reset(seed=seed,options={'object_count':count});P.reset(s,info);t=time.time();trace=[]
  for step in range(1000):
   a=P.get_action(s);s,re,te,tr,inf=E.step(a)
   if step<15 or step%50==0:trace.append((step+1,[float(x) for x in a]))
   if te or tr:break
  row={'seed':seed,'count':count,'success':bool(te),'steps':step+1,'elapsed':round(time.time()-t,3)}
  if not te:
   row['state']={o.name:{f:float(s.get(o,f)) for f in E.observation_space.type_features[ty]} for ty in E.observation_space.types for o in s.get_objects(ty)}
   row['trace']=trace
  results.append(row);print(json.dumps(row),flush=True)
E.close()
print('SUMMARY',len(results),sum(r['success'] for r in results),flush=True)
