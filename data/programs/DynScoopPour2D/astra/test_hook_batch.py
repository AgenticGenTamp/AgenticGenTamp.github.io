from env_client import make_env
from hook_policy import HookPolicy
import json
for seed,count in [(0,30),(150,50),(180,80),(1,30),(2,30)]:
 E=make_env();s,i=E.reset(seed=seed,options={'object_count':count});p=HookPolicy(E.observation_space);p.reset(s)
 for t in range(1000):
  s,r,term,trunc,i=E.step(p.act(s))
  if term or trunc:break
 smalls=[o for typ in ('small_circle','small_square') for o in s.get_objects(E.observation_space.get_type(typ))]
 print(json.dumps({'seed':seed,'count':count,'ok':term,'steps':t+1,'phase':p.phase,'held':s.get(p.h,'held'),'right':sum(s.get(o,'x')>1.8 for o in smalls)}),flush=True);E.close()
