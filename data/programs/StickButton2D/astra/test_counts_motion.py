from env_client import make_env
from approach import GeneratedApproach
import json,time
for count in [1,5,10,20]:
 for seed in [101,102,103,104,105]:
  e=make_env();s,info=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.monotonic();term=False
  for k in range(e.max_steps):
   a=p.get_action(s);s,r,term,trunc,info=e.step(a)
   if term or trunc or time.monotonic()-start>55:break
  rem=[n for n in s.get_object_names() if n.startswith('button') and s.get(s.get_object_from_name(n),'color_g')<.5]
  print(json.dumps({'count':count,'seed':seed,'steps':k+1,'success':term,'remaining':rem,'phase':p.phase,'time':round(time.monotonic()-start,3)}),flush=True);e.close()
