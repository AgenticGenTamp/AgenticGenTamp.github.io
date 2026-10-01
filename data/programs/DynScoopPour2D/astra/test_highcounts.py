from env_client import make_env
from policy_variants import GeneratedApproach
import json,time
for seed,count in [(180,80),(150,50)]:
 e=make_env();s,i=e.reset(seed=seed,options={'object_count':count} if count else None);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);start=time.monotonic()
 for step in range(e.max_steps):
  old=p.phase;s,_,term,trunc,_=e.step(p.get_action(s))
  if p.phase!=old and p.phase==1:
   print('cycle',seed,p.cycle,step+1,'right',sum(s.get(o,'x')>1.8 for o in p.smalls),flush=True)
  if term or trunc:break
 pts=[(round(s.get(o,'x'),3),round(s.get(o,'y'),3)) for o in p.smalls]
 print(json.dumps({'seed':seed,'count':len(pts),'ok':term,'steps':step+1,'right':sum(x>1.8 for x,y in pts),'time':round(time.monotonic()-start,1),'phase':p.phase,'cycle':p.cycle,'xy':pts if not term else []}),flush=True);e.close()
