from env_client import make_env
from approach import GeneratedApproach
import json,time
for count in [4,8]:
 e=make_env();s,info=e.reset(seed=20,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);hist=[];st=time.monotonic()
 for k in range(1000):
  a=p.get_action(s)
  if not hist or hist[-1][1:3]!=[p.phase,p.target]:hist.append([k,p.phase,p.target])
  s,r,d,tr,_=e.step(a)
  if d or tr:break
 print(json.dumps(dict(requested=count,info=info,steps=k+1,done=d,phase=p.phase,target=p.target,history=hist,time=time.monotonic()-st)),flush=True);e.close()
