from env_client import make_env
from approach import GeneratedApproach
import sys,time,json
start,end=map(int,sys.argv[1:3]);results=[]
for seed in range(start,end):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);now=time.time();compute=0
 for t in range(1000):
  q=time.time();a=p.get_action(s);compute+=time.time()-q;s,r,d,tr,i=e.step(a)
  if d or tr:break
 result=dict(seed=seed,steps=t+1,success=bool(d),phase=p.phase,time=round(time.time()-now,3),compute=round(compute,3))
 print(result,flush=True);results.append(result)
 if not d:
  with open('failure_%d.json'%seed,'w') as f:json.dump(s.tolist(),f)
 e.close()
with open('batch_%d_%d.json'%(start,end),'w') as f:json.dump(results,f)
print('TOTAL',sum(x['success'] for x in results),len(results),flush=True)
