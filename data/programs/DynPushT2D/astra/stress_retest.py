from env_client import make_env
from approach import GeneratedApproach,wrap
import numpy as np,json,time
seeds=json.load(open('stress_seeds.json'));results=[]
seeds=[62305604]+[s for s in seeds if s!=62305604]
for seed in seeds:
 e=make_env();s,i=e.reset(seed=seed);a=GeneratedApproach(None,None,{});a.reset(s,i);hist=[];start=time.time()
 for k in range(e.max_steps):
  hist.append(s.tolist());s,r,t,tr,i=e.step(a.get_action(s))
  if t or tr:break
 row=dict(seed=seed,steps=k+1,success=bool(t),time=time.time()-start,err=(s[29:31]-s[:2]).tolist(),angle=wrap(s[31]-s[2]));results.append(row)
 if not t:
  print(row,flush=True);json.dump(hist+[s.tolist()],open('failure_stress_new_%d.json'%seed,'w'))
 json.dump(results,open('stress_summary_new.json','w'))
 if len(results)%20==0:print('stress',len(results),sum(x['success'] for x in results),flush=True)
 e.close()
