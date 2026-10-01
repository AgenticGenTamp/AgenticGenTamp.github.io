from env_client import make_env
from candidate import GeneratedApproach
from approach import wrap
import numpy as np
import sys,time,json
start=int(sys.argv[1]) if len(sys.argv)>1 else 220
count=int(sys.argv[2]) if len(sys.argv)>2 else 100
rng=np.random.default_rng(start)
seeds=[int(x) for x in rng.integers(1000,10000000,count)]
results=[]
for seed in seeds:
 e=make_env();s,i=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,i);t=time.time();hist=[]
 for k in range(e.max_steps):
  hist.append(s.tolist());u=a.get_action(s);s,r,term,trunc,i=e.step(u)
  if term or trunc:break
 row=dict(seed=seed,steps=k+1,success=bool(term),err=(s[29:31]-s[:2]).tolist(),angle=wrap(s[31]-s[2]),time=time.time()-t);results.append(row)
 if not term:
  print(row,flush=True)
  with open('failure_candidate_%d.json'%seed,'w') as f:json.dump(hist+[s.tolist()],f)
 if len(results)%20==0:print('progress',len(results),'successes',sum(x['success'] for x in results),flush=True)
 e.close()
with open('candidate_summary.json','w') as f:json.dump(results,f)
print('TOTAL',sum(x['success'] for x in results),len(results),'avg',np.mean([x['steps'] for x in results]),'max',max(x['steps'] for x in results),flush=True)
