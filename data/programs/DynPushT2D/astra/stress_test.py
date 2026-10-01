from env_client import make_env
from approach import GeneratedApproach,rot,wrap
import numpy as np,json,time
rng=np.random.default_rng(9876);e=make_env();cases=[]
for seed in rng.integers(10000000,100000000,1500):
 s,_=e.reset(seed=int(seed));a=GeneratedApproach(None,None,{});a.reset(s,{});a.geometry(s)
 # Focus on poses whose stem or required approach lies near the arena boundary.
 pts=s[:2]+a.q@rot(s[2]).T;ptsgoal=s[29:31]+a.q@rot(s[31]).T
 clearance=min(pts.min(),5-pts.max(),ptsgoal.min(),5-ptsgoal.max())
 difficulty=-clearance+.03*np.linalg.norm(s[:2]-s[29:31])+.015*abs(wrap(s[2]-s[31]))
 cases.append((difficulty,int(seed)))
e.close();cases.sort(reverse=True);seeds=[seed for _,seed in cases[:120]]
json.dump(seeds,open('stress_seeds.json','w'))
results=[]
for seed in seeds:
 e=make_env();s,i=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,i);hist=[];start=time.time()
 for k in range(e.max_steps):
  hist.append(s.tolist());s,r,t,tr,i=e.step(a.get_action(s))
  if t or tr:break
 row=dict(seed=seed,steps=k+1,success=bool(t),time=time.time()-start,err=(s[29:31]-s[:2]).tolist(),angle=wrap(s[31]-s[2]));results.append(row)
 if not t:
  print(row,flush=True);json.dump(hist+[s.tolist()],open('failure_stress_%d.json'%seed,'w'))
 json.dump(results,open('stress_summary.json','w'))
 if len(results)%20==0:print('stress',len(results),sum(x['success'] for x in results),flush=True)
 e.close()
