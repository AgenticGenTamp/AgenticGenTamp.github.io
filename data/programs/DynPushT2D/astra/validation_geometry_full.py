import json,time,numpy as np
from env_client import make_env
from approach import GeneratedApproach,wrap
rows=[]
for seed in range(0,220):
 t=time.monotonic();e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);trace=[s.tolist()];success=False
 for step in range(e.max_steps):
  s,r,done,trunc,info=e.step(p.get_action(s));trace.append(s.tolist())
  if done or trunc:success=bool(done);break
 row=dict(revision="639abf8",seed=seed,success=success,steps=step+1,error=[float(s[0]-s[29]),float(s[1]-s[30]),float(wrap(s[2]-s[31]))],runtime=time.monotonic()-t);rows.append(row)
 if not success:
  json.dump(trace,open('failure_geometry_full_%d.json'%seed,'w'));print(row,flush=True)
 e.close()
 json.dump(rows,open('validation_geometry_full_summary.json','w'))
 if len(rows)%10==0:print('progress',len(rows),'success',sum(x['success'] for x in rows),flush=True)
print('aggregate',len(rows),sum(x['success'] for x in rows),'meansteps',np.mean([x['steps'] for x in rows]),'maxruntime',max(x['runtime'] for x in rows),flush=True)
