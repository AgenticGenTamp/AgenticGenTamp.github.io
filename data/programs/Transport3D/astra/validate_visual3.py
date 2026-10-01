from env_client import make_env
from approach import GeneratedApproach
import json,time,numpy as np
for seed in [9,20]+list(range(30,50)):
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);hist=[];done=False;st=time.monotonic();last=None;unchanged=0
 for k in range(400):
  a=p.get_action(s)
  if not hist or hist[-1][1:3]!=[p.phase,p.target]:hist.append([k,p.phase,p.target])
  s,r,d,tr,_=e.step(a);rob=s.get_object_from_name('robot');cur=np.array([s.get(rob,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(j) for j in range(1,8)]])
  unchanged=unchanged+1 if last is not None and np.max(abs(cur-last))<1e-6 else 0;last=cur
  if d or tr or unchanged>=100:done=d;break
 objects={o.name:[round(s.get(o,'pose_'+c),4) for c in 'xyz']+[s.get(o,'grasp_active')] for o in s.get_objects(e.observation_space.get_type('Kinematic3DCuboid'))}
 print(json.dumps(dict(seed=seed,count=info.get('object_count'),steps=k+1,done=done,phase=p.phase,target=p.target,base=cur[:3].tolist(),q=cur[3:].tolist(),history=hist,objects=objects,time=round(time.monotonic()-st,2))),flush=True);e.close()
