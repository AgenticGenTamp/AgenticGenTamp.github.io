from env_client import make_env
from approach import GeneratedApproach
import json,time
for seed in range(3,31):
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);hist=[];done=False;st=time.monotonic()
 for k in range(1000):
  a=p.get_action(s)
  if not hist or hist[-1][1:3]!=[p.phase,p.target]:hist.append([k,p.phase,p.target])
  s,r,d,tr,_=e.step(a)
  if d or tr:done=d;break
 r=s.get_object_from_name('robot');objects={o.name:[round(s.get(o,'pose_'+c),4) for c in 'xyz']+[s.get(o,'grasp_active')] for o in s.get_objects(e.observation_space.get_type('Kinematic3DCuboid'))}
 print(json.dumps(dict(seed=seed,count=info.get('object_count'),steps=k+1,done=done,phase=p.phase,target=p.target,base=[s.get(r,'pos_base_x'),s.get(r,'pos_base_y')],history=hist,objects=objects,time=round(time.monotonic()-st,2))),flush=True);e.close()
