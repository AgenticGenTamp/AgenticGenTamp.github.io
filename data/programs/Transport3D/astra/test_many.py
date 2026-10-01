from env_client import make_env
from approach import GeneratedApproach
import sys,time,json
counts=list(map(int,sys.argv[1:] or ['3','4','5','6','8','12']))
for n in counts:
 e=make_env();s,i=e.reset(seed=42,options={'object_count':n});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);start=time.time()
 for k in range(1000):
  s,r,t,tr,info=e.step(p.get_action(s))
  if t or tr:break
 print('COUNT',n,'steps',k+1,'success',t,'phase',p.phase,'target',p.target,'sec',round(time.time()-start,2),flush=True)
 if not t:
  print([(o.name,[round(s.get(o,'pose_'+v),3) for v in 'xyz']) for o in s.get_objects(e.observation_space.get_type('Kinematic3DCuboid'))],flush=True)
 e.close()
