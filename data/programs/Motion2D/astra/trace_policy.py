import sys,numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]);count=int(sys.argv[2]) if len(sys.argv)>2 else None
e=make_env();s,info=e.reset(seed=seed,options=None if count is None else {'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
print('initial',info,'boxes',p.boxes.tolist(), 'goal',p.goal)
for t in range(30):
 pos=np.array([s.get(p.robot,f) for f in ('x','y','theta')]);a=p.get_action(s);s,r,d,tr,i=e.step(a)
 new=np.array([s.get(p.robot,f) for f in ('x','y','theta')]); print(t,'pos',pos,'act',a,'new',new,'stuck',p.stuck,'path',[x.tolist() for x in p.path[:3]],flush=True)
 if d:break
e.close()
