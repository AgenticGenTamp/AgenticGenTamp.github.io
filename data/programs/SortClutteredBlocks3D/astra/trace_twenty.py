from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();s,info=E.reset(seed=0,options={'object_count':20});p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info)
print('INITIAL',[(o.name,p.xyz(s,o).round(3).tolist()) for o in p.cubes],flush=True)
for i in range(1000):
 old=(p.obj.name if p.obj else None,p.stage)
 a=p.get_action(s);s,r,d,tr,_=E.step(a)
 new=(p.obj.name if p.obj else None,p.stage)
 if new!=old:
  rb=p.robot
  print(i,old,'->',new,'POS',p.xyz(s,s.get_object_from_name(old[0])).round(3).tolist() if old[0] else None,'BASE',[round(s.get(rb,f),3) for f in ['pos_base_x','pos_base_y','pos_base_rot']],'target',p.base.round(3).tolist(),'DONE',sum(p.done(s,o) for o in p.cubes),flush=True)
 if d or tr:print('END',i,d,tr,flush=True);break
E.close()
