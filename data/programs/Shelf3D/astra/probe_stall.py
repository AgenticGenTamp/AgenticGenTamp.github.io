from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();s,info=E.reset(seed=11,options={'object_count':8});p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info)
for i in range(760):
 a=p.get_action(s);s,r,t,tr,info=E.step(a)
 if i>=610 and i%10==0:
  q=np.array([s.get(p.robot,'pos_arm_joint'+str(j)) for j in range(1,8)]);b=np.array([s.get(p.robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
  print(i,p.phase,p.age,'qerr',(p.target-q).round(3),'berr',(p.base-b).round(3),'cube',[round(s.get(p.obj,f),3) for f in ['x','y','z']],flush=True)
 if t:print('SUCCESS',i,flush=True);break
E.close()
