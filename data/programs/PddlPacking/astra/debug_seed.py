from env_client import make_env
from approach import GeneratedApproach
from kinematics import fk
import numpy as np,sys
e=make_env();s,info=e.reset(seed=int(sys.argv[1]));p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
for b in s.get_objects(p.bt):print(b.name,p.pos(s,b),p.yaw(s,b))
for i in range(150):
 a=p.get_action(s);s,*_=e.step(a)
 if p.stuck>5:
  print('step',i,'stage',p.stage,'cur',p.conf(s),'goal',p.queue[0] if p.queue else None,'fk',fk(p.conf(s)[3:])[0]);break
e.close()
