from env_client import make_env
from approach import GeneratedApproach
from kinematics import fk
import numpy as np,sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
e=make_env();s,i=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,i)
print('candidates',len(a.candidates), 'g',a.g,'b',a.b,'yaw',a.yaw,flush=True)
for n in range(60):
 ac=a.get_action(s);s,_,t,tr,_=e.step(ac);p=np.array([s.get(a.r,f) for f in a.fs])
 if n>23:
  print(n,'qleft',len(a.queue),'p',p.round(3),'tool',fk(p)[0].round(4),'grip',s.get(a.r,'gripper_opening'),'command',ac.round(3),flush=True)
 if t:break
e.close()
