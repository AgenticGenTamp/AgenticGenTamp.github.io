from env_client import make_env
from approach import GeneratedApproach
from kinematics import fk,HOME
import numpy as np
E=make_env();s,info=E.reset(seed=42);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info)
for i in range(240):
 a=p.get_action(s)
 if i>95:a[-1]=1
 if i==120:p.target=HOME.copy()
 s,r,t,tr,info=E.step(a)
 if i%20==0:
  o=p.objects[0];q=np.array([s.get(p.robot,'pos_arm_joint'+str(j)) for j in range(1,8)])
  print(i,'cube',[round(s.get(o,f),4) for f in ['x','y','z']], 'fk',np.round(fk(q)[0],3),'r',r,'t',t,flush=True)
E.close()
