from env_client import make_env
from approach import GeneratedApproach
from kinematics import fk
import numpy as np
for seed in [0,1,3]:
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
 for i in range(400):
  old=(p.obj,p.stage);a=p.get_action(s)
  if old[1]==3 and p.stage==4:
   rb=p.robot;xyz=p.xyz(s,old[0]);base=np.array([s.get(rb,f) for f in ['pos_base_x','pos_base_y', 'pos_base_rot']]);q=np.array([s.get(rb,'pos_arm_joint'+str(j)) for j in range(1,8)]);pt,rr=fk(q);pt+=rr@np.array([0,0,.12]);c,ss=np.cos(base[2]),np.sin(base[2]);rot=np.array([[c,-ss],[ss,c]]);local=rot.T@(xyz[:2]-base[:2]);print(seed,i,old[0].name,'MOUNT',np.r_[local-pt[:2],xyz[2]-pt[2]].round(5),'QERR',np.max(abs(q-p.ik(p.highz))),flush=True)
  s,r,d,tr,_=e.step(a)
  if d or tr:print('END',seed,i,d,flush=True);break
 e.close()
