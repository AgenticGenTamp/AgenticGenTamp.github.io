from env_client import make_env
from approach import GeneratedApproach
from kinova_candidate import fk
import numpy as np

e=make_env();s,i=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.mount=.35;p.reset(s,i)
for k in range(160):
 a=p.get_action(s)
 if p.t<55:a[0]=np.clip(s.get(p.cube,'x')-.65-s.get(p.robot,'pos_base_x'),-.1,.1)
 s,r,t,tr,i=e.step(a)
 if k%5==4 and k>74:
  q=[s.get(p.robot,'pos_arm_joint'+str(j)) for j in range(1,8)];xyz=np.array([s.get(p.cube,f) for f in ['x','y','z']]);base=np.array([s.get(p.robot,f) for f in ['pos_base_x','pos_base_y', 'pos_base_rot']]);pred=fk(q,.12,(0,0,.35))[0]+[base[0],base[1],0]
  print(k+1,'cube',xyz.round(4),'offset',(xyz-pred).round(4),'v',[round(s.get(p.cube,f),3) for f in ['vx','vy','vz']],flush=True)
e.close()
