from env_client import make_env
import numpy as np
for damping in [0.,.15]:
 e=make_env();s,_=e.reset(seed=0);o=s.get_object_from_name('robot');target=s.get(o,'pos_arm_joint1')+.5
 for i in range(60):
  q=s.get(o,'pos_arm_joint1');v=s.get(o,'vel_arm_joint1')
  a=np.zeros(11,dtype=np.float32);a[3]=np.clip((target-q)*2.5+damping*v,-.1,.1)
  s,*_=e.step(a)
  if i in [0,10,20,30,40,50,59]:print(damping,i,round(s.get(o,'pos_arm_joint1'),5),round(s.get(o,'vel_arm_joint1'),5),flush=True)
 e.close()
