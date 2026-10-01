from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=1);o=s.get_object_from_name('robot')
q0=s.get(o,'pos_arm_joint1')
for i in range(12):
 a=np.zeros(11,dtype=np.float32); a[3]=.1 if i in [0,4,5,6,7] else 0
 s,r,t,tr,info=E.step(a)
 print(i,a[3],round(s.get(o,'pos_arm_joint1')-q0,5),round(s.get(o,'vel_arm_joint1'),5),flush=True)
E.close()
