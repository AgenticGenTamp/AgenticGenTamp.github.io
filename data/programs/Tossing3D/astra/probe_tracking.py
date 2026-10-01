import numpy as np
from env_client import make_env
for gain in [0,5,10]:
 e=make_env();s,info=e.reset(seed=0);robot=s.get_object_from_name('robot')
 def q(s):return np.array([s.get(robot,'pos_arm_joint'+str(i)) for i in range(1,8)])
 target=q(s)+[0,.3,0,.3,0,.3,0]
 for t in range(24):
  err=target-q(s);a=np.zeros(18,np.float32);a[3:10]=np.clip(err,-.1,.1);a[11:18]=err*gain;a[10]=t%2
  s,r,done,tr,info=e.step(a)
  if t in [0,1,2,4,7,11,17,23]:print('gain',gain,'step',t+1,'err',np.round(target-q(s),4).tolist(),'grip',s.get(robot,'pos_gripper'),'info',info,flush=True)
 e.close()
