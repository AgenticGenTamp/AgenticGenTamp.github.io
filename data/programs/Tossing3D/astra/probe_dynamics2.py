import numpy as np
from env_client import make_env
for j,v in [(0,10),(1,10),(3,10)]:
 e=make_env();s,info=e.reset(seed=0);robot=s.get_object_from_name('robot')
 def q(s):return np.array([s.get(robot,'pos_arm_joint'+str(i)) for i in range(1,8)])
 q0=q(s)
 for t in range(8):
  a=np.zeros(18,np.float32);a[11+j]=v if t==0 else 0
  s,r,done,tr,info=e.step(a)
  print('j',j+1,'step',t+1,'qdiff',np.round(q(s)-q0,4).tolist(),flush=True)
 e.close()
