from env_client import make_env
import numpy as np
import time

def joints(s):
 r=s.get_object_from_name('robot')
 return np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]+['pos_gripper']])

e=make_env(); s,info=e.reset(seed=0)
print('info',info,'initial',joints(s), flush=True)
for idx in [0,1,2,3,4,5,6,7,8,9,10]:
 s,info=e.reset(seed=0); start=joints(s)
 for k in range(3):
  a=np.zeros(11); a[idx]=.1 if idx!=10 else 1
  s,r,te,tr,info=e.step(a)
  print('idx',idx,'n',k,'diff',np.round(joints(s)-start,5),'r',r, flush=True)
 a=np.zeros(11)
 s,r,te,tr,info=e.step(a)
 print('zero after',np.round(joints(s)-start,5), flush=True)
e.close()
