from env_client import make_env
import numpy as np

def dump(s):
 for n in sorted(s.get_object_names()):
  o=s.get_object_from_name(n)
  if n=='robot':
   print(n,[round(float(s.get(o,f)),4) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]+['pos_gripper']])
  elif n.startswith(('cube','wiper')):
   print(n,[round(float(s.get(o,f)),4) for f in ['x','y','z','bb_x','bb_y','bb_z']])
e=make_env();s,info=e.reset(seed=0);print('INITIAL',info);dump(s)
for j in [0,1,2,3,4,5,6,7,8,9,10]:
 a=np.zeros(11,dtype=np.float32);a[j]=0.1 if j<10 else 1
 for k in range(5):s,r,t,tr,i=e.step(a)
 print('AFTER',j,r,t,tr,i);dump(s)
e.close()
