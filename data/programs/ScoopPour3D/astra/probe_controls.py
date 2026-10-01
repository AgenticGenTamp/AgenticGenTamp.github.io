from env_client import make_env
import numpy as np

def robot(s):
 o=s.get_object_from_name('robot'); return np.array([s.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]+['pos_gripper']])
for i in range(11):
 e=make_env(); s,info=e.reset(seed=0)
 if i==0:
  print('INFO',info,flush=True)
  print('INIT ROBOT',robot(s).round(4),flush=True)
  for o in s.get_objects(e.observation_space.get_type('mujoco_movable_object')):
   print(o.name,[round(s.get(o,f),4) for f in ['x','y','z','bb_x','bb_y','bb_z']],flush=True)
 a=np.zeros(11,dtype=np.float32);a[i]=.1 if i<10 else 1
 old=robot(s)
 s,r,t,tr,inf=e.step(a)
 print('PROBE',i,'DELTA',(robot(s)-old).round(5),'r',r,'info',inf,flush=True)
 e.close()
