from env_client import make_env
import numpy as np

def dump(s):
 for name in s.get_object_names():
  o=s.get_object_from_name(name)
  fs=['x','y','theta','arm_joint','arm_length','finger_gap','base_radius','gripper_base_width','gripper_base_height','finger_height','finger_width'] if name=='robot' else ['x','y','theta','width','height','held']
  print(name,{f:round(s.get(o,f),4) for f in fs})

e=make_env();s,i=e.reset(seed=0);print('max',e.max_steps);dump(s)
for a,n in [([0,0,0,0,0],1),([.05,0,0,0,0],1),([0,.05,0,0,0],1),([0,0,.196,0,0],1),([0,0,0,.1,0],20),([0,0,0,-.1,0],30),([0,0,0,0,.02],20),([0,0,0,0,-.02],40),([0,.05,0,0,0],40),([.05,0,0,0,0],50)]:
 for _ in range(n): s,r,t,tr,i=e.step(np.array(a,dtype=np.float64)*0.999)
 print('action',a,'n',n,'term',t,tr); dump(s)
e.close()
