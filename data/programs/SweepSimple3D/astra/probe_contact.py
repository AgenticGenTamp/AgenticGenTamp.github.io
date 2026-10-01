from env_client import make_env
import numpy as np
for grip in [0,1]:
 e=make_env();s,info=e.reset(seed=0)
 robot=s.get_object_from_name('robot'); w=s.get_object_from_name('wiper_0')
 def row():
  return [round(float(s.get(robot,f)),3) for f in ['pos_base_x','pos_base_y','pos_base_rot','pos_gripper']]+[round(float(s.get(w,f)),3) for f in ['x','y','z']]+[[n]+[round(float(s.get(s.get_object_from_name(n),f)),3) for f in ['x','y']] for n in s.get_object_names() if n.startswith('cube')]
 print('TRIAL',grip,'INIT',row())
 for k in range(25):
  a=np.zeros(11,dtype=np.float32);a[1]=-.05;a[10]=grip
  s,r,t,tr,i=e.step(a)
  print(k,r,t,row())
 e.close()
