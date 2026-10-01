from probe_hook_lib import *
import time
p=P(44); o=p.obs.get_object_from_name('robot')
for f in ['arm_length','base_radius','gripper_base_width','gripper_base_height','finger_gap','finger_height','finger_width','arm_joint']: print(f, p.obs.get(o,f))
h=p.obs.get_object_from_name('hook')
for f in ['width','length_side1','length_side2','mass','static']: print(f,p.obs.get(h,f))
t=time.time(); p.step([0,0.03,0,0.08,0.015],10); print('10 steps',time.time()-t, p.r())
