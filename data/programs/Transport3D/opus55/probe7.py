from util import *
import sys
e=E(1)
dx=float(sys.argv[1]); dy=float(sys.argv[2]); z=float(sys.argv[3])
c=e.o('cube0')
print(e.base_to(c['pose_x']-0.5-dx, c['pose_y']-dy))
print(e.ee_to([0.5,0,0.0]), e.ee_to([0.5,0,z]))
e.gripper(-1); e.gripper(-1)
r=e.rob(); print(sys.argv[1:], 'fs',r['finger_state'],'ga',r['grasp_active'],[round(r[k],4) for k in RF[12:]], e.o('cube0'))
