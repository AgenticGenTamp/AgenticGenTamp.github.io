from util import *
import sys
e=E(1)
b=e.o('box0'); bx,by=b['pose_x'],b['pose_y']
dx=float(sys.argv[1]); z=float(sys.argv[2])
print(place_and_reach2(e,bx+dx,by,z,pre=0.3))
e.gripper(-1); r=e.rob()
print(sys.argv[1:],'ga',r['grasp_active'],[round(r[k],4) for k in RF[12:]], np.round(fk_arm(e.q())[:3,3],3))
