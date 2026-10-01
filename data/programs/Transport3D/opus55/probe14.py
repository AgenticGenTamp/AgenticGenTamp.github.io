from util import *
import sys
e=E(1)
b=e.o('box0')
off=float(sys.argv[1]); z=float(sys.argv[2])
bx,by=b['pose_x'],b['pose_y']
e.base_to(bx+off-0.6, by)
e.ee_to([0.5,0,0.1]); ok,_=e.ee_to([0.5,0,z],maxstep=0.02)
e.gripper(-1)
r=e.rob()
print(off,z,ok,'ga',r['grasp_active'],'fs',round(r['finger_state'],3),[round(r[k],4) for k in RF[12:]])
