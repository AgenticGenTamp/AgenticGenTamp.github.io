from util import *
import sys
from kin import rpy
e=E(1)
c=e.o('cube0'); cx,cy=c['pose_x'],c['pose_y']
brot=float(sys.argv[1]); yaw=float(sys.argv[2]); m=0.12
R=rpy(0,0,yaw)@Rdown
# want arm-frame target (0.5,0); arm origin world = base + Rz(brot)*(m,0)
ax,ay=cx-0.5*np.cos(brot),cy-0.5*np.sin(brot)
bx,by=ax-m*np.cos(brot),ay-m*np.sin(brot)
print(e.base_to(bx,by,brot))
e.ee_to([0.5,0,-0.1],R); ok,_=e.ee_to([0.5,0,-0.2],R)
e.gripper(-1); r=e.rob()
print(sys.argv[1:],ok,'ga',r['grasp_active'],[round(r[k],4) for k in RF[12:]])
