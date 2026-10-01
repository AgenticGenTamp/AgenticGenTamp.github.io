from util import *
from kin import rpy
import sys
e=E(1)
c=e.o('cube0')
off=float(sys.argv[1]); yaw=float(sys.argv[2]); th=float(sys.argv[3])
place_and_reach2(e,c['pose_x'],c['pose_y'],0.06,pre=0.1)
e.gripper(-1)
R=rpy(0,0,yaw)@Rdown
lin_to(e,[0.5,0,0.45-H],R)
b=e.o('box0'); bx,by=b['pose_x'],b['pose_y']
tx,ty=bx,by+off
print(e.base_to(tx-0.62*np.cos(th),ty-0.62*np.sin(th),th))
lin_to(e,[0.5,0,0.25-H],R)
zs=None
for z in np.arange(0.25,0.0,-0.005):
    if not lin_to(e,[0.5,0,z-H],R,seg=0.005): break
    zs=z
print(sys.argv[1:],'EE stop world z',zs, e.o('cube0'))
