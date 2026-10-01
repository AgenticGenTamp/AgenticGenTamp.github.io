from util import *
e=E(1)
c=e.o('cube0')
print(place_and_reach2(e,c['pose_x'],c['pose_y'],0.06,pre=0.1))
e.gripper(-1); print('ga',e.rob()['grasp_active'])
lin_to(e,[0.5,0,0.45-H])
b=e.o('box0')
bx,by=b['pose_x'],b['pose_y']
# drive so that arm (0.5,0) is at box center
print(e.base_to(bx-0.62,by,0.0))
print(lin_to(e,[0.5,0,0.15-H]))
for z in np.arange(0.06,0.0,-0.002):
    ok=lin_to(e,[0.5,0,z+0.035-H],seg=0.005)
    e.gripper(1)
    cc=e.o('cube0'); print(round(z,3), ok, 'ga',e.rob()['grasp_active'], round(cc['pose_z'],4))
    if e.rob()['grasp_active']<0.5: break
lin_to(e,[0.5,0,0.4-H])
print(e.o('cube0'))
b=e.o('box0'); bx,by=b['pose_x'],b['pose_y']
print(place_and_reach2(e,bx-0.095,by,0.2,pre=0.1,yaw=1.5708))
e.gripper(-1); print('ga',e.rob()['grasp_active'])
from kin import rpy
R=rpy(0,0,1.5708)@Rdown
lin_to(e,[0.5,0,0.75-H],R)
print('box',e.o('box0')); print('cube',e.o('cube0'))
