from util import *
import sys
e=E(1)
dx=float(sys.argv[1])
c=e.o('cube0')
e.base_to(c['pose_x']-0.5-dx, c['pose_y'])
e.ee_to([0.5,0,-0.1])
for dy in np.arange(-0.1,0.101,0.025):
  for z in [-0.19,-0.215,-0.235]:
    e.ee_to([0.5,dy,-0.1]); e.ee_to([0.5,dy,z])
    e.gripper(-1)
    r=e.rob()
    if r['grasp_active']>0 or r['finger_state']!=0:
        print('HIT',dx,dy,z,r['finger_state'],r['grasp_active'],[round(r[k],4) for k in RF[12:]], e.o('cube0'))
    e.gripper(1)
print('done',dx, e.rob()['finger_state'])
