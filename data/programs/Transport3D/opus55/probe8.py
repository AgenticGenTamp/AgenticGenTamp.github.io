from util import *
import sys
e=E(1)
dx=float(sys.argv[1]); dy=float(sys.argv[2])
c=e.o('cube0')
e.base_to(c['pose_x']-0.5-dx, c['pose_y']-dy)
e.ee_to([0.5,0,-0.1])
zs=-0.1
for z in np.arange(-0.1,-0.3,-0.005):
    ok,_=e.ee_to([0.5,0,z],maxstep=0.02)
    if not ok: break
    zs=z
print(dx,dy,'stop',round(zs,3), e.o('cube0')['pose_x'])
