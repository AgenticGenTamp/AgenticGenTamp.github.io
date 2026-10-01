from util import *
import sys
e=E(1)
b=e.o('box0')
off=float(sys.argv[1]); axis=sys.argv[2]
# approach box from -x side: base behind box
bx,by=b['pose_x'],b['pose_y']
if axis=='x': tx,ty=bx+off,by
else: tx,ty=bx,by+off
ok=e.base_to(tx-0.6, ty)
e.ee_to([0.5,0,0.1])
zs=0.1
for z in np.arange(0.1,-0.3,-0.005):
    ok2,_=e.ee_to([0.5,0,z],maxstep=0.02)
    if not ok2: break
    zs=z
print(axis,off,ok,'stop',round(zs,3),'world tipz',round(zs+0.2,3))
