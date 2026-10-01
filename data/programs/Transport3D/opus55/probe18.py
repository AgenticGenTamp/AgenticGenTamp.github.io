from util import *
import sys
e=E(1)
sx,sy,rot,dx,dy=map(float,sys.argv[1:6])
e.base_to(sx,sy,rot)
for i in range(100):
    r=e.rob(); a=np.zeros(11); a[0]=dx; a[1]=dy; e.step(a); r2=e.rob()
    if abs(r2['pos_base_x']-r['pos_base_x'])+abs(r2['pos_base_y']-r['pos_base_y'])<1e-9: break
print(sys.argv[1:], 'stop at', round(r2['pos_base_x'],3), round(r2['pos_base_y'],3))
