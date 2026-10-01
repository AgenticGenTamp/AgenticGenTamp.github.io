from drawer_util import *
import sys
yaw=float(sys.argv[1]); xt=float(sys.argv[2]); y=-0.3
r=R(); r.grip=0.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
Rd=R_down(yaw)
q,e=r.ik([1.15,y,0.6],Rd); r.goto(q,90)
q,e=r.ik([xt,y,0.6],Rd); r.goto(q,40)
z=0.6
while z>0.2:
    z-=0.01; q,e=r.ik([xt,y,z],Rd); err=r.goto(q,6)
    if err>0.015: break
print(f'yaw {yaw:.2f} x {xt} blocked at tool z {r.tool()[2,3]:.3f}',flush=True)
