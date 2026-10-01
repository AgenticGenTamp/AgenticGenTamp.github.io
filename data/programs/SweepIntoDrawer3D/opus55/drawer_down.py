from drawer_util import *
import sys
y=float(sys.argv[1]); yaw=float(sys.argv[2]); xs=[float(v) for v in sys.argv[3:]]
r=R(); r.grip=1.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
Rd=R_down(yaw)
q,e=r.ik([1.15,y,0.55],Rd); r.goto(q,90)
for xt in xs:
    q,e=r.ik([xt,y,0.46],Rd); r.goto(q,30)
    z=0.46; hit=None
    while z>0.06:
        z-=0.01; q,e=r.ik([xt,y,z],Rd); err=r.goto(q,6)
        if err>0.012: hit=r.tool()[2,3]; break
    print(f'y {y} yaw {yaw:.2f} x {xt} top-hit z {hit} tool {r.tool()[:3,3]} dr {r.obs[103:109]}',flush=True)
    q,e=r.ik([xt,y,0.46],Rd); r.goto(q,40)
