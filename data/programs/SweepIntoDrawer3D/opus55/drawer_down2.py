from drawer_util import *
import sys
xt=float(sys.argv[1]); ys=[float(v) for v in sys.argv[2:]]
Rd=R_down(1.5708)
r=None
for y in ys:
    if r is None or r.n>800:
        r=R(); r.grip=1.0
    bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
    q,e=r.ik([1.1,y,0.46],Rd); r.goto(q,60)
    q,e=r.ik([xt,y,0.46],Rd); r.goto(q,30)
    z=0.46; hits=[]
    while z>0.06:
        z-=0.01; q,e=r.ik([xt,y,z],Rd); err=r.goto(q,5)
        if err>0.012:
            hits.append(round(r.tool()[2,3],3))
            q,e=r.ik([1.1,y,z],Rd); r.goto(q,25); z-=0.04
            q,e=r.ik([1.1,y,z],Rd); r.goto(q,25); q,e=r.ik([xt,y,z],Rd); r.goto(q,25)
    print(f'x {xt} y {y:+.2f} hits {hits}',flush=True)
    q,e=r.ik([1.1,y,0.46],Rd); r.goto(q,40)
