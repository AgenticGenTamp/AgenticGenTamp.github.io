from drawer_util import *
import sys
y=float(sys.argv[1]); xs=[float(v) for v in sys.argv[2:]]
Rd=R_down(1.5708)
r=R(); r.grip=1.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
for xt in xs:
    if r.n>850:
        r=R(); r.grip=1.0; r.goto(r.q,40,base_d=bd)
    q,e=r.ik([1.1,y,0.52],Rd); r.goto(q,60)
    q,e=r.ik([xt,y,0.52],Rd); r.goto(q,30)
    z=0.52; hits=[]
    while z>0.06:
        z-=0.01; q,e=r.ik([xt,y,z],Rd); err=r.goto(q,5)
        if err>0.012:
            hits.append(round(r.tool()[2,3],3)); break
    print(f'y {y:+.2f} x {xt} hits {hits} dr {r.obs[103:109]}',flush=True)
