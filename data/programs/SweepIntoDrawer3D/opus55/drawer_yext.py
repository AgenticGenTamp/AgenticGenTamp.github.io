from drawer_util import *
import sys
ys=[float(v) for v in sys.argv[1:]]
Rd=R_down(np.pi/2); r=None
for y in ys:
    if r is None or r.n>850: r=R(); r.grip=1.0
    bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
    q,e=r.ik([1.1,y,0.55],Rd); r.goto(q,50); q,e=r.ik([0.85,y,0.55],Rd); r.goto(q,30)
    z=0.55; hit=None
    while z>0.3:
        z-=0.01; q,e=r.ik([0.85,y,z],Rd); err=r.goto(q,4)
        if err>0.012: hit=r.tool()[2,3]; break
    print(f'y{y:+.2f} x0.85 hit {hit}',flush=True)
    q,e=r.ik([0.85,y,0.55],Rd); r.goto(q,30)
