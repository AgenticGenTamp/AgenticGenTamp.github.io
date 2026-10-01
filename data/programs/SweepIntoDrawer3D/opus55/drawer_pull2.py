from drawer_util import *
import sys
Rn=sys.argv[1]; xg=float(sys.argv[2]); jobs=[tuple(map(float,s.split(','))) for s in sys.argv[3:]]
Rd={'A':RA,'B':RB}[Rn]
for (y,z) in jobs:
    r=R(); r.grip=0.0
    bd=r.base.copy(); bd[0]=1.6; bd[1]=y
    r.goto(r.q,40,base_d=bd)
    q,e=r.ik([1.10,y,0.3],Rd); r.goto(q,90)
    q,e=r.ik([1.10,y,z],Rd); r.goto(q,30)
    x=1.10
    while x>xg+1e-6:
        x-=0.01; q,e=r.ik([x,y,z],Rd); err=r.goto(q,6)
        if err>0.015: break
    r.goto(q,15); tx=r.tool()[:3,3].copy()
    r.grip=1.0; r.hold(20); g=r.obs[135:147].copy()
    for k in range(8):
        bd=r.base.copy(); bd[0]+=0.03; r.goto(q,5,base_d=bd)
    d=r.obs[103:109]
    print(f'R{Rn} y{y:+.2f} z{z:.2f} tool@grasp {tx} dr {d} {"***" if d.max()>0.01 else ""}',flush=True)
