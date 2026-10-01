from drawer_util import *
import sys
Rn=sys.argv[1]; y=float(sys.argv[2]); z=float(sys.argv[3]); xg=float(sys.argv[4])
Rd={'A':RA,'B':RB}[Rn]
r=R(); r.grip=0.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
q,e=r.ik([1.10,y,0.3],Rd); r.goto(q,90)
q,e=r.ik([1.10,y,z],Rd); r.goto(q,30)
x=1.10
while x>xg+1e-6:
    x-=0.005; q,e=r.ik([x,y,z],Rd); err=r.goto(q,4)
    if err>0.012: break
r.goto(q,15); tx=r.tool()[:3,3].copy()
r.grip=1.0; r.hold(25); g=r.obs[135]
for k in range(10):
    bd=r.base.copy(); bd[0]+=0.03; r.goto(q,5,base_d=bd)
print(f'R{Rn} y{y:+.2f} z{z:.3f} xg{xg} tool@grasp {tx} g {g:.2f} dr {r.obs[103:109]} tool_end {r.tool()[:3,3]}',flush=True)
