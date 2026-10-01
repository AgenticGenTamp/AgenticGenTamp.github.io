from drawer_util import *
import sys
y=float(sys.argv[1]); z=float(sys.argv[2]); Rn=sys.argv[3]; xg=float(sys.argv[4])
Rd={'A':RA,'B':RB}[Rn]
r=R(); r.grip=0.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y
r.goto(r.q,40,base_d=bd)
q,e=r.ik([1.10,y,z],Rd); r.goto(q,100)
x=1.10
while x>xg+1e-6:
    x-=0.01; q,e=r.ik([x,y,z],Rd); err=r.goto(q,6)
    if err>0.015: break
r.goto(q,20); tx=r.tool()[:3,3].copy()
r.grip=1.0; r.hold(25); g=r.obs[135]
# pull back by moving base +x
res=[]
for k in range(12):
    bd=r.base.copy(); bd[0]+=0.03
    r.goto(q,5,base_d=bd); res.append(r.obs[103:109].max())
print(f'y{y} z{z} R{Rn} xg{xg} tool_at_grasp {tx} grip {g:.3f} dr {r.obs[103:109]} tool_end {r.tool()[:3,3]} qerr {np.abs(q-r.q).max():.3f}',flush=True)
