from drawer_util import *
import sys
y=float(sys.argv[1]); xh=float(sys.argv[2]); zh=float(sys.argv[3]); yaw=float(sys.argv[4])
Rd=R_down(yaw)
r=R(); r.grip=1.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
q,e=r.ik([1.1,y,0.52],Rd); r.goto(q,60)
q,e=r.ik([xh,y,0.52],Rd); r.goto(q,30)
z=0.52
while z>zh+1e-6:
    z-=0.01; q,e=r.ik([xh,y,z],Rd); err=r.goto(q,5)
    if err>0.012: break
r.goto(q,10); t0=r.tool()[:3,3].copy()
log=[]
for k in range(14):
    bd=r.base.copy(); bd[0]+=0.03; r.goto(q,4,base_d=bd); log.append(round(float(r.obs[103:109].max()),3))
print(f'y{y:+.2f} xh{xh} zh{zh} yaw{yaw:.2f} hook_at {t0} pull_log {log} dr {r.obs[103:109]} tool_end {r.tool()[:3,3]}',flush=True)
