from drawer_util import *
import sys
y=float(sys.argv[1]); pull=float(sys.argv[2]); probe=len(sys.argv)>3
Rd=R_down(np.pi/2)
r=R(); r.grip=1.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
q,e=r.ik([1.1,y,0.50],Rd); r.goto(q,60)
q,e=r.ik([0.917,y,0.50],Rd); r.goto(q,30)
z=0.50
while z>0.31:
    z-=0.01; q,e=r.ik([0.917,y,z],Rd); r.goto(q,5)
r.goto(q,10); t0=r.tool()[:3,3].copy()
n0=r.n
for k in range(int(pull/0.03)):
    bd=r.base.copy(); bd[0]+=0.03; r.goto(q,4,base_d=bd)
v=r.obs[103:109].copy(); tl=r.tool()[:3,3].copy()
# release: lift
for dz in np.arange(0.02,0.25,0.02):
        q2,e=r.ik(tl+np.array([0,0,dz]),Rd); r.goto(q2,5)
r.hold(20)
print(f'y{y:+.2f} hook_at {t0} after pull dr {v} tool {tl} steps_pull {r.n-n0} after release+hold dr {r.obs[103:109]} base {r.base} rew {set(r.rews)}',flush=True)
if probe:
    b=r.base.copy()
    for (px,py) in [(0.95,y),(1.0,y),(1.1,y),(1.2,y),(1.25,y),(1.27,y),(1.29,y),(1.1,y+0.2),(1.1,y+0.26),(1.1,y+0.30),(1.1,y-0.2),(1.1,y-0.26),(1.1,y-0.30)]:
        q,e=r.ik([px,py,0.55],Rd); r.goto(q,40)
        z=0.55; hit=None
        while z>0.0:
            z-=0.01; q,e=r.ik([px,py,z],Rd); err=r.goto(q,5)
            if err>0.012: hit=r.tool()[2,3]; break
        print(f'  probe x{px:.2f} y{py:+.2f} hit z {hit} ikerr {e:.4f} dr {r.obs[103:109].max():.3f}',flush=True)
        q,e=r.ik([px,py,0.50],Rd); r.goto(q,30)
        if r.n>950: print('near step limit'); break
