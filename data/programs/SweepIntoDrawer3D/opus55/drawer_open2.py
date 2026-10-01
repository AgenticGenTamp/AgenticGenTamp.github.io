from drawer_util import *
import sys
y=float(sys.argv[1]); pull=float(sys.argv[2]); pts=[tuple(map(float,s.split(','))) for s in sys.argv[3:]]
Rd=R_down(np.pi/2)
r=R(); r.grip=1.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
q,e=r.ik([1.1,y,0.50],Rd); r.goto(q,60)
q,e=r.ik([0.917,y,0.50],Rd); r.goto(q,30)
z=0.50
while z>0.31:
    z-=0.01; q,e=r.ik([0.917,y,z],Rd); r.goto(q,5)
r.goto(q,10)
log=[]
for k in range(int(pull/0.03)):
    bd=r.base.copy(); bd[0]+=0.03; bd[2]=np.pi; r.goto(q,4,base_d=bd); log.append(round(float(r.obs[103:109].max()),3))
tl=r.tool()[:3,3].copy()
print('pull log',log,'tool',tl)
for dz in np.arange(0.02,0.25,0.02):
    q2,e=r.ik(tl+np.array([0,0,dz]),Rd); r.goto(q2,5)
bd=np.array([1.75,y,np.pi]); r.goto(r.q,30,base_d=bd)
print(f'y{y:+.2f} opened dr {r.obs[103:109]} base {r.base} n {r.n}',flush=True)
for (px,py) in pts:
    py=y+py
    q,e=r.ik([px,py,0.52],Rd); r.goto(q,35)
    z=0.52; hit=None
    while z>-0.05:
        z-=0.01; q,e=r.ik([px,py,z],Rd); err=r.goto(q,4)
        if err>0.012: hit=r.tool()[2,3]; break
    print(f'  probe x{px:.2f} y{py:+.2f} hit z {hit} ikerr {e:.4f} dr {r.obs[103:109].max():.3f} n {r.n}',flush=True)
    q,e=r.ik([px,py,0.52],Rd); r.goto(q,25)
