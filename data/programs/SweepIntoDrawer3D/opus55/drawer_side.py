from drawer_util import *
import sys
y=float(sys.argv[1]); z=float(sys.argv[2]); xin=float(sys.argv[3])
r=R(); r.grip=1.0
bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
q,e=r.ik([1.1,y,0.30],RA); r.goto(q,80)
q,e=r.ik([1.1,y,z],RA); r.goto(q,30)
x=1.1
while x>xin:
    x-=0.02; q,e=r.ik([x,y,z],RA); err=r.goto(q,8)
    if err>0.02: break
t0=r.tool()[:3,3].copy()
# move toward island (in y) by 0.03 to get behind the front edge
ys=np.sign(-y)
q,e=r.ik(t0+np.array([0,ys*0.04,0]),RA); r.goto(q,20); t1=r.tool()[:3,3].copy()
for k in range(20):
    bd=r.base.copy(); bd[0]+=0.03; bd[2]=np.pi; r.goto(q,4,base_d=bd)
print(f'y{y:+.2f} z{z} in {t0} shifted {t1} dr {r.obs[103:109]} tool_end {r.tool()[:3,3]}',flush=True)
