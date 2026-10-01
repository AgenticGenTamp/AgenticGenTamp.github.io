from drawer_util import *
import sys
y=float(sys.argv[1]); zs=[float(v) for v in sys.argv[2:]]
Rd=RB
r=None; out=[]
for z in zs:
    if r is None or r.n>850:
        r=R(); r.grip=1.0; bd=r.base.copy(); bd[0]=1.6; bd[1]=y; r.goto(r.q,40,base_d=bd)
        q,e=r.ik([1.1,y,0.3],Rd); r.goto(q,90)
    q,e=r.ik([1.03,y,z],Rd); r.goto(q,30)
    x=1.03; hit=None
    while x>0.85:
        x-=0.005; q,e=r.ik([x,y,z],Rd); err=r.goto(q,4)
        if err>0.01: hit=r.tool()[0,3]; break
    out.append('%.2f:%.3f'%(z,hit if hit else 0))
    q,e=r.ik([1.03,y,z],Rd); r.goto(q,20)
print('y %+.2f '%y+' '.join(out),'dr',r.obs[103:109],flush=True)
