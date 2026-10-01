from drawer_util import *
import sys
y=float(sys.argv[1])
r=R()
bd=r.base.copy(); bd[0]=1.6; bd[1]=y
r.goto(r.q,40,base_d=bd)
def mon(tag):
    print(tag,'n',r.n,'tool',r.tool()[:3,3],'dr',r.obs[103:109],flush=True)
q,e=r.ik([1.1,y,0.30],RA); r.goto(q,80); mon('a')
x=1.1
while x>0.6:
    x-=0.02; q,e=r.ik([x,y,0.30],RA); err=r.goto(q,10)
    if err>0.02: break
mon('b')
q,e=r.ik([1.1,y,0.30],RA); r.goto(q,30); mon('c')
q,e=r.ik([1.1,y,0.44],RA); r.goto(q,40); mon('d')
q,e=r.ik([0.975,y,0.44],RA); r.goto(q,40); mon('e')
z=0.44
while z>0.04:
    z-=0.01; q,e=r.ik([0.975,y,z],RA); err=r.goto(q,8)
    if err>0.02:
        mon('hit'); q,e=r.ik([1.1,y,z],RA); r.goto(q,25); mon('ret'); z-=0.06
        q,e=r.ik([1.1,y,z],RA); r.goto(q,25)
mon('end')
