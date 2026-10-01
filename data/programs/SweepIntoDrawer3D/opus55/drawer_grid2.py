from drawer_util import *
import sys
ys=[float(v) for v in sys.argv[1:]]
zs=np.arange(0.10,0.345,0.02)
def fresh(y):
    r=R(); r.grip=1.0
    bd=r.base.copy(); bd[0]=1.6; bd[1]=y
    r.goto(r.q,40,base_d=bd)
    q,e=r.ik([1.05,y,0.3],RA); r.goto(q,80)
    return r
for y in ys:
    r=fresh(y); out=[]
    for z in zs:
        if r.n>850: r=fresh(y)
        q,e=r.ik([1.05,y,z],RA); r.goto(q,40)
        x=1.05; hit=None
        while x>0.90:
            x-=0.01
            q,e=r.ik([x,y,z],RA); err=r.goto(q,6)
            if err>0.015: hit=r.tool()[0,3]; break
        out.append(hit if hit else 0.0)
        q,e=r.ik([1.05,y,z],RA); r.goto(q,20)
    print('y %+.2f '%y+' '.join('%.3f'%h for h in out)+' dr '+str(r.obs[103:109]),flush=True)
