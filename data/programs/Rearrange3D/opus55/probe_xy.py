from probe_lib import *
import sys
pr=P(int(sys.argv[2]) if len(sys.argv)>2 else 0); pr.grip=1.0
psi=float(sys.argv[1])
obj=32
zf=0.49+0.0275
res={}
for d in [(-1,0),(1,0),(0,-1),(0,1)]:
    c=pr.obs[obj:obj+3].copy()
    d=np.array([d[0],d[1],0.])
    s=c+0.12*d; s[2]=zf
    pr.goto(pr.ik(s+[0,0,0.15],psi)); pr.goto(pr.ik(s,psi))
    p,why,qe=pr.line(s,c.copy()*[1,1,0]+[0,0,zf],psi,ds=0.001,watch=obj,thr=0.0015,qthr=0.02)
    f=pr.fk()
    dist=np.dot(f-c,d)
    print(d[:2],why,'fk-c',(f-c)[:2],'dist along d',dist)
    res[tuple(d[:2])]=dist
    pr.goto(pr.ik(s,psi)); pr.goto(pr.ik(s+[0,0,0.15],psi))
print('x offset (center err) =',(res[(1,0)]-res[(-1,0)])/2*-1,' half-width x',(res[(1,0)]+res[(-1,0)])/2)
print('y offset =',(res[(0,1)]-res[(0,-1)])/2*-1,' half-width y',(res[(0,1)]+res[(0,-1)])/2)
