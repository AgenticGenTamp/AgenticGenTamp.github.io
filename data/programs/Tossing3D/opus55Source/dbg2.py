import numpy as np, sys
import throw_test as T
from common import *
k=T.k; env=T.env
s,_=env.reset(seed=0, options={'object_count':1}); T.S['s']=s
c=obj(s,'cube_0'); base,q=robot(s)
th=np.arctan2(c[1]-base[1],c[0]-base[0]); R=0.45
bt=np.array([c[0]-(R+0.12)*np.cos(th), c[1]-(R+0.12)*np.sin(th), th])
T.go(bt,q,12,0)
cands=[T.yawq(c)+j*np.pi/2 for j in range(4)]
psi=min(cands,key=lambda p: abs(T.wrap(p-th)))
qnom=np.array([0,0.6,np.pi,-1.6,0,-1.2,np.pi/2])
qa,err=k.ik(bt,qnom,c[:3]+[0,0,0.15],T.down_rot(psi),qnom=qnom)
T.go(bt,qa,18,0)
qb,err=k.ik(bt,qa,c[:3]+[0,0,float(sys.argv[1])],T.down_rot(psi),qnom=qa)
def rep(tag):
    s=T.S['s']; p=k.fk(*robot(s))[0]; cc=obj(s,'cube_0'); print(tag, 'cube-pinch', (cc[:3]-p).round(3))
T.go(bt,qb,12,0); rep('down')
for i in range(8): T.go(bt,qb,1,1.0); rep('close%d'%i)
for i in range(6): T.slow(bt,qa,1.0,0.3,extra=0) if False else T.go(bt,qb+(qa-qb)*(i+1)/6,2,1.0); rep('lift%d'%i)
base,q=robot(T.S['s'])
for j,sg in [(5,1)]:
  for i in range(16):
    qt=q.copy(); qt[j]+=sg*0.1*(i+1); T.go(bt,qt,2,1.0)
    s=T.S['s']; p,Rm=k.fk(*robot(s)); cc=obj(s,'cube_0'); print(i,'zaxis',Rm[:,2].round(2),'dist', np.linalg.norm(cc[:3]-p).round(3))
