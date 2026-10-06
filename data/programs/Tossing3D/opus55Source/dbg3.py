import numpy as np, sys
import throw_test as T
from common import *
k=T.k; env=T.env
def tilt_rot(hd, al):
    z=np.array([np.cos(hd)*np.sin(al), np.sin(hd)*np.sin(al), -np.cos(al)])
    x=np.array([-np.sin(hd), np.cos(hd), 0.]); y=np.cross(z,x)
    return np.stack([x,y,z],1)
def rep(tag):
    s=T.S['s']; p,Rm=k.fk(*robot(s)); cc=obj(s,'cube_0'); print(tag, 'cube-pinch', (cc[:3]-p).round(3), 'dist', np.linalg.norm(cc[:3]-p).round(3))
NC=8; LS=0.5
def tgrasp(seed, al, R=0.45, depth=0.0):
    s,_=env.reset(seed=seed, options={'object_count':1}); T.S['s']=s
    c=obj(s,'cube_0'); base,q=robot(s)
    th=np.arctan2(c[1]-base[1],c[0]-base[0])
    cands=[T.yawq(c)+j*np.pi/2 for j in range(4)]
    hd=min(cands,key=lambda p: abs(T.wrap(p-th)))
    bt=np.array([c[0]-(R+0.12)*np.cos(hd), c[1]-(R+0.12)*np.sin(hd), hd])
    T.go(bt,q,12,0)
    Rg=tilt_rot(hd,al)
    qnom=np.array([0,0.6,np.pi,-1.6,0,-1.2,np.pi/2])
    qa,err=k.ik(bt,qnom,c[:3]-0.12*Rg[:,2]+[0,0,0.03],Rg,qnom=qnom); print('ik',err)
    T.go(bt,qa,18,0)
    qb,err=k.ik(bt,qa,c[:3]+depth*Rg[:,2],Rg,qnom=qa); print('ik',err)
    T.slow(bt,qb,0,0.5,extra=6); rep('down')
    T.go(bt,qb,NC,1.0); rep('closed')
    qc,err=k.ik(bt,qb,c[:3]+[0,0,0.25],Rg,qnom=qb)
    T.slow(bt,qc,1.0,LS,extra=3); rep('lift')
    return bt
if __name__=='__main__':
    bt=tgrasp(int(sys.argv[1]), float(sys.argv[2]), depth=float(sys.argv[3]))
    base,q=robot(T.S['s'])
    for i in range(16):
        qt=q.copy(); qt[5]+=0.15*(i+1); T.go(bt,qt,2,1.0)
        s=T.S['s']; p,Rm=k.fk(*robot(s)); cc=obj(s,'cube_0'); print(i,'zaxis',Rm[:,2].round(2),'dist', np.linalg.norm(cc[:3]-p).round(3))
