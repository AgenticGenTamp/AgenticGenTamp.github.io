import numpy as np, sys
from common import *
from tosslib import Kin
k=Kin(); env=make_env()
seed=int(sys.argv[1]); off=float(sys.argv[2]); R=float(sys.argv[3])
s,_=env.reset(seed=seed, options={'object_count':1})
def yawq(o):
    qw,qx,qy,qz=o[3:7]; return np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
def down_rot(psi):
    x=np.array([np.cos(psi),np.sin(psi),0]); z=np.array([0,0,-1.]); y=np.cross(z,x)
    return np.stack([x,y,z],1)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def step(bt, qt, g):
    global s
    base,q=robot(s); e=qt-q
    a=np.zeros(18); a[3:10]=np.clip(e,-0.1,0.1); a[11:18]=np.clip(e/0.3,-3,3); a[10]=g
    db=bt-base; db[2]=wrap(db[2]); a[0:3]=np.clip(db,-0.1,0.1)
    s,*_=env.step(a)
def go(bt,qt,n,g):
    for i in range(n): step(bt,qt,g)
    base,q=robot(s); return np.abs(qt-q).max(), np.abs(bt-base).max()
c=obj(s,'cube_0'); base,q=robot(s)
th=np.arctan2(c[1]-base[1],c[0]-base[0])
bt=np.array([c[0]-(R+0.12)*np.cos(th), c[1]-(R+0.12)*np.sin(th), th])
print('base go', go(bt,q,15,0))
psi=yawq(c)+off
psi = th + wrap(psi-th)  # nothing
# choose psi equivalent mod pi/2 closest to th
cands=[yawq(c)+off+j*np.pi/2 for j in range(4)]
psi=min(cands,key=lambda p: abs(wrap(p-th)))
qnom=np.array([0,0.6,np.pi,-1.6,0,-1.2,np.pi/2])
qa,err=k.ik(bt,qnom,c[:3]+[0,0,0.15],down_rot(psi),qnom=qnom); print('ik err',err, qa.round(2))
print('track', go(bt,qa,20,0))
qb,err=k.ik(bt,qa,c[:3]+[0,0,0.0],down_rot(psi),qnom=qa); print('ik err',err, qb.round(2))
print('track', go(bt,qb,15,0))
print('pinch', k.fk(*robot(s))[0], 'cube', obj(s,'cube_0')[:3])
go(bt,qb,8,1.0)
go(bt,qa,15,1.0)
print('cube after lift', obj(s,'cube_0')[:3], 'pinch', k.fk(*robot(s))[0])
