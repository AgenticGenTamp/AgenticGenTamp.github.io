import numpy as np, sys, json
from common import *
from tosslib import Kin
k=Kin(); env=make_env([1])
def yawq(o):
    qw,qx,qy,qz=o[3:7]; return np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
def down_rot(psi):
    x=np.array([np.cos(psi),np.sin(psi),0]); z=np.array([0,0,-1.]); y=np.cross(z,x)
    return np.stack([x,y,z],1)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
S={}
def step(bt, qt, g, vel=None, dq=None):
    base,q=robot(S["s"]); e=(qt if qt is not None else q)-q
    a=np.zeros(18); a[3:10]=np.clip(e,-0.1,0.1); a[11:18]=np.clip(e/0.3,-3,3); a[10]=g
    if vel is not None: a[11:18]=vel; a[3:10]=dq
    db=bt-base; db[2]=wrap(db[2]); a[0:3]=np.clip(db,-0.1,0.1)
    S['s'],*_=env.step(a)
def go(bt,qt,n,g):
    for i in range(n): step(bt,qt,g)
def grasp(seed, R=0.45, dz=0.0):
    s,_=env.reset(seed=seed, options={'object_count':1}); S['s']=s
    c=obj(s,'cube_0'); base,q=robot(s)
    th=np.arctan2(c[1]-base[1],c[0]-base[0])
    bt=np.array([c[0]-(R+0.12)*np.cos(th), c[1]-(R+0.12)*np.sin(th), th])
    go(bt,q,12,0)
    cands=[yawq(c)+j*np.pi/2 for j in range(4)]
    psi=min(cands,key=lambda p: abs(wrap(p-th)))
    qnom=np.array([0,0.6,np.pi,-1.6,0,-1.2,np.pi/2])
    qa,err=k.ik(bt,qnom,c[:3]+[0,0,0.15],down_rot(psi),qnom=qnom)
    go(bt,qa,18,0)
    qb,err=k.ik(bt,qa,c[:3]+[0,0,dz],down_rot(psi),qnom=qa)
    go(bt,qb,12,0)
    go(bt,qb,6,1.0)
    go(bt,qa,10,1.0)
    return bt
def slow(bt,qt,g,speed=1.0,extra=8):
    base,q=robot(S['s']); q0=q.copy()
    N=int(np.ceil(np.abs(qt-q0).max()/(speed*0.1)))
    for i in range(1,N+1):
        qi=q0+(qt-q0)*i/N
        base,q=robot(S['s']); e=qi-q
        a=np.zeros(18); a[3:10]=np.clip(e,-0.1,0.1); a[11:18]=(qt-q0)/N/0.1 + e/0.3; a[10]=g
        S['s'],*_=env.step(a)
    go(bt,qt,extra,g)
def throw(bt, qw, w, V, r, n, wait=25):
    slow(bt,qw,1.0,float(sys.argv[7]) if len(sys.argv)>7 else 1.0)
    base,q=robot(S['s']); print('windup err', np.abs(q-qw).max(), 'pinch', k.fk(base,q)[0].round(3))
    base,q=robot(S['s']); p0=k.fk(base,q)[0]; c0=obj(S['s'],'cube_0')
    traj=[]
    w=np.array(w)
    for i in range(n+wait):
        if i<n: step(bt,None,1.0 if i<r else 0.0, vel=V*w, dq=0.1*np.sign(w))
        else:
            base,q=robot(S['s']); step(bt,q,0.0)
        traj.append(obj(S['s'],'cube_0'))
    traj=np.array(traj)
    return c0, traj
if __name__=='__main__':
    seed=int(sys.argv[1]); qw=json.loads(sys.argv[2]); w=json.loads(sys.argv[3]); V=float(sys.argv[4]); r=int(sys.argv[5]); n=int(sys.argv[6])
    bt=grasp(seed)
    c0,traj=throw(bt,np.array(qw),w,V,r,n)
    print('base',bt.round(3),'held cube',c0[:3].round(3))
    for i,t in enumerate(traj[:15]): print(i, t[:3].round(2), t[7:10].round(2))
    print('final', traj[-1][:3].round(3), 'dist from base', np.hypot(*(traj[-1][:2]-bt[:2])).round(3))
