import numpy as np, sys, time
from simutil import *
from kinlib import Kin, down_R
N=int(sys.argv[1]) if len(sys.argv)>1 else 10
ZC=float(sys.argv[2]) if len(sys.argv)>2 else 0.72
BX=float(sys.argv[3]) if len(sys.argv)>3 else -0.12
env=make_env(N); s,_=env.reset(seed=0,options={'object_count':N})
K=Kin()
nsteps=0
def Rx(a):
    c,s_=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s_],[0,s_,c]])
def go(bt, pos, R, g, steps, tol=0.01):
    global s, nsteps
    for t in range(steps):
        base,arm=robot_q(s)
        q,err=K.ik(bt, arm, pos, R)
        a=act(base,arm,s,bt,q,g)
        s,_,term,_,_=env.step(a); nsteps+=1
        if term: print('TERMINATED', nsteps); return True
        base,arm=robot_q(s)
        p,_=K.fk(base,arm)
        if np.linalg.norm(p-pos)<tol and np.abs(base-bt).max()<0.01: break
    print(nsteps,'reached',t,np.round(p,3),np.round(pos,3),'ik err',round(err,4), 'bin',np.round(obj(s,'bin_yellow_0'),3))
    return False
def incount():
    g=obj(s,'bin_green_0'); c=0
    for i in range(N):
        o=obj(s,'cube_%d'%i)
        if abs(o[0]-g[0])<0.195 and abs(o[1]-g[1])<0.12 and 0.02<o[2]-g[2]<0.065: c+=1
    return c
bt=np.array([BX,-0.3,0.0]); R0=down_R(np.pi/2)
go(bt,np.array([0.5,-0.34,0.62]),R0,0,60)
go(bt,np.array([0.5,-0.34,0.515]),R0,0,40,0.005)
go(bt,np.array([0.5,-0.34,0.515]),R0,1,4,0.0)
off=np.array([0,-0.14,0.515-0.459-0.035])  # grip rel bin center (bin frame)
cen=np.array([0.5,-0.2,ZC])
go(bt,cen+off,R0,1,40)
# translate
for y in np.linspace(-0.2,0.2,5)[1:]:
    cen=np.array([0.5,y,ZC]); bt=np.array([BX,y-0.1,0.0])
    go(bt,cen+off,R0,1,15,0.02)
for phi in np.linspace(0,-np.pi,13)[1:]:
    Rb=Rx(phi); bt=np.array([BX,cen[1]+(Rb@off)[1]-0.1,0])
    if go(bt,cen+Rb@off,Rb@R0,1,15,0.02): break
    print('phi',round(np.degrees(phi)),'in',incount())
for k in range(10):
    if go(bt,cen+Rb@off,Rb@R0,1,5,0.0): break
    print('in',incount())
