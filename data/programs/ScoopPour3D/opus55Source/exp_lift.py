import numpy as np, sys
from simutil import *
from kinlib import Kin, down_R
N=int(sys.argv[1]) if len(sys.argv)>1 else 10
env=make_env(N); s,_=env.reset(seed=0,options={'object_count':N})
K=Kin()
def go(bt, pos, R, g, steps, tol=0.01):
    global s
    for t in range(steps):
        base,arm=robot_q(s)
        q,err=K.ik(bt, arm, pos, R)
        a=act(base,arm,s,bt,q,g)
        s,_,term,_,_=env.step(a)
        base,arm=robot_q(s)
        p,_=K.fk(base,arm)
        if np.linalg.norm(p-pos)<tol and np.abs(base-bt).max()<0.01: break
    print('reached',t,np.round(p,3),'ik err',round(err,4), 'bin',np.round(obj(s,'bin_yellow_0')[:3],3))
bt=np.array([-0.15,-0.3,0.0]); R=down_R(np.pi/2)
go(bt,np.array([0.5,-0.34,0.62]),R,0,60)
go(bt,np.array([0.5,-0.34,0.515]),R,0,40,0.005)
go(bt,np.array([0.5,-0.34,0.515]),R,1,5,0.0)
for z in [0.56,0.62,0.68]:
    go(bt,np.array([0.5,-0.34,z]),R,1,30)
    print(np.round(obj(s,'bin_yellow_0'),3))
