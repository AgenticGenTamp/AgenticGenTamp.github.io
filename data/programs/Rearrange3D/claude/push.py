import sys
import numpy as np
from env_client import make_env
import kinova_fk as K
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.,0.,0.44])
G=float(sys.argv[1]); ROLL=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
SL=32  # can
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
def step(a):
    global obs
    o,r,te,tr,i=env.step(np.asarray(a,float)); obs=np.asarray(o,float)
def w2r(p):
    bx,by,th=obs[93],obs[94],obs[95]; c,s=np.cos(-th),np.sin(-th)
    d=np.asarray(p,float)-np.array([bx,by,0.])
    return np.array([c*d[0]-s*d[1], s*d[0]+c*d[1], d[2]])-MOUNT
def r2w(p):
    bx,by,th=obs[93],obs[94],obs[95]; p=np.asarray(p,float)+MOUNT; c,s=np.cos(th),np.sin(th)
    return np.array([bx+c*p[0]-s*p[1], by+s*p[0]+c*p[1], p[2]])
def ee(): return r2w(K.fk(obs[96:103])[:3,3])
# tool z -> +x ; roll about tool z
Rb=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
cr,sr=np.cos(ROLL),np.sin(ROLL)
Rz=np.array([[cr,-sr,0],[sr,cr,0],[0,0,1.]])
R=Rb@Rz
def servo(tw,n):
    for _ in range(n):
        qt,ok,inf=K.ik(w2r(tw), target_rot=R, q_init=obs[96:103], restarts=1, max_iters=120)
        a=np.zeros(11); a[3:10]=np.clip((qt-obs[96:103])*4.0,-0.1,0.1); a[10]=G
        step(a)
    return ee()
can=obs[SL:SL+3].copy()
start=can+np.array([-0.28,0,0.0])
e=servo(start,140)
print("start EE",e,"err",np.linalg.norm(e-start),"can",can)
p0=obs[SL:SL+3].copy(); contact=None
for k in range(1,46):
    tw=start+np.array([0.008*k,0,0])
    e=servo(tw,7)
    d=np.linalg.norm(obs[SL:SL+3]-p0)
    if d>0.004 and contact is None:
        contact=(e.copy(), tw.copy(), d)
        print("CONTACT at EE_x=%.4f (cmd %.4f) can_x=%.4f gap=%.4f moved=%.4f"%(e[0],tw[0],can[0],can[0]-e[0],d))
        break
if contact is None:
    print("NO CONTACT; final EE",e,"err",np.linalg.norm(e-tw),"can",obs[SL:SL+3])
print("G=%.1f ROLL=%.2f"%(G,ROLL))
env.close()
