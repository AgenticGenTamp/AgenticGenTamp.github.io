import sys, numpy as np
from env_client import make_env
import kinova_fk as K
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.,0.,0.44]); SL=32
G=float(sys.argv[1])
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
def step(a):
    global obs
    o,r,te,tr,i=env.step(np.asarray(a,float)); obs=np.asarray(o,float)
def w2r(p):
    bx,by,th=obs[93],obs[94],obs[95]; c,s=np.cos(-th),np.sin(-th)
    d=np.asarray(p,float)-np.array([bx,by,0.]); return np.array([c*d[0]-s*d[1],s*d[0]+c*d[1],d[2]])-MOUNT
def r2w(p):
    bx,by,th=obs[93],obs[94],obs[95]; p=np.asarray(p,float)+MOUNT; c,s=np.cos(th),np.sin(th)
    return np.array([bx+c*p[0]-s*p[1],by+s*p[0]+c*p[1],p[2]])
def ee(): return r2w(K.fk(obs[96:103])[:3,3])
def Rd(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])
def servo(tw,n,g,yaw=0.0):
    for _ in range(n):
        qt,ok,inf=K.ik(w2r(tw),target_rot=Rd(yaw),q_init=obs[96:103],restarts=1,max_iters=120)
        a=np.zeros(11); a[3:10]=np.clip((qt-obs[96:103])*4.0,-0.1,0.1); a[10]=g
        step(a)
    return ee()
can0=obs[SL:SL+3].copy()
# go high above, well to -y side of can, at can height; then sweep in +y THROUGH the can
start=can0+np.array([0.0,-0.28,0.10])
servo(start,80,G); servo(can0+np.array([0.0,-0.28,-0.02]),90,G)
p_before=obs[SL:SL+3].copy(); ee_b=ee()
for k in range(1,29):
    servo(can0+np.array([0.0,-0.28+0.02*k,-0.02]),6,G)
p_after=obs[SL:SL+3].copy()
print(f"G={G} ee_before={ee_b} can_before={p_before} can_after={p_after} disp={np.linalg.norm(p_after-p_before):.4f} obs103={obs[103]:.4f}")
env.close()
