import sys
import numpy as np
from env_client import make_env
import kinova_fk as K
np.set_printoptions(precision=4, suppress=True)
MOUNT = np.array([0.,0.,0.44])
FLAG = float(sys.argv[1]); OPEN = 1.0-FLAG
OBJ = int(sys.argv[2]) if len(sys.argv)>2 else 2   # 0=bowl 1=drink 2=can
SL = [0,16,32][OBJ]

def R_down(yaw=0.0):
    c,s=np.cos(yaw),np.sin(yaw)
    Rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return Rz@np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])

env = make_env(); obs,_ = env.reset(seed=0); obs=np.asarray(obs,float)
def step(a):
    global obs
    o,r,te,tr,i=env.step(np.asarray(a,float)); obs=np.asarray(o,float); return r
def w2r(p):
    bx,by,th = obs[93],obs[94],obs[95]
    c,s=np.cos(-th),np.sin(-th); d=np.asarray(p,float)-np.array([bx,by,0.])
    return np.array([c*d[0]-s*d[1], s*d[0]+c*d[1], d[2]])-MOUNT
def r2w(p):
    bx,by,th=obs[93],obs[94],obs[95]; p=np.asarray(p,float)+MOUNT
    c,s=np.cos(th),np.sin(th)
    return np.array([bx+c*p[0]-s*p[1], by+s*p[0]+c*p[1], p[2]])
def ee_world():
    return r2w(K.fk(obs[96:103])[:3,3])

def servo(tgt_w, grip, nsteps, yaw=0.0):
    for t in range(nsteps):
        rel = w2r(tgt_w)
        qt,ok,inf = K.ik(rel, target_rot=R_down(yaw), q_init=obs[96:103], restarts=1, max_iters=120)
        a=np.zeros(11); a[3:10]=np.clip((qt-obs[96:103])*4.0,-0.1,0.1); a[10]=grip
        step(a)
    return ee_world()

obj0 = obs[SL:SL+3].copy()
print("obj", OBJ, obj0, "bbox", obs[SL+13:SL+16], "EE0", ee_world())
a=np.zeros(11); a[10]=OPEN
for _ in range(5): step(a)
ee=servo(obj0+np.array([0,0,0.18]), OPEN, 100); print("above: EE",ee,"err",np.linalg.norm(ee-(obj0+[0,0,0.18])))
found=False
for dz in [0.0,-0.03,0.03,0.06]:
    tgt=obj0+np.array([0,0,dz])
    ee=servo(tgt, OPEN, 55)
    zb=obs[SL+2]; pb=obs[SL:SL+3].copy()
    a=np.zeros(11); a[10]=FLAG
    for _ in range(10): step(a)
    ee2=servo(obj0+np.array([0,0,0.22]), FLAG, 60)
    d=obs[SL:SL+3]-pb
    print("dz=%+.2f EE=%s err=%.4f | obj moved=%s |d|=%.4f"%(dz,ee,np.linalg.norm(ee-tgt),d,np.linalg.norm(d)))
    if np.linalg.norm(d)>0.03:
        found=True
        # base drive confirm
        b0=obs[93]; p0=obs[SL:SL+3].copy()
        a=np.zeros(11); a[0]=-0.1; a[10]=FLAG
        for _ in range(20): step(a)
        print("  BASE dx=%.4f objd=%s"%(obs[93]-b0, obs[SL:SL+3]-p0))
        break
    a=np.zeros(11); a[10]=OPEN
    for _ in range(5): step(a)
    servo(obj0+np.array([0,0,0.18]), OPEN, 30)
print("FLAG=%.1f GRASPED=%s"%(FLAG,found))
env.close()
