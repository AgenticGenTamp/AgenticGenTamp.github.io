import sys
import numpy as np
from env_client import make_env
import kinova_fk as K
np.set_printoptions(precision=4, suppress=True)

FLAG = float(sys.argv[1])
OPEN = 1.0-FLAG
env = make_env()
obs,_ = env.reset(seed=0)
obs = np.asarray(obs,float)
objA0 = obs[0:3].copy()

def step(a):
    global obs
    o,r,te,tr,i = env.step(np.clip(np.asarray(a,float),-0.1,None))
    obs = np.asarray(o,float); return r

# tool pointing DOWN: tool z axis = -world z
R_down = np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])

def servo(target, grip, nsteps, rot=R_down):
    for t in range(nsteps):
        q = obs[96:103]
        qt, ok, inf = K.ik(target, target_rot=rot, q_init=q, base_pose=obs[93:96], restarts=1, max_iters=120)
        dq = qt - q
        a = np.zeros(11); a[3:10] = np.clip(dq*4.0,-0.1,0.1); a[10]=grip
        step(a)
    T = K.fk(obs[96:103], obs[93:96])
    return T[:3,3], np.linalg.norm(T[:3,3]-target)

res=[]
for dz in [0.0, 0.03, -0.02]:
    tgt = objA0 + np.array([0,0,dz])
    # open first
    a=np.zeros(11); a[10]=OPEN
    for _ in range(5): step(a)
    ee,err = servo(tgt, OPEN, 90)
    z_before = obs[2]
    a=np.zeros(11); a[10]=FLAG
    for _ in range(15): step(a)
    ee2,_ = servo(tgt+np.array([0,0,0.20]), FLAG, 70)
    dzobj = obs[2]-z_before
    grabbed = dzobj > 0.03
    print("dz=%+.2f EEerr=%.4f EE=%s | objA dz after lift=%+.4f objA=%s GRABBED=%s"%(dz,err,ee,dzobj,obs[0:3],grabbed))
    res.append(grabbed)
    if grabbed: break
    # release and reset arm-ish
    a=np.zeros(11); a[10]=OPEN
    for _ in range(5): step(a)
print("FLAG=%.1f ANYGRAB=%s"%(FLAG, any(res)))
env.close()
