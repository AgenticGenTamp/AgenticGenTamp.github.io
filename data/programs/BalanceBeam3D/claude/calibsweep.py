import numpy as np, sys
import kinova, ctrl
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
AX=int(sys.argv[1]); SGN=float(sys.argv[2]); Z=float(sys.argv[3]); G=float(sys.argv[4])
env=make_env(); obs,_=env.reset(seed=0)
def drive(bx,by,byaw,n=80):
    global obs
    for i in range(n):
        a=np.zeros(11)
        a[0]=np.clip((bx-obs[16])/ctrl.BASE_GAIN,-0.1,0.1)
        a[1]=np.clip((by-obs[17])/ctrl.BASE_GAIN,-0.1,0.1)
        a[2]=np.clip((byaw-obs[18])/ctrl.YAW_GAIN,-0.1,0.1)
        a[10]=G
        obs,r,te,tr,_=env.step(a.astype(np.float32))
def servo(pt,n,grip,tol=None):
    global obs
    for i in range(n):
        dq=kinova.ik_step(obs[19:26], ctrl.w2a(pt,obs[16:19]), ctrl.Rdown, tool_offset=ctrl.TOOL)
        a=np.zeros(11); a[3:10]=np.clip(dq/ctrl.ARM_GAIN,-0.1,0.1); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if tol is not None and np.linalg.norm(ctrl.ee_world(obs)-pt)<tol and i>3: break
    return ctrl.ee_world(obs)
blk=obs[54:57].copy(); print("blk",blk)
drive(blk[0]-0.45, blk[1], 0.0)
start=blk.copy(); start[2]=0.15; start[AX]-=SGN*0.14
servo(start,250,G,tol=0.004)
start[2]=Z; servo(start,250,G,tol=0.004)
print("at start", ctrl.ee_world(obs), "blk", obs[54:57])
for k in range(29):
    start[AX]+=SGN*0.01
    ee=servo(start.copy(),40,G,tol=0.002)
    b=obs[54:57]
    print(f"  cmd{AX}={start[AX]:.3f} ee={ee} blk={b}")
    if np.linalg.norm(b[:2]-blk[:2])>0.008: 
        print("  MOVED at ee",ee); break
env.close()
