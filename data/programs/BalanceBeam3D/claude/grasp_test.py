import numpy as np, sys, os
import kinova, ctrl
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
ZG=float(sys.argv[1]) if len(sys.argv)>1 else 0.005
env=make_env(); obs,_=env.reset(seed=0)
def drive(bx,by,byaw,n=120,g=0.0):
    global obs
    for i in range(n):
        a=np.zeros(11)
        a[0]=np.clip((bx-obs[16])/ctrl.BASE_GAIN,-0.1,0.1)
        a[1]=np.clip((by-obs[17])/ctrl.BASE_GAIN,-0.1,0.1)
        a[2]=np.clip((byaw-obs[18])/ctrl.YAW_GAIN,-0.1,0.1)
        a[10]=g
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if abs(bx-obs[16])<5e-4 and abs(by-obs[17])<5e-4 and abs(byaw-obs[18])<1e-3: break
def servo(pt,n,grip,tol=None):
    global obs
    for i in range(n):
        dq=kinova.ik_step(obs[19:26], ctrl.w2a(pt,obs[16:19]), ctrl.Rdown, tool_offset=ctrl.TOOL)
        a=np.zeros(11); a[3:10]=np.clip(dq/ctrl.ARM_GAIN,-0.1,0.1); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if tol is not None and np.linalg.norm(ctrl.ee_world(obs)-pt)<tol and i>3: break
    return ctrl.ee_world(obs),i
blk=obs[54:57].copy()
drive(blk[0]-0.45, blk[1], 0.0)
print("hi",servo(np.array([blk[0],blk[1],0.25]),300,0.0,tol=0.004), obs[54:57])
print("lo",servo(np.array([blk[0],blk[1],ZG]),300,0.0,tol=0.003), obs[54:57])
for i in range(30):
    a=np.zeros(11); a[10]=1.0; obs,r,te,tr,_=env.step(a.astype(np.float32))
print("lift",servo(np.array([blk[0],blk[1],0.25]),300,1.0,tol=0.005))
print("BLK",obs[54:57], "grasped" if obs[56]>0.1 else "FAILED")
env.close()
