import numpy as np, sys
import kinova, ctrl
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
GD=float(sys.argv[1])   # gripper during descent
GC=float(sys.argv[2])   # gripper at close
ZG=float(sys.argv[3])
env=make_env(); obs,_=env.reset(seed=0)
def drive(bx,by,byaw,n=80):
    global obs
    for i in range(n):
        a=np.zeros(11)
        a[0]=np.clip((bx-obs[16])/ctrl.BASE_GAIN,-0.1,0.1)
        a[1]=np.clip((by-obs[17])/ctrl.BASE_GAIN,-0.1,0.1)
        a[2]=np.clip((byaw-obs[18])/ctrl.YAW_GAIN,-0.1,0.1)
        a[10]=GD
        obs,r,te,tr,_=env.step(a.astype(np.float32))
def servo(pt,n,grip=0.0,Rd=ctrl.Rdown,tol=None):
    global obs
    for i in range(n):
        dq=kinova.ik_step(obs[19:26], ctrl.w2a(pt,obs[16:19]), Rd, tool_offset=ctrl.TOOL)
        a=np.zeros(11); a[3:10]=np.clip(dq/ctrl.ARM_GAIN,-0.1,0.1); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if tol is not None and np.linalg.norm(ctrl.ee_world(obs)-pt)<tol and i>5: break
    return ctrl.ee_world(obs), i
blk=obs[54:57].copy()
drive(blk[0]-0.45, blk[1], 0.0)
servo(np.array([blk[0],blk[1],0.15]),200,grip=GD,tol=0.003)
print("descend:",servo(np.array([blk[0],blk[1],ZG]),250,grip=GD,tol=0.002), "blk",obs[54:57])
for i in range(40):
    a=np.zeros(11); a[10]=GC; obs,r,te,tr,_=env.step(a.astype(np.float32))
print("close blk",obs[54:57])
print("lift:",servo(np.array([blk[0],blk[1],0.25]),200,grip=GC,tol=0.005))
print("blk",obs[54:57])
env.close()
