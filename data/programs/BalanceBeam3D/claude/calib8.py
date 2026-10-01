import numpy as np, sys
import kinova, ctrl
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
DX=float(sys.argv[1]); ZG=float(sys.argv[2])
env=make_env(); obs,_=env.reset(seed=0)
def step(a):
    global obs
    obs,r,te,tr,_=env.step(np.clip(a,-0.1,0.1).astype(np.float32) if False else a.astype(np.float32))
def drive(bx,by,byaw,n=60):
    global obs
    for i in range(n):
        a=np.zeros(11)
        a[0]=np.clip((bx-obs[16])/ctrl.BASE_GAIN,-0.1,0.1)
        a[1]=np.clip((by-obs[17])/ctrl.BASE_GAIN,-0.1,0.1)
        a[2]=np.clip((byaw-obs[18])/ctrl.YAW_GAIN,-0.1,0.1)
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if abs(bx-obs[16])<1e-3 and abs(by-obs[17])<1e-3 and abs(byaw-obs[18])<1e-3: break
def servo(pt,n,grip=0.0,Rd=ctrl.Rdown,tol=None):
    global obs
    for i in range(n):
        dq=kinova.ik_step(obs[19:26], ctrl.w2a(pt,obs[16:19]), Rd, tool_offset=ctrl.TOOL)
        a=np.zeros(11); a[3:10]=np.clip(dq/ctrl.ARM_GAIN,-0.1,0.1); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if tol is not None and np.linalg.norm(ctrl.ee_world(obs)-pt)<tol and i>5: break
    return ctrl.ee_world(obs), i
blk=obs[54:57].copy(); print("blk",blk)
drive(blk[0]-DX, blk[1], 0.0)
print("base",obs[16:19])
print(servo(np.array([blk[0],blk[1],0.15]),200,tol=0.003))
print(servo(np.array([blk[0],blk[1],ZG]),250,tol=0.002), "blk",obs[54:57])
for i in range(30):
    a=np.zeros(11); a[10]=1.0; obs,r,te,tr,_=env.step(a.astype(np.float32))
print(servo(np.array([blk[0],blk[1],0.25]),200,grip=1.0,tol=0.003))
print("after lift blk",obs[54:57], "ee", ctrl.ee_world(obs))
env.close()
