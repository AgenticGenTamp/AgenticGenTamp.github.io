import numpy as np
import kinova
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.0,0.0,0.45]); TOOL=0.188
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
def w2a(p,base):
    yaw=base[2]; c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return R.T@(p-np.array([base[0],base[1],0.0]))-MOUNT
env=make_env(); obs,_=env.reset(seed=0)
I=np.zeros(3)
def goto(pt,n,grip=0.0,ki=0.0):
    global obs,I
    for i in range(n):
        T=kinova.fk(obs[19:26],TOOL)
        pa=w2a(pt,obs[16:19])
        I+= ki*(pa-T[:3,3])
        dq=kinova.ik_step(obs[19:26], pa+I, Rdown, tool_offset=TOOL, pos_w=1.0)
        a=np.zeros(11); a[3:10]=np.clip(dq,-0.1,0.1); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
    return kinova.fk(obs[19:26],TOOL)[:3,3]+MOUNT
print("no I:",goto(np.array([0.55,0.0,0.30]),120))
print("with I:",goto(np.array([0.55,0.0,0.30]),150,ki=0.05))
print("move:",goto(np.array([0.45,0.15,0.25]),150,ki=0.05))
print("joints",obs[19:26])
env.close()
