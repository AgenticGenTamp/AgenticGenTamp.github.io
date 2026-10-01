import numpy as np
import kinova
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
MOUNT = np.array([0.0,0.0,0.45]); TOOL=0.13
Rdown = np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
def w2a(p, base):
    yaw=base[2]; c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return R.T@(p-np.array([base[0],base[1],0.0]))-MOUNT
env=make_env(); obs,_=env.reset(seed=0)
tgt=np.array([0.55,0.0,0.30])
def goto(pt,n,grip=0.0):
    global obs
    for i in range(n):
        dq=kinova.ik_step(obs[19:26], w2a(pt,obs[16:19]), Rdown, tool_offset=TOOL)
        a=np.zeros(11); a[3:10]=np.clip(dq,-0.1,0.1); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
    return kinova.fk(obs[19:26],TOOL)[:3,3]+MOUNT
print("up:",goto(tgt,120))
# descend
for z in np.arange(0.28,-0.15,-0.02):
    p=np.array([0.55,0.0,z])
    ee=goto(p,25)
    print(f"cmd z={z:+.3f} actual_ee_worldz={ee[2]:+.4f} xy={ee[:2]}")
env.close()
