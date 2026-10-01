import numpy as np, sys, json
import kinova
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.0,0.0,0.45]); TOOL=0.19
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
def w2a(p,base):
    yaw=base[2]; c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])@(p-np.array([base[0],base[1],0.0]))-MOUNT
def ee_world(obs):
    b=obs[16:19]; yaw=b[2]; c,s=np.cos(yaw),np.sin(yaw)
    Rm=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return np.array([b[0],b[1],0.0])+Rm@(MOUNT+kinova.fk(obs[19:26],TOOL)[:3,3])
env=make_env(); obs,_=env.reset(seed=0)
def goto(pt,n,grip=0.0,log=0):
    global obs
    for i in range(n):
        dq=kinova.ik_step(obs[19:26], w2a(pt,obs[16:19]), Rdown, tool_offset=TOOL)
        a=np.zeros(11); a[3:10]=np.clip(dq,-0.1,0.1); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
        if log and i%log==0: print("  ",i, ee_world(obs), obs[54:57])
    return ee_world(obs)
blk=obs[54:57].copy()
print("above:",goto(np.array([blk[0],blk[1],0.20]),150,grip=0.0))
print("down:"); goto(np.array([blk[0],blk[1],-0.02]),200,grip=0.0,log=20)
json.dump(obs.tolist(), open("state_down.json","w"))
env.close()
