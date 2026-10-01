import numpy as np, sys, json
import kinova
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.0,0.0,0.45]); TOOL=0.20
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
def w2a(p,base):
    yaw=base[2]; c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])@(p-np.array([base[0],base[1],0.0]))-MOUNT
def ee_world(obs):
    b=obs[16:19]; yaw=b[2]; c,s=np.cos(yaw),np.sin(yaw)
    Rm=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return np.array([b[0],b[1],0.0])+Rm@(MOUNT+kinova.fk(obs[19:26],TOOL)[:3,3])
GRIP=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
ZG=float(sys.argv[2]) if len(sys.argv)>2 else 0.005
env=make_env(); obs,_=env.reset(seed=0)
def servo(pt,n,grip=0.0,Rd=Rdown,maxstep=0.05):
    global obs
    for i in range(n):
        dq=kinova.ik_step(obs[19:26], w2a(pt,obs[16:19]), Rd, tool_offset=TOOL)
        a=np.zeros(11); a[3:10]=np.clip(dq,-maxstep,maxstep); a[10]=grip
        obs,r,te,tr,_=env.step(a.astype(np.float32))
    return ee_world(obs)
blk=obs[54:57].copy(); print("blk",blk)
servo(np.array([blk[0],blk[1],0.20]),150,grip=GRIP if False else 0.0)
# open gripper explicitly first
print("pre",ee_world(obs))
for z in [0.15,0.10,0.06,0.03,ZG]:
    servo(np.array([blk[0],blk[1],z]),40,grip=0.0)
print("at grasp",ee_world(obs),"blk",obs[54:57])
for i in range(40):
    a=np.zeros(11); a[10]=GRIP; obs,r,te,tr,_=env.step(a.astype(np.float32))
print("after close grip=",obs[26],"blk",obs[54:57])
print("lift",servo(np.array([blk[0],blk[1],0.25]),150,grip=GRIP))
print("blk",obs[54:57])
json.dump(obs.tolist(),open("state_lift.json","w"))
env.close()
