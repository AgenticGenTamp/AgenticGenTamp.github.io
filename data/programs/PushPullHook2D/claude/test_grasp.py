import numpy as np
from env_client import make_env

env=make_env()
obs,_=env.reset(seed=42)
log=[]
def step(a):
    global obs
    obs,r,term,trunc,info=env.step(np.array(a,dtype=np.float32))
    return obs,term,trunc

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def goto(tx,ty,tth,tarm,vac,maxn=200,verbose=False):
    n=0
    while n<maxn:
        dx=np.clip(tx-obs[0],-0.05,0.05)
        dy=np.clip(ty-obs[1],-0.05,0.05)
        dth=np.clip(wrap(tth-obs[2]),-0.196,0.196)
        da=np.clip(tarm-obs[4],-0.1,0.1)
        if abs(dx)<1e-4 and abs(dy)<1e-4 and abs(dth)<1e-4 and abs(da)<1e-4: break
        step([dx,dy,dth,da,vac]); n+=1
        if verbose: print(n,obs[:7])
    return n

corner=obs[9:11].copy(); th=obs[11]
u=np.array([-np.cos(th),-np.sin(th)])
nrm=np.array([-np.sin(th),np.cos(th)])   # one perpendicular
d=1.0
P=corner+u*d
app=-nrm  # approach from this side => robot at P + nrm*dist? choose
robot_pos=P+nrm*0.24
face_th=np.arctan2(-nrm[1],-nrm[0])
print("corner",corner,"th",th,"P",P,"robot_pos",robot_pos,"face_th",face_th)
# first rotate
goto(obs[0],obs[1],face_th,0.1,0.0)
print("after rot",obs[:7])
goto(robot_pos[0],robot_pos[1],face_th,0.1,0.0)
print("after move",obs[:7], "hook",obs[9:12])
goto(robot_pos[0],robot_pos[1],face_th,0.2,0.0)
print("after arm",obs[:7],"hook",obs[9:12])
# push in along -nrm until blocked
for i in range(10):
    prev=obs[:2].copy()
    step([-nrm[0]*0.01,-nrm[1]*0.01,0,0,0])
    if np.linalg.norm(obs[:2]-prev)<1e-6:
        print("blocked at",obs[:2],"i",i); break
print("pos",obs[:7],"hook",obs[9:12])
# vacuum on
step([0,0,0,0,1.0])
print("vac",obs[:7],"hook",obs[9:12])
# move back
for i in range(5): step([nrm[0]*0.05,nrm[1]*0.05,0,0,1.0])
print("after pull",obs[:7],"hook",obs[9:12])
np.save("grasped42.npy",obs)
# rotate
for i in range(5): step([0,0,0.196,0,1.0])
print("after rot",obs[:7],"hook",obs[9:12])
np.save("grasped42b.npy",obs)
env.close()
