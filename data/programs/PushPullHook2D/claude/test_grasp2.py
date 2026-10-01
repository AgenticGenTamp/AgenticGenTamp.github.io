import numpy as np
from env_client import make_env
env=make_env()
obs,_=env.reset(seed=42)
def step(a):
    global obs
    obs,r,term,trunc,info=env.step(np.array(a,dtype=np.float32)); return term
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def goto(tx,ty,tth,tarm,vac,maxn=300):
    n=0; stuck=0
    while n<maxn:
        dx=np.clip(tx-obs[0],-0.05,0.05); dy=np.clip(ty-obs[1],-0.05,0.05)
        dth=np.clip(wrap(tth-obs[2]),-0.196,0.196); da=np.clip(tarm-obs[4],-0.1,0.1)
        if max(abs(dx),abs(dy),abs(dth),abs(da))<1e-4: break
        p=obs[[0,1,2,4]].copy()
        step([dx,dy,dth,da,vac]); n+=1
        if np.abs(obs[[0,1,2,4]]-p).max()<1e-7:
            stuck+=1
            if stuck>3: print("STUCK at",obs[:7]); break
        else: stuck=0
    return n
corner=obs[9:11].copy(); th=obs[11]
u=np.array([-np.cos(th),-np.sin(th)])
nrm=np.array([-np.sin(th),np.cos(th)])
d=1.0; P=corner+u*d
side=np.sign(np.dot(obs[:2]-P,nrm)); nrm=nrm*side
robot_pos=P+nrm*0.30
face_th=np.arctan2(-nrm[1],-nrm[0])
print("P",P,"robot_pos",robot_pos,"face_th",face_th)
goto(obs[0],obs[1],face_th,0.1,0.0)
goto(robot_pos[0],robot_pos[1],face_th,0.1,0.0)
print("after move",obs[:7])
goto(robot_pos[0],robot_pos[1],face_th,0.2,0.0)
print("after arm",obs[:7])
for i in range(20):
    p=obs[:2].copy(); step([-nrm[0]*0.01,-nrm[1]*0.01,0,0,0])
    if np.linalg.norm(obs[:2]-p)<1e-7:
        print("blocked i",i,"pos",obs[:2],"dist to line",np.dot(obs[:2]-P,nrm)); break
print("pos",obs[:7],"hook",obs[9:12])
step([0,0,0,0,1.0])
print("vac on ->",obs[6])
for i in range(4): step([nrm[0]*0.05,nrm[1]*0.05,0,0,1.0])
print("after pull",obs[:3],"hook",obs[9:12])
for i in range(4): step([0,0,0.196,0,1.0])
print("after rot",obs[:3],"hook",obs[9:12])
np.save("g42.npy",obs)
env.close()
