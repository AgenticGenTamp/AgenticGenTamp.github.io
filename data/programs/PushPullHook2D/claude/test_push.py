import numpy as np
from env_client import make_env
env=make_env()
obs,_=env.reset(seed=42)
def step(a):
    global obs
    obs,r,term,trunc,info=env.step(np.array(a,dtype=np.float32)); return term
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def goto(tx,ty,tth,tarm,vac,maxn=400):
    n=0; stuck=0
    while n<maxn:
        dx=np.clip(tx-obs[0],-0.05,0.05); dy=np.clip(ty-obs[1],-0.05,0.05)
        dth=np.clip(wrap(tth-obs[2]),-0.196,0.196); da=np.clip(tarm-obs[4],-0.1,0.1)
        if max(abs(dx),abs(dy),abs(dth),abs(da))<1e-4: break
        p=obs[[0,1,2,4]].copy(); step([dx,dy,dth,da,vac]); n+=1
        if np.abs(obs[[0,1,2,4]]-p).max()<1e-7:
            stuck+=1
            if stuck>3: return -n
        else: stuck=0
    return n
# grasp hook as before
corner=obs[9:11].copy(); th=obs[11]
a=np.array([-np.cos(th),-np.sin(th)])
nrm=np.array([-np.sin(th),np.cos(th)])
d=1.0; P=corner+a*d
side=np.sign(np.dot(obs[:2]-P,nrm)); nrm=nrm*side
rp=P+nrm*0.30; face=np.arctan2(-nrm[1],-nrm[0])
goto(obs[0],obs[1],face,0.1,0.0)
print("mv",goto(rp[0],rp[1],face,0.1,0.0))
print("arm",goto(rp[0],rp[1],face,0.2,0.0))
for i in range(20):
    p=obs[:2].copy(); step([-nrm[0]*0.01,-nrm[1]*0.01,0,0,0])
    if np.linalg.norm(obs[:2]-p)<1e-7: break
step([0,0,0,0,1.0])
print("grasped. hook",obs[9:12],"robot",obs[:3])
grabrel = obs[9:11]-obs[:2]
print("rel corner",grabrel, "hookth-robotth",wrap(obs[11]-obs[2]))
# now move up toward button, see if hook pushes it
mb=obs[20:22].copy(); print("mb",mb)
for i in range(30):
    p=obs.copy(); step([0,0.05,0,0,1.0])
    if np.abs(obs-p).max()<1e-7: print("blocked up at",i,obs[:3]); break
print("robot",obs[:3],"hook",obs[9:12],"mb",obs[20:22])
np.save("p42.npy",obs)
env.close()
