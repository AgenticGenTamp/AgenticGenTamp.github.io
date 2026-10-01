import numpy as np
from env_client import make_env
env=make_env()
obs,_=env.reset(seed=42)
def step(a):
    global obs
    obs,r,term,trunc,info=env.step(np.array(a,dtype=np.float32)); return term
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def R(t): c,s=np.cos(t),np.sin(t); return np.array([[c,-s],[s,c]])
def goto(tx,ty,tth,tarm,vac,maxn=400,sc=1.0):
    n=0; stuck=0
    while n<maxn:
        dx=np.clip(tx-obs[0],-0.05,0.05); dy=np.clip(ty-obs[1],-0.05,0.05)
        dth=np.clip(wrap(tth-obs[2]),-0.196,0.196); da=np.clip(tarm-obs[4],-0.1,0.1)
        if max(abs(dx),abs(dy),abs(dth),abs(da))<1e-4: break
        p=obs[[0,1,2,4]].copy(); step([dx*sc,dy*sc,dth*sc,da,vac]); n+=1
        if np.abs(obs[[0,1,2,4]]-p).max()<1e-7:
            stuck+=1
            if stuck>3: return -n
        else: stuck=0
    return n
th=obs[11]; corner=obs[9:11].copy()
a=np.array([-np.cos(th),-np.sin(th)]); nh=np.array([-np.sin(th),np.cos(th)])
d=1.0; P=corner+a*d
s=np.sign(np.dot(obs[:2]-P,nh)); n=nh*s
rp=P+n*0.30; face=np.arctan2(-n[1],-n[0])
goto(obs[0],obs[1],face,0.1,0.0); goto(rp[0],rp[1],face,0.1,0.0); goto(rp[0],rp[1],face,0.2,0.0)
for i in range(20):
    p=obs[:2].copy(); step([-n[0]*0.01,-n[1]*0.01,0,0,0])
    if np.linalg.norm(obs[:2]-p)<1e-7: break
step([0,0,0,0,1.0])
thr=obs[2]; rel=R(-thr)@(obs[9:11]-obs[:2]); dth_rel=wrap(obs[11]-thr)
print("rel",rel,"dth_rel",dth_rel)
M=obs[20:22].copy(); T=obs[29:31].copy()
v=(T-M)/np.linalg.norm(T-M); print("v",v,"dist",np.linalg.norm(T-M))
t=np.arctan2(v[1],v[0])+3*np.pi/4
C=M-0.106*v
thr_f=wrap(t-dth_rel)
rpos=C-R(thr_f)@rel
print("target hook C",C,"t",t,"robot",rpos,thr_f)
start=rpos-v*0.35
print("start",start)
# stage: rotate then move to start then advance
print("rot",goto(obs[0],obs[1],thr_f,0.2,1.0))
print("obs after rot",obs[:3],obs[9:12])
print("mv",goto(start[0],start[1],thr_f,0.2,1.0))
print("at start",obs[:3],"hook",obs[9:12],"mb",obs[20:22])
np.save("ps.npy",obs)
for i in range(40):
    p=obs.copy(); step([v[0]*0.01,v[1]*0.01,0,0,1.0])
    if np.abs(obs[:2]-p[:2]).max()<1e-7: print("blocked i",i); break
    if np.abs(obs[20:22]-p[20:22]).max()>1e-7: print("BUTTON MOVED i",i,obs[20:22])
    print(i, obs[:2], obs[20:22], np.linalg.norm(obs[20:22]-T))
np.save("pe.npy",obs)
env.close()
