from probe_motion_lib import *
import numpy as np
import probe_motion_lib as L
env, obs, info = new_env(0)
log=[]
orig=L.step
def step2(env,dx=0.,dy=0.,dth=0.,i=0):
    o=orig(env,dx,dy,dth,i); log.append((len(log),i,dx,dy,pose(o,0).copy(),pose(o,1).copy())); return o
L.step=step2
obs=L.setth(env,obs,0.0,0); obs=L.setth(env,obs,0.0,1)
for y in [2.2,2.0,1.5,1.0,0.5,0.0,-0.5]:
    obs,ok=L.goto(env,obs,1.2,y,0)
    r0=None
    if ok: obs,p=L.push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00005); r0=p[0]
    obs,ok1=L.goto(env,obs,-1.2,y,1)
    r1=None
    if ok1: obs,p=L.push_dir(env,obs,1,0,1); r1=p[0]
    print("y=%5.2f r0res=%s (r0 at %s) r1res=%s (r1 at %s)"%(y,r0,np.round(pose(obs,0),3),r1,np.round(pose(obs,1),3)))
arr=[(k,i,np.round(p1,3)) for (k,i,dx,dy,p0,p1) in log]
x1=np.array([p1[0] for (k,i,dx,dy,p0,p1) in log])
idx=np.where(np.sign(x1[:-1])*np.sign(x1[1:])<0)[0]
print("crossings idx",idx[:5])
for k in idx[:3]:
    for j in range(max(0,k-2),min(len(log),k+3)):
        kk,i,dx,dy,p0,p1=log[j]; print("   step",kk,"actor",i,"d",dx,dy,"r1",np.round(p1,4))
env.close()
