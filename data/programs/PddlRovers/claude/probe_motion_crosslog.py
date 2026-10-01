from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
traj=[]
def log(obs):
    traj.append(pose(obs,1).copy())
# replay midwall1 pattern: rover1 goto (-1.2,y) then push east
import probe_motion_lib as L
orig_step = L.step
def step2(env,dx=0.,dy=0.,dth=0.,i=0):
    o = orig_step(env,dx,dy,dth,i); traj.append(pose(o,1).copy()); return o
L.step = step2
for y in [2.2,2.0,1.5,1.0,0.5,0.0]:
    obs,ok=L.goto(env,obs,1.2,y,0)
    obs,ok1=L.goto(env,obs,-1.2,y,1)
    print("y",y,"r1 at",np.round(pose(obs,1),3))
tr=np.array(traj)
# find crossings of x=0
xs=tr[:,0]
idx=np.where(np.sign(xs[:-1])*np.sign(xs[1:])<0)[0]
print("num steps",len(tr),"crossings",idx[:10])
for k in idx[:6]:
    print("  cross at step",k, np.round(tr[max(0,k-2):k+3],3).tolist())
print("min |x| overall", np.abs(xs).min(), "at y", tr[np.abs(xs).argmin(),1])
env.close()
