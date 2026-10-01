import numpy as np
from probe_vis_lib import *
env,obs,info=new_env(0)
def pushx(obs,i,sign):
    s=0.2
    while s>=0.0005:
        p0=pose(obs,i)[:2].copy()
        obs,_,_,_,_=st(env,dx=sign*s,i=i)
        if np.linalg.norm(pose(obs,i)[:2]-p0)<1e-7: s/=2
    return obs,pose(obs,i)[0]
ys=[-2.4,-2.3,-2.2,-2.1,-2.0,-1.8,-1.0,0.0,1.0,2.0,2.4]
for yi,y in enumerate(ys):
    i=yi%2
    # move along x back to safe side first
    obs,_=goto(env,obs,0.8 if i==0 else -0.8, pose(obs,i)[1],i=i,tol=0.01)
    obs,ok=goto(env,obs,0.8 if i==0 else -0.8, y,i=i,tol=0.01)
    obs,xs=pushx(obs,i,-1 if i==0 else +1)
    print("y=%+.2f rover%d stop x=%+.4f (reach_ok=%s ypos=%.3f)"%(y,i,xs,ok,pose(obs,i)[1]))
env.close()
