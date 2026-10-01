from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
def setth(obs,th,i=0):
    for _ in range(60):
        d=th-pose(obs,i)[2]
        if abs(d)<0.0005: break
        obs,m,dd=try_move(env,obs,0,0,float(np.clip(d,-0.4,0.4)),i)
    return obs
obs=setth(obs,0.0,0); obs=setth(obs,0.0,1)
print("r0",np.round(pose(obs,0),4),"r1",np.round(pose(obs,1),4))
for y in [2.2,2.0,1.5,1.0,0.5,0.0,-0.5,-1.0,-1.2,-1.4,-1.5,-1.6,-1.8,-2.0,-2.2]:
    # rover0 from east side
    obs,ok=goto(env,obs,1.2,y,0)
    r0=None
    if ok:
        obs,p=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00005); r0=p[0]
    # rover1 from west side
    obs,ok1=goto(env,obs,-1.2,y,1)
    r1=None
    if ok1:
        obs,p=push_dir(env,obs,1,0,0,coarse=0.1,fine=0.00005); r1=p[0]
    print("y=%5.2f  from_east_xmin=%s  from_west_xmax=%s"%(y, "%.5f"%r0 if r0 is not None else "NA", "%.5f"%r1 if r1 is not None else "NA"))
env.close()
