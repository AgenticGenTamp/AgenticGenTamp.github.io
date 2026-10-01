from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
def setth(obs,th,i=0):
    for _ in range(60):
        d=th-pose(obs,i)[2]
        if abs(d)<0.0005: break
        obs,m,dd=try_move(env,obs,0,0,float(np.clip(d,-0.4,0.4)),i)
    return obs
obs=setth(obs,0.0,1)
# move rover0 far east out of the way
obs,ok=goto(env,obs,2.2,-2.2,0); print("r0 parked",ok,np.round(pose(obs,0),3))
for y in [-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.2]:
    obs,ok=goto(env,obs,-1.2,y,1)
    p0=pose(obs,1)
    if abs(p0[0]+1.2)>0.02 or abs(p0[1]-y)>0.02:
        print("y=%5.2f  goto FAILED at %s"%(y,np.round(p0,3))); continue
    obs,p=push_dir(env,obs,1,0,1,coarse=0.1,fine=0.00005)
    print("y=%5.2f  west->east blocked at x=%.5f (ypos %.3f)"%(y,p[0],p[1]))
    # return west
    obs,ok=goto(env,obs,-1.2,y,1)
env.close()
