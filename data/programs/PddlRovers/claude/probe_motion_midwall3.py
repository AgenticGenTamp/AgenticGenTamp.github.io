from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
obs=setth(env,obs,0.0,0)
ys=[round(v,2) for v in np.arange(-2.25,2.30,0.25)]
for y in ys:
    obs,ok=nav(env,obs,1.6,y,0)
    if not ok: print("y=%5.2f NAVFAIL %s"%(y,np.round(pose(obs,0),2))); continue
    obs,pp=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.0002)
    print("y=%5.2f xmin=%.4f"%(y,pp[0]))
env.close()
