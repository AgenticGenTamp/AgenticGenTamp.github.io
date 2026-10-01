from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
PX,PY=0.6543,-0.1987
def setth(obs,th,i=0):
    for _ in range(60):
        d=th-pose(obs,i)[2]
        if abs(d)<0.0005: break
        obs,m,dd=try_move(env,obs,0,0,float(np.clip(d,-0.4,0.4)),i)
    return obs
res=[]
for deg in range(0,181,15):
    th=np.deg2rad(deg)
    obs,ok=goto(env,obs,PX+0.5,PY,0)
    obs=setth(obs,th)
    obs,p=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00002)
    c=p[0]-PX-0.05
    res.append((deg,c))
    print("theta=%3d  x-support=%.5f"%(deg,c))
env.close()
