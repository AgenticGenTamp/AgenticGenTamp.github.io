from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
obs,ok=goto(env,obs,-1.2,0.0,1); print("start",np.round(pose(obs,1),4))
xs=[pose(obs,1)[0]]
s=0.2; it=0
while s>=0.001 and it<400:
    obs,m,d=try_move(env,obs,s,0,0,1); it+=1
    if not m: s/=2.0
    else: xs.append(pose(obs,1)[0])
print("final",np.round(pose(obs,1),5))
print("path x:", np.round(np.array(xs),4).tolist())
env.close()
