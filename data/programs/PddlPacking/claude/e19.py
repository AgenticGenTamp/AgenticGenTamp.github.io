import numpy as np, ctrl
from env_client import make_env
env=make_env()
cnts={}
for seed in range(40):
    obs,info=env.reset(seed=seed)
    n=info.get("object_count")
    cnts[n]=cnts.get(n,0)+1
    p=obs.get_object_from_name("plate"); t=obs.get_object_from_name("table")
    pp=tuple(round(obs.get(p,f),4) for f in ["pose_x","pose_y","pose_z","half_extent_x","half_extent_y"])
    tt=tuple(round(obs.get(t,f),4) for f in ["pose_x","pose_y","pose_z","half_extent_x","half_extent_y","half_extent_z"])
    if seed<3 or pp!=(0,0,0.7305,0.135,0.135): print(seed,n,pp,tt)
    r=ctrl.rob(obs)
    if seed==0: print("robot",np.round(r,4))
print(cnts)
print("max_steps",env.max_steps)
env.close()
