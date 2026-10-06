import numpy as np
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from robocode.environments.ss_pybullet import *
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=5)
b = env.current_backend
with b.client():
    for name, lst in [("walls",b._walls),("mounds",b._mounds),("pillars",b._obstacle_bodies[:2]),("samples",b._samples[:3]),("objs",b._objectives[:2]),("lander",[b._lander])]:
        for h in lst: 
            a=get_aabb(h); print(name, np.round(a[0],3), np.round(a[1],3), np.round(get_point(h),3))
    print("floor?", [k for k in vars(b) if 'floor' in k])
