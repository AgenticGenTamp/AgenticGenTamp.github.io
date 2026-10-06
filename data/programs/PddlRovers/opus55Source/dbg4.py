import numpy as np, math
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from robocode.environments.ss_pybullet import *
import pybullet as p
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=3)
b = env.current_backend
with b.client():
    rover=b._rovers[1]; joints=b._base_joints[1]
    for th in [0, math.pi/8, math.pi/4, math.pi/2]:
        set_joint_positions(rover, joints, (0-1.0,0,th))
        a=get_aabb(rover); print(th, np.round(a[0],3), np.round(a[1],3))
    set_joint_positions(rover, joints, (-1.0,0,0))
    for l in range(-1, p.getNumJoints(rover, physicsClientId=get_client() if 'get_client' in dir() else 0) if False else 40):
        try:
            aa=p.getAABB(rover, l, physicsClientId=b._client if hasattr(b,'_client') else 0)
        except Exception as e: break
        if aa[1][2]<0.03 and False: continue
        print(l, np.round(aa[0],3), np.round(aa[1],3))
