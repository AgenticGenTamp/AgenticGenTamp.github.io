import numpy as np
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from robocode.environments.ss_pybullet import *
import pybullet as p
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=3)
b = env.current_backend
with b.client():
    rover=b._rovers[1]; joints=b._base_joints[1]
    for tgt in [(-1.35,-1.35,-0.8),(-1.35,-1.35,0),(-1.3,-1.35,0)]:
        set_joint_positions(rover, joints, tgt)
        hits=[bd for bd in b._fixed+[b._rovers[0]] if pairwise_collision(rover, bd)]
        print(tgt, hits, [get_aabb(h) for h in hits])
    print("lander", b._lander, get_aabb(b._lander))
