import time, numpy as np
from robocode.environments.pr2_tamp_blocked_env import PR2BlockedEnv
from robocode.environments.ss_pybullet import *
import pybullet as p
t=time.time()
env = PR2BlockedEnv(num_spares=1)
obs,_ = env.reset(seed=0)
print("build", time.time()-t)
with env.client():
    r=env.robot
    print("tool", get_link_pose(r, env.tool_link))
    print("base joints", env.base_joints, "arm", env.arm_joints)
    from pybullet_tools.utils import get_movable_joints, get_joint_names, get_link_name, CLIENT
    mj=get_movable_joints(r); print(len(mj), get_joint_names(r, mj))
    t=time.time()
    for i in range(1000):
        set_joint_positions(r, env.arm_joints, obs[3:10]+0.001*i)
        get_link_pose(r, env.tool_link)
    print("fk ms", (time.time()-t))
    t=time.time()
    for i in range(1000):
        env._in_collision()
    print("coll ms", (time.time()-t))
    for w in env.walls: print("wall", get_pose(w), get_aabb(w))
    print("penned", get_pose(env.penned), "blocker", get_pose(env.blocker))
    print("grip joints", env._gripper_joints, get_joint_positions(r, env._gripper_joints))
