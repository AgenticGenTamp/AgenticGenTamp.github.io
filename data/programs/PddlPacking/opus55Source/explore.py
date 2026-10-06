import numpy as np, time
from robocode.environments.pr2_tamp_env import PR2PackedEnv
import robocode.environments.ss_pybullet as ssp
import pybullet as p
env = PR2PackedEnv(num_blocks=4)
obs,_ = env.reset(seed=0)
with env.client():
    from pybullet_tools import utils as U
    r = env.robot
    print('client', env.client_id, 'arm', env.arm_joints, 'base', env.base_joints, 'tool', env.tool_link)
    print('njoints', p.getNumJoints(r, physicsClientId=env.client_id))
    for j in env.arm_joints+list(env.base_joints):
        info = p.getJointInfo(r, j, physicsClientId=env.client_id)
        print(j, info[1], info[2], info[8], info[9])
    print('tool pose', U.get_link_pose(r, env.tool_link))
    # shoulder pose
    print('shoulder', U.get_link_pose(r, env.arm_joints[0]))
    print('base link', U.get_link_pose(r, U.get_link_parent(r, env.arm_joints[0]) if hasattr(U,'get_link_parent') else 0))
    print('movable', U.get_movable_joints(r)[:20], len(U.get_movable_joints(r)))
