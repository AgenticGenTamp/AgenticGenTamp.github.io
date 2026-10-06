import numpy as np, pybullet as p
from pybullet_helpers.robots import create_pybullet_mobile_robot
from pybullet_helpers.geometry import SE2Pose, Pose
from pybullet_helpers.inverse_kinematics import pybullet_inverse_kinematics
from scipy.spatial.transform import Rotation as R
cid = p.connect(p.DIRECT)
robot = create_pybullet_mobile_robot("tidybot-kinova", cid, base_z=-0.4, base_home_pose=SE2Pose(-0.12, 0, 0))
arm = robot.arm
print(type(arm), arm.arm_joints, arm.end_effector_id)
q = R.from_matrix([[0,1,0],[1,0,0],[0,0,-1]]).as_quat()
j = pybullet_inverse_kinematics(arm, Pose((0.3,0.0,0.15), tuple(q)), validate=True)
print(np.round(j,3))
for fs in [0.0, 0.4, 0.8]:
    arm.set_joints(list(j[:7]) + [fs]*6)
    print('ee', arm.get_end_effector_pose())
    for li in range(p.getNumJoints(arm.robot_id, physicsClientId=cid)):
        info = p.getJointInfo(arm.robot_id, li, physicsClientId=cid)
        name = info[12].decode()
        if 'finger' in name or 'pad' in name or 'knuckle' in name:
            ls = p.getLinkState(arm.robot_id, li, physicsClientId=cid)
            print(fs, name, np.round(ls[0],3))
import inspect
print(inspect.signature(arm.set_joints))
