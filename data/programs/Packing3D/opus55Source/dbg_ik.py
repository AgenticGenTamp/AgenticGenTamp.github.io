import numpy as np, pybullet as p
from planner import TwinPlanner, _ik_meta, yaw_quat, quat_mul
from pybullet_helpers.geometry import Pose
tp = TwinPlanner(2); arm = tp.arm
free, arm_idx, lo, hi, is_arm = _ik_meta(arm)
print(free, arm_idx, lo, hi, is_arm, arm.end_effector_id)
pose = Pose((0.4,0.1,0.25), tp.down)
cid=arm.physics_client_id
vals = p.calculateInverseKinematics(arm.robot_id, arm.end_effector_id, pose.position, targetOrientation=pose.orientation, maxNumIterations=100, residualThreshold=1e-6, physicsClientId=cid)
for j,v in zip(free, vals): p.resetJointState(arm.robot_id,j,v,physicsClientId=cid)
ls=p.getLinkState(arm.robot_id, arm.end_effector_id, computeForwardKinematics=True, physicsClientId=cid)
print(np.round(vals,3)); print(ls[4], ls[5], ls[0], ls[1]); print(arm.get_end_effector_pose())
