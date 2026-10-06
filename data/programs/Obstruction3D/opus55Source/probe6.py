import numpy as np, pybullet as p
from kinder.envs.kinematic3d.obstruction3d import ObjectCentricObstruction3DEnv
from approach import _yaw_quat
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import pybullet_inverse_kinematics
sim = ObjectCentricObstruction3DEnv(num_obstructions=1, allow_state_access=True)
arm = sim.robot.arm
j = pybullet_inverse_kinematics(arm, Pose((0.3,0.0,0.15), _yaw_quat(0.0)), validate=True)
for fs in [0.0, 0.8]:
    arm.set_joints(list(j[:7]) + [fs]*6)
    lo = np.full(3, 9.); hi = np.full(3,-9.)
    for li in range(8, p.getNumJoints(arm.robot_id, physicsClientId=arm.physics_client_id)):
        a, b = p.getAABB(arm.robot_id, li, physicsClientId=arm.physics_client_id)
        lo = np.minimum(lo, a); hi = np.maximum(hi, b)
    print(fs, 'gripper aabb', np.round(lo,3), np.round(hi,3), 'ee', arm.get_end_effector_pose().position)
print(p.getAABB(sim.end_effector_viz_id, physicsClientId=sim.physics_client_id))
