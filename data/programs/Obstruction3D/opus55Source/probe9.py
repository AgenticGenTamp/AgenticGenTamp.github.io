import numpy as np, pybullet as p
from approach import Planner, _Info, _yaw_quat
pl = Planner(1, 1e18)
s,_ = pl.sim.reset(seed=0)
arm = pl.arm; cid = arm.physics_client_id
q = pl.ik([0.3, 0.0, 0.2], _yaw_quat(0.0), _Info(s).joints(), 1e-5)
arm.set_joints(list(q) + [0.0]*6)
ee = np.array(arm.get_end_effector_pose().position)
print('ee', ee.round(4))
for li in range(p.getNumJoints(arm.robot_id, physicsClientId=cid)):
    name = p.getJointInfo(arm.robot_id, li, physicsClientId=cid)[12].decode()
    a, b = p.getAABB(arm.robot_id, li, physicsClientId=cid)
    a = np.array(a) - ee; b = np.array(b) - ee
    if a[2] < 0.06:
        print(li, name, 'x', a[0].round(3), b[0].round(3), 'y', a[1].round(3), b[1].round(3), 'z', a[2].round(3), b[2].round(3))
