import time, numpy as np, pybullet as p
from planner import TwinPlanner, _ik_meta
from pybullet_helpers.geometry import Pose
tp = TwinPlanner(2); arm = tp.arm; cid=arm.physics_client_id
free, arm_idx, lo, hi, is_arm = _ik_meta(arm)
pose = Pose((0.4,0.1,0.25), tp.down)
for it in (1,5,20,100):
    t=time.time()
    for _ in range(50):
        p.calculateInverseKinematics(arm.robot_id, arm.end_effector_id, pose.position, targetOrientation=pose.orientation, maxNumIterations=it, residualThreshold=1e-9, physicsClientId=cid)
    print("iters",it,"ms/call",(time.time()-t)/50*1000)
t=time.time()
for _ in range(50):
    p.calculateInverseKinematics(arm.robot_id, arm.end_effector_id, pose.position, targetOrientation=pose.orientation, lowerLimits=lo, upperLimits=hi, maxNumIterations=20, residualThreshold=1e-9, physicsClientId=cid)
print("limits20 ms/call",(time.time()-t)/50*1000)
t=time.time()
for _ in range(50):
    p.getLinkState(arm.robot_id, arm.end_effector_id, computeForwardKinematics=True, physicsClientId=cid)
print("linkstate ms",(time.time()-t)/50*1000)
print(p.getNumJoints(arm.robot_id, physicsClientId=cid))
