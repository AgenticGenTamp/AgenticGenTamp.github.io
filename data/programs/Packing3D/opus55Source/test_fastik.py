import time, math, numpy as np
from planner import TwinPlanner, jac_ik as fast_ik, yaw_quat, quat_mul
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import pybullet_inverse_kinematics
tp = TwinPlanner(2); arm = tp.arm
rng = np.random.default_rng(0)
home = list(arm.get_joint_positions())
res = {'both':0,'orig_only':0,'fast_only':0,'none':0}; to=tf=0; bad=0
for i in range(300):
    pos = (rng.uniform(0.15,0.6), rng.uniform(-0.35,0.35), rng.uniform(0.12,0.4))
    ori = quat_mul(yaw_quat(rng.uniform(-3.14,3.14)), tp.down)
    pose = Pose(pos, ori)
    seed = home if i%2 else list(np.array(home)+np.r_[rng.normal(0,0.3,7),np.zeros(len(home)-7)])
    arm.set_joints(seed); t=time.time()
    try: pybullet_inverse_kinematics(arm, pose, validate=True); o=True
    except Exception: o=False
    to+=time.time()-t
    arm.set_joints(seed); t=time.time()
    try: fast_ik(arm, pose); f=True
    except Exception: f=False
    tf+=time.time()-t
    if f:
        e = arm.get_end_effector_pose()
        if not e.allclose(pose, atol=1.5e-3): bad+=1
    res['both' if o and f else 'orig_only' if o else 'fast_only' if f else 'none']+=1
print(res, 'orig', round(to,2), 'fast', round(tf,2), 'bad', bad)
