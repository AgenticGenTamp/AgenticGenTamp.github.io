import numpy as np, logging
logging.disable(logging.WARNING)
from kinder.envs.kinematic3d.transport3d import ObjectCentricTransport3DEnv
from pybullet_helpers.inverse_kinematics import pybullet_inverse_kinematics
from pybullet_helpers.geometry import Pose
from scipy.spatial.transform import Rotation as R
env = ObjectCentricTransport3DEnv(num_cubes=1)
arm = env.robot.arm
s,_ = env.reset(seed=0)
DOWN = tuple(R.from_matrix(np.array([[0,1,0],[1,0,0],[0,0,-1]]).T.T).as_quat())  # x->world y? 
DOWN = tuple(R.from_euler("xyz",[np.pi,0,0]).as_quat())
def P(*a): print("##",*a)
def step_to(jt=None, base=None, grip=0.0, maxit=200):
    global s
    for _ in range(maxit):
        cur = np.array(s.joint_positions); b=s.base_pose
        d = np.zeros(7) if jt is None else np.clip(np.array(jt)-cur, -0.2, 0.2)
        bd = np.zeros(3) if base is None else np.clip(np.array(base)-np.array([b.x,b.y,b.rot]),-0.2,0.2)
        if np.abs(d).max()<1e-5 and np.abs(bd).max()<1e-5: return True
        s2,_,term,_,_ = env.step(np.concatenate([bd, d, [grip]]).astype(np.float32))
        if np.allclose(s2.joint_positions,cur) and np.allclose([s2.base_pose.x,s2.base_pose.y,s2.base_pose.rot],[b.x,b.y,b.rot]):
            P("blocked at", s2.base_pose, np.round(arm.get_end_effector_pose().position,3)); s=s2; return False
        s = s2
    return False
def ik(pos, orn=DOWN):
    return pybullet_inverse_kinematics(arm, Pose(tuple(pos), orn), validate=True)[:7]
cube = np.array(s.get_object_pose("cube0").position); P("cube", cube, "box", s.get_object_pose("box0").position)
# drive: y first then x (base far from table)
step_to(base=(0.0, cube[1], 0)); step_to(base=(cube[0]-0.55, cube[1], 0)); P("base", s.base_pose)
P("pre", step_to(ik(cube+[0,0,0.15]))); P("down", step_to(ik(cube+[0,0,0.0])), np.round(arm.get_end_effector_pose().position,3))
s,*_ = env.step(np.array([0]*10+[-1],dtype=np.float32)); P("grasped:", s.grasped_object, "finger", round(s.finger_state,3))
P("lift", step_to(ik(cube+[0,0,0.5])), np.round(s.get_object_pose("cube0").position,3))
# carry to table: table x in [0.4,0.8], y in [-0.4,0.4], top 0.4. approach from -x side? robot at x<0.4-? facing +x
step_to(base=(s.base_pose.x, 0.0, 0)); step_to(base=(0.0, 0.0, 0)); P("base", s.base_pose)
tgt = np.array([0.6,0.0,0.4+0.025])
P("pre place", step_to(ik(tgt+[0,0,0.1])), np.round(s.get_object_pose("cube0").position,3))
P("place", step_to(ik(tgt+[0,0,0.003])), np.round(s.get_object_pose("cube0").position,3))
s,_,term,*_ = env.step(np.array([0]*10+[1],dtype=np.float32)); P("released:", s.grasped_object, "finger", s.finger_state, "term", term)
P("retreat", step_to(ik(tgt+[0,0,0.3])))
P("goal", env.goal_reached(), "ee", np.round(arm.get_end_effector_pose().position,3))
