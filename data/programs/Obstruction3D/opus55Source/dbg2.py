import sys, time, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import approach as A
from approach import GeneratedApproach, _Info
c=int(sys.argv[1]); seed=int(sys.argv[2]); target=sys.argv[3]
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv", count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[c], eval_counts=[c])
s, info = env.reset(seed=seed, options={'object_count': c})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(s, info)
zeros=0
for i in range(1000):
    a = ap.get_action(s)
    if not np.any(a): zeros+=1
    else: zeros=0
    if zeros>2: break
    s, r, term, _, _ = env.step(a)
    if term: print('solved'); sys.exit()
A.DEBUG=True
pl = ap.planner; pl.deadline=1e18
ss = ap._sim_state(s); I=_Info(ss)
for n in sorted(I.objs):
    if n!='robot': print(n, I.pos(n).round(3), I.he(n).round(3))
OPTS = [(y, z) for z in (0.034, 0.03, 0.028, 0.026, 0.024, 0.022) for y in (0.0, 1.5708)] if len(sys.argv) > 4 else pl.grasp_options(I, target)
for yaw, z in OPTS:
    print('opt', yaw, z)
    # stage detail of precheck
    c_ = I.pos(target); he=I.he(target); top=c_[2]+he[2]; quat=A._yaw_quat(yaw)
    pl.set(ss); q = pl.ik([c_[0],c_[1],max(0.25, top+0.08)], quat, I.joints())
    if q is None: print('  hover IK fail'); continue
    fake = pl.with_joints(ss, q); plan=[]
    acts = pl.move_actions(fake, [c_[0],c_[1],top+z], quat)
    if acts is None: print('  descent IK fail'); continue
    st = pl.run(fake, acts)
    if st is None: print('  descent collision'); continue
    _, st2, _ = pl.gripper(st, True)
    print('  held after close:', _Info(st2).held())
    if _Info(st2).held() == target:
        pl.set(st2); ee_now,_ = pl.ee_pose()
        ld = pl.lift_dist(I, target)
        acts = pl.move_actions(st2, [ee_now[0], ee_now[1], ee_now[2]+ld], quat, 0.003, 0)
        print('  lift dist', ld, 'acts', None if acts is None else len(acts))
        if acts is not None:
            st3 = pl.run(st2, acts); print('  lift ok', st3 is not None)
print('---- detailed close test')
from pybullet_helpers.inverse_kinematics import check_body_collisions
import pybullet as p
c_ = I.pos(target); he=I.he(target); top=c_[2]+he[2]; quat=A._yaw_quat(0.0)
pl.set(ss); q = pl.ik([c_[0],c_[1],0.25], quat, I.joints())
fake = pl.with_joints(ss, q)
acts = pl.move_actions(fake, [c_[0],c_[1],top+0.03], quat)
st = pl.run(fake, acts)
sim = pl.sim
pl.set(st)
p.performCollisionDetection(physicsClientId=sim.physics_client_id)
for obj in sorted(sim._get_movable_object_names()):
    oid = sim._object_name_to_pybullet_id(obj)
    if check_body_collisions(oid, sim.end_effector_viz_id, sim.physics_client_id, perform_collision_detection=False):
        print('in zone:', obj)
print('ee', pl.ee_pose()[0], 'viz aabb', p.getAABB(sim.end_effector_viz_id, physicsClientId=sim.physics_client_id))
arm = pl.arm
cur = arm.get_joint_positions()
names = {sim._object_name_to_pybullet_id(o): o for o in sim._get_movable_object_names()}
for fs in [0.0, 0.2, 0.4, 0.6, 0.8]:
    arm.set_finger_state(fs)
    p.performCollisionDetection(physicsClientId=sim.physics_client_id)
    hits = []
    for bid in sim._get_collision_object_ids():
        pts = p.getClosestPoints(arm.robot_id, bid, 0.0, physicsClientId=sim.physics_client_id)
        for pt in pts:
            hits.append((names.get(bid, bid), p.getJointInfo(arm.robot_id, pt[3], physicsClientId=sim.physics_client_id)[12].decode() if pt[3]>=0 else 'base', round(pt[8],4)))
    print('fs', fs, sorted(set(hits))[:6])
