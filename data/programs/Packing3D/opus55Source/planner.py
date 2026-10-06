"""Twin-simulation planner for Packing3D."""
from __future__ import annotations

import math

import numpy as np
from kinder.envs.kinematic3d.packing3d import ObjectCentricPacking3DEnv
from kinder.envs.kinematic3d.utils import remove_fingers_from_extended_joints
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    pybullet_inverse_kinematics,
)
from pybullet_helpers.joint import get_jointwise_difference

MAX_D = 0.2
DOWN_QUAT = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def yaw_quat(theta):
    return (0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2))


class PlanFailure(Exception):
    pass


class TwinPlanner:
    def __init__(self, num_parts):
        self.n = num_parts
        self.env = ObjectCentricPacking3DEnv(num_parts=num_parts,
                                             allow_state_access=True)
        self.env.reset(seed=0)
        self.arm = self.env.robot.arm
        self.jinfos = self.arm.get_arm_joint_infos()[:7]
        self.down = self.arm.get_end_effector_pose().orientation
        self.actions = []
        self.expected = []
        self.state = None

    # ---------------- low level ----------------
    def set_state(self, state):
        self.env.set_state(state)
        self.state = self.env.get_state()

    def cur_joints(self):
        return np.array(remove_fingers_from_extended_joints(
            self.arm.get_joint_positions()))

    def step(self, action):
        obs, _, term, _, _ = self.env.step(np.asarray(action, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.expected.append(self.cur_joints())
        self.state = obs
        self.terminated = term
        return obs

    def ik(self, pose):
        snap = self.env.get_state()
        try:
            q = jac_ik(self.arm, pose)
        except InverseKinematicsError:
            q = None
        except Exception:
            q = None
        self.env.set_state(snap)
        if q is None:
            raise PlanFailure("ik")
        return np.array(q[:7])

    def joint_diff(self, q):
        cur = self.cur_joints()
        return np.array(get_jointwise_difference(self.jinfos, list(q), list(cur)))

    def move_joints(self, q, tol=1e-4, max_steps=40):
        for _ in range(max_steps):
            d = self.joint_diff(q)
            if np.max(np.abs(d)) < tol:
                return
            before = self.cur_joints()
            a = np.zeros(11, dtype=np.float32)
            a[3:10] = np.clip(d, -MAX_D, MAX_D)
            self.step(a)
            if np.max(np.abs(self.cur_joints() - before)) < 1e-9:
                raise PlanFailure("blocked")
        if np.max(np.abs(self.joint_diff(q))) > 1e-3:
            raise PlanFailure("not reached")

    def ee(self):
        return self.arm.get_end_effector_pose()

    def move_cartesian(self, target, step_len=0.02, yaw_step=0.15):
        """Move EE to target via straight-line interpolation."""
        start = self.ee()
        p0 = np.array(start.position)
        p1 = np.array(target.position)
        dist = np.linalg.norm(p1 - p0)
        # orientation difference measured as quaternion angle
        q0 = np.array(start.orientation)
        q1 = np.array(target.orientation)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        ang = 2 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))
        k = max(1, int(math.ceil(max(dist / step_len, ang / yaw_step))))
        for i in range(1, k + 1):
            t = i / k
            p = p0 + t * (p1 - p0)
            q = slerp(q0, q1, t)
            qj = self.ik(Pose(tuple(p), tuple(q)))
            self.move_joints(qj)

    def gripper(self, close, max_tries=3):
        a = np.zeros(11, dtype=np.float32)
        a[10] = -1.0 if close else 1.0
        for _ in range(max_tries):
            self.step(a)
            held = self.env._grasped_object is not None
            if held == close:
                return
        raise PlanFailure("gripper")

    # ---------------- high level ----------------
    def down_yaw(self, yaw):
        return quat_mul(yaw_quat(yaw), self.down)


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ee_yaw(planner, orientation):
    """Yaw gamma such that orientation ~= yaw_quat(gamma) * down."""
    R = _quat_to_mat(orientation) @ _quat_to_mat(planner.down).T
    return math.atan2(R[1, 0], R[0, 0])


def _move_joints_linear(self, q, grip=None, tol=1e-4):
    """Straight-line joint move at max speed; optional gripper cmd on last step."""
    d = self.joint_diff(q)
    m = float(np.max(np.abs(d)))
    k = max(1, int(math.ceil(m / (MAX_D - 1e-6))))
    if m < tol:
        k = 0
    for i in range(k):
        d = self.joint_diff(q)
        before = self.cur_joints()
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = d / (k - i)
        a[3:10] = np.clip(a[3:10], -MAX_D, MAX_D)
        if grip is not None and i == k - 1:
            a[10] = grip
        self.step(a)
        if np.max(np.abs(self.cur_joints() - before)) < 1e-9 and m > tol:
            raise PlanFailure("blocked")
    if k == 0 and grip is not None:
        a = np.zeros(11, dtype=np.float32)
        a[10] = grip
        self.step(a)
    if np.max(np.abs(self.joint_diff(q))) > 1e-3:
        raise PlanFailure("not reached")


def _move_direct(self, pose, grip=None):
    q = self.ik(pose)
    self.move_joints_linear(q, grip=grip)


def _snapshot(self):
    return (self.env.get_state(), len(self.actions))


def _restore(self, snap):
    self.env.set_state(snap[0])
    self.state = self.env.get_state()
    del self.actions[snap[1]:]
    del self.expected[snap[1]:]


def _set_check_res(self, res):
    import dataclasses
    cfg = self.env.config
    try:
        object.__setattr__(cfg, "max_collision_check_step", res)
    except Exception:
        self.env.config = dataclasses.replace(cfg, max_collision_check_step=res)


TwinPlanner.set_check_res = _set_check_res


TwinPlanner.move_joints_linear = _move_joints_linear
TwinPlanner.move_direct = _move_direct
TwinPlanner.snapshot = _snapshot
TwinPlanner.restore = _restore


_TOL = 1e-6


def fast_convex_components_overlap(first, second):
    """Vectorized equivalent of Packing3D's SAT overlap test."""
    v1, n1, e1 = first
    v2, n2, e2 = second
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    cr = np.cross(np.asarray(e1)[:, None, :], np.asarray(e2)[None, :, :]).reshape(-1, 3)
    axes = np.concatenate([np.asarray(n1), np.asarray(n2), cr], axis=0)
    norms = np.linalg.norm(axes, axis=1)
    mask = norms > _TOL
    axes = axes[mask] / norms[mask, None]
    p1 = v1 @ axes.T
    p2 = v2 @ axes.T
    overlap = np.minimum(p1.max(0), p2.max(0)) - np.maximum(p1.min(0), p2.min(0))
    return not bool(np.any(overlap <= _TOL))


_orig_init = TwinPlanner.__init__


def _patched_init(self, num_parts):
    _orig_init(self, num_parts)
    self.env._convex_components_overlap = fast_convex_components_overlap


TwinPlanner.__init__ = _patched_init


def _install_overlap_early_out(env):
    import pybullet as p
    orig = env._parts_overlap

    def radius(name):
        try:
            pid = env._object_name_to_pybullet_id(name)
            if pid in env._part_id_to_half_extents:
                hx, hy, hz = env._part_id_to_half_extents[pid]
                return math.sqrt(hx * hx + hy * hy + (abs(hz) + 0.06) ** 2) + 0.005
            a, b, depth, _ = env._part_ids_to_triangle_features[pid]
            return math.hypot(a, b) + abs(depth) + 0.06 + 0.005
        except Exception:
            return 10.0

    def fast_overlap(a, b):
        try:
            pa = p.getBasePositionAndOrientation(env._object_name_to_pybullet_id(a),
                                                 physicsClientId=env.physics_client_id)[0]
            pb = p.getBasePositionAndOrientation(env._object_name_to_pybullet_id(b),
                                                 physicsClientId=env.physics_client_id)[0]
            if np.linalg.norm(np.subtract(pa, pb)) > radius(a) + radius(b):
                return False
        except Exception:
            pass
        return orig(a, b)

    env._parts_overlap = fast_overlap


_orig_init2 = TwinPlanner.__init__


def _patched_init2(self, num_parts):
    _orig_init2(self, num_parts)
    _install_overlap_early_out(self.env)


TwinPlanner.__init__ = _patched_init2


# ---------------------------------------------------------------------------
# Fast IK: let pybullet iterate internally, validate once in python.
import pybullet as _p
from pybullet_helpers.joint import get_joint_infos as _gji, get_joints as _gj

_IK_CACHE = {}


def _ik_meta(robot):
    key = (robot.robot_id, robot.physics_client_id)
    if key not in _IK_CACHE:
        cid = robot.physics_client_id
        infos = _gji(robot.robot_id, _gj(robot.robot_id, physics_client_id=cid),
                     physics_client_id=cid)
        free = [ji.jointIndex for ji in infos if ji.qIndex > -1]
        arm_idx = [free.index(j) for j in robot.arm_joints]
        lo = [ji.jointLowerLimit for ji in infos if ji.qIndex > -1]
        hi = [ji.jointUpperLimit for ji in infos if ji.qIndex > -1]
        armset = set(robot.arm_joints)
        is_arm = [j in armset for j in free]
        _IK_CACHE[key] = (free, arm_idx, lo, hi, is_arm)
    return _IK_CACHE[key]


def fast_ik(robot, pose, atol=1e-3, tries=30):
    """Returns full free-joint solution list (like pybullet_inverse_kinematics)
    or raises InverseKinematicsError. Leaves the robot at the solution."""
    free, arm_idx, lo0, hi0, is_arm = _ik_meta(robot)
    cid = robot.physics_client_id
    rid = robot.robot_id
    cur = [s[0] for s in _p.getJointStates(rid, free, physicsClientId=cid)]
    lo = [l if a else c for l, c, a in zip(lo0, cur, is_arm)]
    hi = [h if a else c for h, c, a in zip(hi0, cur, is_arm)]
    tp_ = np.array(pose.position)
    tq = np.array(pose.orientation)
    for _ in range(tries):
        vals = _p.calculateInverseKinematics(
            rid, robot.end_effector_id, pose.position,
            targetOrientation=pose.orientation, lowerLimits=lo, upperLimits=hi,
            maxNumIterations=20, residualThreshold=1e-7, physicsClientId=cid)
        ok_lim = all((l > h) or (l <= v <= h) for l, v, h in zip(lo, vals, hi))
        for j, v in zip(free, vals):
            _p.resetJointState(rid, j, targetValue=v, physicsClientId=cid)
        if not ok_lim:
            continue
        ls = _p.getLinkState(rid, robot.end_effector_id, computeForwardKinematics=True,
                             physicsClientId=cid)
        pp, qq = np.array(ls[4]), np.array(ls[5])
        if np.all(np.abs(pp - tp_) <= atol) and (
                np.all(np.abs(qq - tq) <= atol) or np.all(np.abs(qq + tq) <= atol)):
            return list(vals)
    raise InverseKinematicsError("fast ik failed")


def _qmul(a, b):
    return np.array(quat_mul(a, b))


def jac_ik(robot, pose, atol=1e-3, max_iters=30, damping=1e-3):
    """Damped least-squares IK on the 7 arm joints, seeded from current joints.
    Returns full free-joint list; raises InverseKinematicsError on failure.
    Leaves the robot at the final iterate."""
    free, arm_idx, lo0, hi0, is_arm = _ik_meta(robot)
    cid = robot.physics_client_id
    rid = robot.robot_id
    ee = robot.end_effector_id
    arm7 = free[:7]
    allq = [s[0] for s in _p.getJointStates(rid, free, physicsClientId=cid)]
    q = np.array(allq[:7])
    tp_ = np.array(pose.position, dtype=float)
    tq = np.array(pose.orientation, dtype=float)
    zeros = [0.0] * len(free)
    for it in range(max_iters):
        ls = _p.getLinkState(rid, ee, computeForwardKinematics=True, physicsClientId=cid)
        pl = np.array(ls[4])
        ql = np.array(ls[5])
        ep = tp_ - pl
        qe = _qmul(tq, (-ql[0], -ql[1], -ql[2], ql[3]))
        if qe[3] < 0:
            qe = -qe
        eo = 2.0 * qe[:3]
        if (np.all(np.abs(ep) <= atol * 0.5) and
                (np.all(np.abs(ql - tq) <= atol * 0.5) or np.all(np.abs(ql + tq) <= atol * 0.5))):
            for l, v, h in zip(lo0[:7], q, hi0[:7]):
                if l <= h and not (l - 1e-9 <= v <= h + 1e-9):
                    raise InverseKinematicsError("limits")
            return list(q) + list(allq[7:])
        objq = list(q) + list(allq[7:])
        jl, ja = _p.calculateJacobian(rid, ee, [0.0, 0.0, 0.0], objq, zeros, zeros,
                                      physicsClientId=cid)
        jl = np.array(jl)[:, :7]
        ja = np.array(ja)[:, :7]
        r = pl - np.array(ls[0])
        # linear velocity of link frame origin = v_com + w x r
        S = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
        jl = jl - S @ ja
        J = np.vstack([jl, ja])
        e = np.concatenate([ep, eo])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
        mx = np.max(np.abs(dq))
        if mx > 0.5:
            dq *= 0.5 / mx
        q = q + dq
        for i, (l, h) in enumerate(zip(lo0[:7], hi0[:7])):
            if l <= h:
                q[i] = min(max(q[i], l), h)
        _p.resetJointStatesMultiDof(rid, arm7, targetValues=[[float(v)] for v in q],
                                    physicsClientId=cid)
    raise InverseKinematicsError("jac ik no convergence")


# ---------------------------------------------------------------------------
# Cached convex components + AABB early-out for part/part overlap.
def _quat_mat_np(q):
    return _quat_to_mat(q)


def _install_fast_components(env):
    import pybullet as p
    orig_get = env._get_part_convex_components
    local_cache = {}
    world_cache = {}

    def geom_key(name):
        pid = env._object_name_to_pybullet_id(name)
        feats = (env._part_ids_to_type.get(pid).__name__ if hasattr(env._part_ids_to_type.get(pid), "__name__") else str(env._part_ids_to_type.get(pid)),
                 tuple(np.round(np.asarray(env._part_id_to_half_extents.get(pid, ()), dtype=float), 9)),
                 repr(env._part_ids_to_triangle_features.get(pid)))
        return (name, pid, feats)

    def local_geom(name):
        gk = geom_key(name)
        if local_cache.get(name, (None,))[0] != gk:
            world_cache.pop(name, None)
            pid = env._object_name_to_pybullet_id(name)
            pose = p.getBasePositionAndOrientation(pid, physicsClientId=env.physics_client_id)
            comps = orig_get(name)
            R = _quat_to_mat(pose[1])
            t = np.asarray(pose[0])
            loc = []
            for v, nrm, e in comps:
                loc.append(((np.asarray(v) - t) @ R, np.asarray(nrm) @ R, np.asarray(e) @ R))
            local_cache[name] = (gk, pid, loc)
        return local_cache[name][1:]

    def get_components(name):
        pid, loc = local_geom(name)
        pos, orn = p.getBasePositionAndOrientation(pid, physicsClientId=env.physics_client_id)
        key = (pos, orn)
        hit = world_cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        R = _quat_to_mat(orn)
        t = np.asarray(pos)
        comps = []
        for v, nrm, e in loc:
            wv = v @ R.T + t
            comps.append((wv, nrm @ R.T, e @ R.T, wv.min(0), wv.max(0)))
        world_cache[name] = (key, comps)
        return comps

    def sat(first, second):
        v1, n1, e1 = first[0], first[1], first[2]
        v2, n2, e2 = second[0], second[1], second[2]
        a = e1[:, None, :]
        b = e2[None, :, :]
        cr = np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                       a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                       a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1).reshape(-1, 3)
        axes = np.concatenate([n1, n2, cr], axis=0)
        norms = np.sqrt((axes * axes).sum(1))
        mask = norms > _TOL
        axes = axes[mask] / norms[mask, None]
        p1 = v1 @ axes.T
        p2 = v2 @ axes.T
        overlap = np.minimum(p1.max(0), p2.max(0)) - np.maximum(p1.min(0), p2.min(0))
        return not bool(np.any(overlap <= _TOL))

    def parts_overlap(a, b):
        ca = get_components(a)
        cb = get_components(b)
        for f in ca:
            for s in cb:
                if np.any(f[4] < s[3] - 1e-5) or np.any(s[4] < f[3] - 1e-5):
                    continue
                if sat(f, s):
                    return True
        return False

    env._get_part_convex_components_fast = get_components
    env._parts_overlap = parts_overlap


_orig_init3 = TwinPlanner.__init__


def _patched_init3(self, num_parts):
    _orig_init3(self, num_parts)
    _install_fast_components(self.env)


TwinPlanner.__init__ = _patched_init3
