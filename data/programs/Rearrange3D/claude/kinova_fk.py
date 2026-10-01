"""
Forward / inverse kinematics for the Kinova Gen3 7-DoF arm (spherical wrist).

Pure numpy + stdlib.  All kinematic constants come from the official
`kortex_description` URDF (GEN3_URDF_V12 / gen3.xacro, 7-DoF *spherical*
wrist variant).  Every number is exposed in `URDF_JOINTS` /
`EE_FRAME_OFFSET` / `TOOL_OFFSET` / `ARM_MOUNT` so they can be tweaked or
calibrated later without touching the math.

Frame chain
-----------
    world --ARM_MOUNT(base_pose)--> arm base_link
        --joint_1..joint_7--> bracelet_link      (frame of the last joint)
        --EE_FRAME_OFFSET--> end_effector_link   (fixed URDF joint)
        --TOOL_OFFSET--> TCP (gripper tool centre point)

URDF convention: a joint origin transform is  Trans(xyz) @ Rot_rpy(rpy)
(fixed-axis roll-pitch-yaw, i.e. Rz(yaw) @ Ry(pitch) @ Rx(roll)), and the
joint rotation about `axis` is applied *after* the origin.
"""

import math
import numpy as np

# --------------------------------------------------------------------------
# URDF parameters (kortex_description, gen3 7dof spherical wrist)
# --------------------------------------------------------------------------
PI = math.pi

#: One entry per actuated joint, in order.  ``xyz``/``rpy`` are the URDF
#: <origin> of the joint (parent frame -> joint frame), ``axis`` is the
#: rotation axis expressed in the joint frame, ``limit`` is (lo, hi) in
#: radians or ``None`` for a continuous joint.
URDF_JOINTS = [
    # name        parent -> child                 xyz                          rpy              axis        limit
    dict(name="joint_1", parent="base_link",      child="shoulder_link",
         xyz=(0.0,  0.0,        0.15643),      rpy=(-PI,   0.0, 0.0), axis=(0, 0, 1), limit=None),
    dict(name="joint_2", parent="shoulder_link",  child="half_arm_1_link",
         xyz=(0.0,  0.005375,  -0.12838),      rpy=( PI/2, 0.0, 0.0), axis=(0, 0, 1),
         limit=(-math.radians(128.9), math.radians(128.9))),
    dict(name="joint_3", parent="half_arm_1_link", child="half_arm_2_link",
         xyz=(0.0, -0.21038,   -0.006375),     rpy=(-PI/2, 0.0, 0.0), axis=(0, 0, 1), limit=None),
    dict(name="joint_4", parent="half_arm_2_link", child="forearm_link",
         xyz=(0.0,  0.006375,  -0.21038),      rpy=( PI/2, 0.0, 0.0), axis=(0, 0, 1),
         limit=(-math.radians(147.8), math.radians(147.8))),
    dict(name="joint_5", parent="forearm_link",   child="spherical_wrist_1_link",
         xyz=(0.0, -0.20843,   -0.006375),     rpy=(-PI/2, 0.0, 0.0), axis=(0, 0, 1), limit=None),
    dict(name="joint_6", parent="spherical_wrist_1_link", child="spherical_wrist_2_link",
         xyz=(0.0,  0.00017505, -0.10593),     rpy=( PI/2, 0.0, 0.0), axis=(0, 0, 1),
         limit=(-math.radians(120.3), math.radians(120.3))),
    dict(name="joint_7", parent="spherical_wrist_2_link", child="bracelet_link",
         xyz=(0.0, -0.10593,   -0.00017505),   rpy=(-PI/2, 0.0, 0.0), axis=(0, 0, 1), limit=None),
]

N_JOINTS = len(URDF_JOINTS)

#: Fixed URDF joint ``end_effector`` : bracelet_link -> end_effector_link.
#: (xyz = 0 0 -0.061525, rpy = pi 0 0).  With the Robotiq 2F-85 the same
#: transform is used for the ``tool_frame`` / gripper mounting flange.
EE_FRAME_OFFSET = dict(xyz=(0.0, 0.0, -0.061525), rpy=(PI, 0.0, 0.0))

#: Transform from the ``end_effector_link`` frame to the gripper TCP.
#: Robotiq 2F-85 style: ~0.12-0.13 m straight out along the tool +z axis.
#: Tweak this (or set it to a full 4x4) to calibrate a different gripper.
TOOL_OFFSET = dict(xyz=(0.0, 0.0, 0.130), rpy=(0.0, 0.0, 0.0))

#: Arm base_link relative to the mobile-base origin.  Default: identity
#: rotation, z = 0.  Set e.g. xyz=(0.2, 0.0, 0.7) for a real mobile base.
ARM_MOUNT = dict(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0))

#: Joint limits as arrays (continuous joints get +-inf).
JOINT_LIMITS_LOWER = np.array([-np.inf if j["limit"] is None else j["limit"][0]
                               for j in URDF_JOINTS])
JOINT_LIMITS_UPPER = np.array([np.inf if j["limit"] is None else j["limit"][1]
                               for j in URDF_JOINTS])

#: Velocity limits (rad/s) from the datasheet, handy for IK step clamping.
JOINT_VELOCITY_LIMITS = np.array([1.3963] * 4 + [1.2218] * 3)


# --------------------------------------------------------------------------
# small SE(3) helpers
# --------------------------------------------------------------------------
def rpy_to_matrix(rpy):
    """Fixed-axis roll-pitch-yaw (URDF convention) -> 3x3 rotation."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def make_transform(xyz, rpy):
    """URDF <origin> -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def axis_angle_to_matrix(axis, angle):
    """Rodrigues formula for a unit (or normalisable) axis."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    c, s = math.cos(angle), math.sin(angle)
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def transform_inverse(T):
    Ti = np.eye(4)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def rotation_log(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    cos_t = (np.trace(R) - 1.0) * 0.5
    cos_t = min(1.0, max(-1.0, cos_t))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if abs(PI - theta) < 1e-6:
        # near pi: use the symmetric part for numerical stability
        A = 0.5 * (R + np.eye(3))
        v = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(v))
        if v[k] > 1e-9:
            v = A[:, k] / v[k]
        n = np.linalg.norm(v)
        if n > 1e-12:
            v = v / n
        return v * theta
    return (theta / (2.0 * math.sin(theta))) * np.array([R[2, 1] - R[1, 2],
                                                         R[0, 2] - R[2, 0],
                                                         R[1, 0] - R[0, 1]])


# --------------------------------------------------------------------------
# pre-computed constant transforms
# --------------------------------------------------------------------------
def _origins():
    return [make_transform(j["xyz"], j["rpy"]) for j in URDF_JOINTS]


def tool_transform():
    """bracelet_link -> TCP  (EE_FRAME_OFFSET followed by TOOL_OFFSET)."""
    return (make_transform(EE_FRAME_OFFSET["xyz"], EE_FRAME_OFFSET["rpy"])
            @ make_transform(TOOL_OFFSET["xyz"], TOOL_OFFSET["rpy"]))


def arm_mount_transform(base_pose=None):
    """mobile-base pose (x, y, theta) + ARM_MOUNT -> world->arm base_link."""
    T = np.eye(4)
    if base_pose is not None:
        x, y, theta = (list(base_pose) + [0.0, 0.0, 0.0])[:3]
        T = make_transform((x, y, 0.0), (0.0, 0.0, theta))
    return T @ make_transform(ARM_MOUNT["xyz"], ARM_MOUNT["rpy"])


# --------------------------------------------------------------------------
# forward kinematics
# --------------------------------------------------------------------------
def fk_all(q, base_pose=None):
    """
    Full kinematic chain.

    Returns
    -------
    frames : list of 9 4x4 transforms, all in world coordinates:
             [arm base_link, after joint_1, ..., after joint_7, TCP]
             i.e. ``frames[i + 1]`` is the child frame of joint ``i+1``.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != N_JOINTS:
        raise ValueError("q must have %d elements, got %d" % (N_JOINTS, q.size))

    T = arm_mount_transform(base_pose)
    frames = [T.copy()]
    for i, (origin, j) in enumerate(zip(_origins(), URDF_JOINTS)):
        Tj = np.eye(4)
        Tj[:3, :3] = axis_angle_to_matrix(j["axis"], q[i])
        T = T @ origin @ Tj
        frames.append(T.copy())
    frames.append(T @ tool_transform())
    return frames


def fk(q, base_pose=None):
    """4x4 world transform of the tool frame (TCP) for joint angles ``q``."""
    return fk_all(q, base_pose)[-1]


def fk_position(q, base_pose=None):
    return fk(q, base_pose)[:3, 3]


def fk_flange(q, base_pose=None):
    """4x4 world transform of ``end_effector_link`` (no TOOL_OFFSET)."""
    frames = fk_all(q, base_pose)
    return frames[-2] @ make_transform(EE_FRAME_OFFSET["xyz"], EE_FRAME_OFFSET["rpy"])


# --------------------------------------------------------------------------
# geometric Jacobian
# --------------------------------------------------------------------------
def jacobian(q, base_pose=None):
    """
    6x7 geometric Jacobian of the TCP.

    Rows 0:3 -> linear velocity, rows 3:6 -> angular velocity.
    With ``base_pose=None`` (default) the Jacobian is expressed in the arm
    base_link frame, which is what a controller normally wants.
    """
    frames = fk_all(q, base_pose)
    p_ee = frames[-1][:3, 3]
    J = np.zeros((6, N_JOINTS))
    for i, j in enumerate(URDF_JOINTS):
        Ti = frames[i + 1]                      # child frame of joint i+1
        z = Ti[:3, :3] @ (np.asarray(j["axis"], dtype=float))
        z = z / np.linalg.norm(z)
        p = Ti[:3, 3]
        J[:3, i] = np.cross(z, p_ee - p)
        J[3:, i] = z
    return J


def jacobian_fd(q, base_pose=None, eps=1e-6):
    """Finite-difference Jacobian of ``fk`` (for validation)."""
    q = np.asarray(q, dtype=float).reshape(-1)
    T0 = fk(q, base_pose)
    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        dq = q.copy()
        dq[i] += eps
        T1 = fk(dq, base_pose)
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        J[3:, i] = rotation_log(T1[:3, :3] @ T0[:3, :3].T) / eps
    return J


# --------------------------------------------------------------------------
# inverse kinematics (damped least squares)
# --------------------------------------------------------------------------
def clamp_to_limits(q):
    """Wrap continuous joints to (-pi, pi] and clip the limited ones."""
    q = np.asarray(q, dtype=float).copy()
    for i, j in enumerate(URDF_JOINTS):
        if j["limit"] is None:
            q[i] = (q[i] + PI) % (2 * PI) - PI
        else:
            q[i] = min(j["limit"][1], max(j["limit"][0], q[i]))
    return q


def pose_error(T_current, target_pos, target_rot=None):
    """6-vector [translation error; rotation error] in the current frame's parent."""
    e = np.zeros(6)
    e[:3] = np.asarray(target_pos, dtype=float) - T_current[:3, 3]
    if target_rot is not None:
        e[3:] = rotation_log(np.asarray(target_rot, dtype=float) @ T_current[:3, :3].T)
    return e


def ik(target_pos, target_rot=None, q_init=None, base_pose=None,
       max_iters=300, damping=0.05, pos_tol=1e-4, rot_tol=1e-3,
       step_clamp=0.35, orientation_weight=1.0, seed=0, restarts=5):
    """
    Damped-least-squares IK for the TCP.

    Parameters
    ----------
    target_pos : (3,) desired TCP position (world frame if ``base_pose`` given).
    target_rot : optional 3x3 desired TCP orientation.  If ``None``, only
                 position is solved (the null space is left free).
    q_init     : optional (7,) seed configuration.
    restarts   : number of random restarts if the first solve fails.

    Returns
    -------
    q       : (7,) joint solution (clamped to the Gen3 limits)
    success : bool, True if within ``pos_tol`` / ``rot_tol``
    info    : dict with 'pos_err', 'rot_err', 'iters'
    """
    rng = np.random.default_rng(seed)
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)

    lo = np.where(np.isfinite(JOINT_LIMITS_LOWER), JOINT_LIMITS_LOWER, -PI)
    hi = np.where(np.isfinite(JOINT_LIMITS_UPPER), JOINT_LIMITS_UPPER, PI)

    if q_init is None:
        q_init = np.zeros(N_JOINTS)
    q_init = clamp_to_limits(q_init)

    best = (np.inf, q_init.copy(), 0.0, 0.0, 0)
    for attempt in range(max(1, restarts)):
        q = q_init.copy() if attempt == 0 else rng.uniform(lo, hi)
        it = 0
        for it in range(1, max_iters + 1):
            T = fk(q, base_pose)
            e = pose_error(T, target_pos, target_rot)
            if target_rot is None:
                e[3:] = 0.0
            else:
                e[3:] *= orientation_weight
            pos_err = float(np.linalg.norm(e[:3]))
            rot_err = float(np.linalg.norm(e[3:]) / max(orientation_weight, 1e-9))
            if pos_err < pos_tol and (target_rot is None or rot_err < rot_tol):
                return clamp_to_limits(q), True, dict(pos_err=pos_err,
                                                      rot_err=rot_err, iters=it)
            J = jacobian(q, base_pose)
            if target_rot is None:
                J = J[:3]
                rhs = e[:3]
            else:
                rhs = e
            # damped least squares: dq = J^T (J J^T + l^2 I)^-1 e
            lam2 = damping ** 2
            A = J @ J.T + lam2 * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(A, rhs)
            n = np.linalg.norm(dq)
            if n > step_clamp:
                dq *= step_clamp / n
            q = clamp_to_limits(q + dq)
        T = fk(q, base_pose)
        e = pose_error(T, target_pos, target_rot)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = 0.0 if target_rot is None else float(np.linalg.norm(e[3:]))
        score = pos_err + (0.0 if target_rot is None else rot_err)
        if score < best[0]:
            best = (score, q.copy(), pos_err, rot_err, it)

    _, q, pos_err, rot_err, it = best
    ok = pos_err < pos_tol and (target_rot is None or rot_err < rot_tol)
    return q, ok, dict(pos_err=pos_err, rot_err=rot_err, iters=it)


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    print("Kinova Gen3 7-DoF (spherical wrist) -- kortex_description URDF")
    print("tool transform (bracelet -> TCP):\n", tool_transform(), "\n")

    print("--- forward kinematics ---")
    configs = {
        "zero            ": np.zeros(7),
        "home (retract)  ": np.radians([0, 15, 180, -130, 0, 55, 90]),
        "elbow bend      ": np.radians([0, 45, 0, 90, 0, 45, 0]),
        "all 30 deg      ": np.radians([30] * 7),
    }
    for name, q in configs.items():
        T = fk(q)
        print("%s pos=%s  z-axis=%s" % (name, np.round(T[:3, 3], 5),
                                        np.round(T[:3, 2], 4)))
    print("zero-config flange (end_effector_link) pos:",
          np.round(fk_flange(np.zeros(7))[:3, 3], 6))

    print("\n--- mobile base ---")
    q = np.radians([10, -20, 30, -40, 50, -60, 70])
    print("base (0,0,0)        :", np.round(fk(q, (0, 0, 0))[:3, 3], 5))
    print("base (1,2,pi/2)     :", np.round(fk(q, (1.0, 2.0, PI / 2))[:3, 3], 5))

    print("\n--- jacobian vs finite differences ---")
    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(200):
        qq = rng.uniform(-PI, PI, 7)
        bp = None if k % 2 else (rng.uniform(-2, 2), rng.uniform(-2, 2),
                                 rng.uniform(-PI, PI))
        err = float(np.max(np.abs(jacobian(qq, bp) - jacobian_fd(qq, bp))))
        worst = max(worst, err)
    print("max |J_analytic - J_fd| over 200 random configs: %.3e" % worst)
    print("JACOBIAN CHECK:", "PASS" if worst < 1e-5 else "FAIL")

    print("\n--- IK round trip ---")
    ok_count = 0
    for k in range(10):
        q_true = np.clip(rng.uniform(-1.6, 1.6, 7),
                         np.where(np.isfinite(JOINT_LIMITS_LOWER), JOINT_LIMITS_LOWER, -PI),
                         np.where(np.isfinite(JOINT_LIMITS_UPPER), JOINT_LIMITS_UPPER, PI))
        T = fk(q_true)
        q_sol, ok, info = ik(T[:3, 3], T[:3, :3], q_init=q_true + 0.15)
        ok_count += ok
        if k < 3:
            print("  target %s -> pos_err=%.2e rot_err=%.2e ok=%s"
                  % (np.round(T[:3, 3], 3), info["pos_err"], info["rot_err"], ok))
    print("IK full-pose success: %d/10" % ok_count)

    q_sol, ok, info = ik([0.45, 0.10, 0.70])
    print("position-only IK to (0.45,0.10,0.70): ok=%s err=%.2e -> %s"
          % (ok, info["pos_err"], np.round(fk(q_sol)[:3, 3], 5)))
