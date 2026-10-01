"""
Forward / inverse kinematics for a Kinova Gen3 7-DoF arm mounted on a
TidyBot-style holonomic mobile base (tidybot2 / Stanford "TidyBot++").

Pure numpy (scipy optional, not required).

--------------------------------------------------------------------------
KINEMATIC CONVENTION
--------------------------------------------------------------------------
We do NOT use a DH table.  We use the *URDF link chain* published by Kinova
in `ros_kortex` (`kortex_description/arms/gen3/7dof/urdf/GEN3_URDF_V12.urdf`)
which is the same chain shipped in `mujoco_menagerie/kinova_gen3/gen3.xml`.

Each joint i is expressed as a fixed transform (translation + fixed RPY,
extrinsic X-Y-Z / roll-pitch-yaw) from the parent link frame, followed by a
rotation of q_i about the child frame's local +Z axis (every Gen3 joint axis
is 0 0 1 in its own frame).

    T_parent_child(q_i) = Trans(xyz_i) * RPY(rpy_i) * Rz(q_i)

Chain (all axes 0 0 1):

  base_link        -> shoulder_link        xyz (0,  0.0,        0.15643)   rpy (-pi, 0, 0)
  shoulder_link    -> half_arm_1_link      xyz (0,  0.005375,  -0.12838)   rpy ( pi/2, 0, 0)
  half_arm_1_link  -> half_arm_2_link      xyz (0, -0.21038,   -0.006375)  rpy (-pi/2, 0, 0)
  half_arm_2_link  -> forearm_link         xyz (0,  0.006375,  -0.21038)   rpy ( pi/2, 0, 0)
  forearm_link     -> spherical_wrist_1    xyz (0, -0.20843,   -0.006375)  rpy (-pi/2, 0, 0)
  sph_wrist_1_link -> spherical_wrist_2    xyz (0,  0.00017505,-0.10593)   rpy ( pi/2, 0, 0)
  sph_wrist_2_link -> bracelet_link        xyz (0, -0.10593,   -0.00017505)rpy (-pi/2, 0, 0)
  bracelet_link    -> end_effector_link    xyz (0,  0,         -0.0615250) rpy ( pi, 0, 0)   [fixed]

Sanity check against Kinova's published DH numbers (user guide, Table
"Denavit-Hartenberg parameters", *modified* DH):
    d1 = 0.15643 + 0.12838 = 0.28481   (guide: 0.2848)  OK
    d2 = 0.005375 + 0.006375 = 0.01175 (guide: 0.0118)  OK
    d3 = 0.21038 + 0.21038 = 0.42076   (guide: 0.4208)  OK
    d4 = 0.006375 + 0.006375 = 0.01275 (guide: 0.0128)  OK
    d5 = 0.20843 + 0.10593 = 0.31436   (guide: 0.3143)  OK
    d6 = 0.00017505 + 0.00017505 ~ 0   (guide: 0.0)     OK
    d7 = 0.10593 + 0.061525 = 0.167455 (guide: 0.1674)  OK  (interface plate)
So the chain below reproduces the official Gen3 DH table to <0.1 mm.
CONFIDENCE: HIGH for the arm chain itself (these constants are the
"no-vision-module" GEN3 7-DoF values and match the DH table exactly).

The `end_effector_link` frame is the *tool interface plate* frame: +Z points
out of the plate (away from the arm), which is why the final fixed transform
carries an rpy of (pi, 0, 0) relative to `bracelet_link`.

--------------------------------------------------------------------------
GRIPPER / TIP
--------------------------------------------------------------------------
TOOL_OFFSET_Z is the distance from `end_effector_link` (tool interface plate)
along its +Z to the tool centre point (TCP) between the Robotiq 2F-85
fingertips.  Kinova's own "Gen3 with Robotiq 2F-85" tool frame is documented
at 0.1200 m; `mujoco_menagerie/robotiq_2f85/2f85.xml` places its `pinch`
site at z = 0.145 m in the gripper base body, and the 2F-85 base is mounted
on `end_effector_link` with zero translation.  Since tidybot2 uses the
menagerie assets, the default here is 0.145 m.
CONFIDENCE: MEDIUM (+-15 mm; true value is somewhere in 0.130-0.150 m and
depends on the finger opening, since the 2F-85 fingers curl inward as they
close).  Recalibrate TOOL_OFFSET_Z against the sim if millimetres matter.

--------------------------------------------------------------------------
BASE MOUNT
--------------------------------------------------------------------------
ARM_MOUNT_XYZ / ARM_MOUNT_YAW place `base_link` of the arm in the mobile
base frame (base frame = ground-projected centre of the holonomic base, +X
forward, +Z up).  TidyBot++ mounts the Gen3 on a riser on top of the
powered-caster base.  Best guess: the arm sits on the deck ~0.39 m above the floor, at or
slightly forward of the base centre (values in the literature/XML are in the
range x = 0.00-0.15 m, z = 0.38-0.43 m, yaw = 0).
CONFIDENCE: LOW.  These are the numbers most likely to be wrong; they are
module-level constants precisely so they can be recalibrated by comparing
`fk()` against the simulator's reported end-effector pose.
--------------------------------------------------------------------------
"""

import numpy as np

__all__ = [
    "fk", "fk_all_frames", "jacobian", "ik",
    "ARM_MOUNT_XYZ", "ARM_MOUNT_YAW", "TOOL_OFFSET_Z",
    "JOINT_LIMITS_LOWER", "JOINT_LIMITS_UPPER", "JOINT_CONTINUOUS",
    "HOME_Q", "RETRACT_Q",
]

# ---------------------------------------------------------------------------
# RECALIBRATABLE MOUNT CONSTANTS
# ---------------------------------------------------------------------------
# Position of the Kinova Gen3 `base_link` origin expressed in the mobile-base
# frame (x forward, y left, z up, origin on the floor at the base centre).
# NOTE: the x offset may well be ~0.10-0.15 m (arm mounted forward of centre).
ARM_MOUNT_XYZ = np.array([0.0, 0.0, 0.39], dtype=float)
# Yaw of the arm base about the mobile base +Z (rad).  0 => arm +X == base +X.
ARM_MOUNT_YAW = 0.0
# Extra roll/pitch of the mount (rad).  Normally zero (arm bolted flat).
ARM_MOUNT_ROLL = 0.0
ARM_MOUNT_PITCH = 0.0
# end_effector_link -> gripper TCP, along tool +Z (m).
TOOL_OFFSET_Z = 0.145
# Lateral TCP offsets (m), normally zero for the 2F-85 (symmetric fingers).
TOOL_OFFSET_X = 0.0
TOOL_OFFSET_Y = 0.0

# ---------------------------------------------------------------------------
# GEN3 7-DoF LINK CHAIN  (xyz, rpy) of each revolute joint frame in its parent
# ---------------------------------------------------------------------------
_PI = np.pi
_LINKS = [
    # (x, y, z, roll, pitch, yaw)
    (0.0, 0.0, 0.15643, -_PI, 0.0, 0.0),            # J1  base -> shoulder
    (0.0, 0.005375, -0.12838, _PI / 2, 0.0, 0.0),   # J2  shoulder -> half_arm_1
    (0.0, -0.21038, -0.006375, -_PI / 2, 0.0, 0.0),  # J3  half_arm_1 -> half_arm_2
    (0.0, 0.006375, -0.21038, _PI / 2, 0.0, 0.0),   # J4  half_arm_2 -> forearm
    (0.0, -0.20843, -0.006375, -_PI / 2, 0.0, 0.0),  # J5  forearm -> wrist_1
    (0.0, 0.00017505, -0.10593, _PI / 2, 0.0, 0.0),  # J6  wrist_1 -> wrist_2
    (0.0, -0.10593, -0.00017505, -_PI / 2, 0.0, 0.0),  # J7  wrist_2 -> bracelet
]
# fixed: bracelet_link -> end_effector_link (tool interface plate)
_EE_FIXED = (0.0, 0.0, -0.061525, _PI, 0.0, 0.0)

N_JOINTS = 7

# ---------------------------------------------------------------------------
# JOINT LIMITS (rad).  Gen3 7-DoF: joints 1,3,5,7 are continuous (unlimited).
# ---------------------------------------------------------------------------
JOINT_CONTINUOUS = np.array([True, False, True, False, True, False, True])
_BIG = 1e9
JOINT_LIMITS_LOWER = np.array([-_BIG, -2.41, -_BIG, -2.66, -_BIG, -2.23, -_BIG])
JOINT_LIMITS_UPPER = np.array([+_BIG, +2.41, +_BIG, +2.66, +_BIG, +2.23, +_BIG])

HOME_Q = np.array([0.0, -0.349, 3.142, -2.548, 0.0, -0.873, 1.571])
# Kinova factory "Home" (0, 15, 180, -130, 0, 55, 90 deg) for reference:
RETRACT_Q = np.array([0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708])


# ---------------------------------------------------------------------------
# Small SE(3) helpers
# ---------------------------------------------------------------------------
def _rpy(roll, pitch, yaw):
    """Extrinsic X-Y-Z (URDF / MuJoCo `euler` default) rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _T(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _fixed(link):
    x, y, z, r, p, yw = link
    return _T(_rpy(r, p, yw), np.array([x, y, z]))


_LINK_T = [_fixed(l) for l in _LINKS]
_EE_T = _fixed(_EE_FIXED)
_TOOL_T = _T(np.eye(3), np.array([TOOL_OFFSET_X, TOOL_OFFSET_Y, TOOL_OFFSET_Z]))


def _tool_T():
    """Rebuilt each call so runtime edits to TOOL_OFFSET_* take effect."""
    return _T(np.eye(3), np.array([TOOL_OFFSET_X, TOOL_OFFSET_Y, TOOL_OFFSET_Z]))


def base_T(base_x=0.0, base_y=0.0, base_yaw=0.0):
    """World <- mobile-base frame."""
    return _T(_rotz(base_yaw), np.array([base_x, base_y, 0.0]))


def arm_mount_T():
    """Mobile-base frame <- Gen3 base_link frame."""
    return _T(_rpy(ARM_MOUNT_ROLL, ARM_MOUNT_PITCH, ARM_MOUNT_YAW),
              np.asarray(ARM_MOUNT_XYZ, dtype=float))


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------
def fk_all_frames(joints, base_x=0.0, base_y=0.0, base_yaw=0.0):
    """Return the list of 4x4 world transforms:
    [base, arm_base, J1..J7 (link frames), end_effector_link, tip].
    """
    q = np.asarray(joints, dtype=float).ravel()
    if q.size != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joint angles, got {q.size}")

    frames = []
    T = base_T(base_x, base_y, base_yaw)
    frames.append(T.copy())
    T = T @ arm_mount_T()
    frames.append(T.copy())
    for i in range(N_JOINTS):
        T = T @ _LINK_T[i] @ _T(_rotz(q[i]), np.zeros(3))
        frames.append(T.copy())
    T = T @ _EE_T
    frames.append(T.copy())
    T = T @ _tool_T()
    frames.append(T.copy())
    return frames


def fk(joints, base_x=0.0, base_y=0.0, base_yaw=0.0):
    """Forward kinematics of the gripper TIP (TCP between the fingers).

    Parameters
    ----------
    joints : (7,) array of Gen3 joint angles (rad)
    base_x, base_y, base_yaw : holonomic mobile-base pose in the world

    Returns
    -------
    (pos, R) : pos is (3,) world XYZ of the TCP,
               R is the (3,3) world rotation matrix of the tool frame
               (tool +Z points out of the gripper, along the approach axis).
    """
    T = fk_all_frames(joints, base_x, base_y, base_yaw)[-1]
    return T[:3, 3].copy(), T[:3, :3].copy()


def fk_pose(joints, base_x=0.0, base_y=0.0, base_yaw=0.0):
    """Same as `fk` but returns the full 4x4 world transform."""
    return fk_all_frames(joints, base_x, base_y, base_yaw)[-1]


def fk_flange(joints, base_x=0.0, base_y=0.0, base_yaw=0.0):
    """Pose of `end_effector_link` (tool interface plate), no tool offset."""
    T = fk_all_frames(joints, base_x, base_y, base_yaw)[-2]
    return T[:3, 3].copy(), T[:3, :3].copy()


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------
def jacobian(joints, base_x=0.0, base_y=0.0, base_yaw=0.0, eps=1e-6,
             analytic=True):
    """6x7 geometric Jacobian of the TIP in WORLD coordinates.

    Rows 0:3 -> linear velocity, rows 3:6 -> angular velocity, such that
    [v; w] = J @ qdot.

    `analytic=True` uses the exact revolute-joint formula (z_i x (p_e - p_i));
    `analytic=False` uses central finite differences (rotation error taken as
    the log of R_perturbed @ R_0^T).
    """
    q = np.asarray(joints, dtype=float).ravel()
    if analytic:
        frames = fk_all_frames(q, base_x, base_y, base_yaw)
        p_e = frames[-1][:3, 3]
        J = np.zeros((6, N_JOINTS))
        for i in range(N_JOINTS):
            Ti = frames[2 + i]          # world frame of joint i's link
            z = Ti[:3, 2]               # joint axis is local +Z
            p = Ti[:3, 3]
            J[:3, i] = np.cross(z, p_e - p)
            J[3:, i] = z
        return J

    J = np.zeros((6, N_JOINTS))
    p0, R0 = fk(q, base_x, base_y, base_yaw)
    for i in range(N_JOINTS):
        dq = np.zeros(N_JOINTS)
        dq[i] = eps
        pp, Rp = fk(q + dq, base_x, base_y, base_yaw)
        pm, Rm = fk(q - dq, base_x, base_y, base_yaw)
        J[:3, i] = (pp - pm) / (2 * eps)
        J[3:, i] = _so3_log(Rp @ Rm.T) / (2 * eps)
    return J


def _so3_log(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if abs(np.pi - th) < 1e-6:
        # near pi: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        # fix signs from the off-diagonals
        k = int(np.argmax(axis))
        if axis[k] > 1e-9:
            for j in range(3):
                if j != k:
                    axis[j] = A[k, j] / axis[k]
        n = np.linalg.norm(axis)
        if n > 1e-9:
            axis = axis / n
        return axis * th
    return (th / (2.0 * np.sin(th))) * np.array([R[2, 1] - R[1, 2],
                                                 R[0, 2] - R[2, 0],
                                                 R[1, 0] - R[0, 1]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def clamp_joints(q):
    """Wrap continuous joints to (-pi, pi] and clamp the limited ones."""
    q = np.asarray(q, dtype=float).copy()
    q[JOINT_CONTINUOUS] = _wrap(q[JOINT_CONTINUOUS])
    lim = ~JOINT_CONTINUOUS
    q[lim] = np.clip(q[lim], JOINT_LIMITS_LOWER[lim], JOINT_LIMITS_UPPER[lim])
    return q


def pose_error(q, target_pos, target_rot, base_x, base_y, base_yaw,
               rot_weight=1.0):
    """6-vector [dp; dw] taking the current TIP pose to the target."""
    p, R = fk(q, base_x, base_y, base_yaw)
    e = np.zeros(6)
    e[:3] = np.asarray(target_pos, dtype=float) - p
    if target_rot is not None:
        e[3:] = rot_weight * _so3_log(np.asarray(target_rot, dtype=float) @ R.T)
    return e


# ---------------------------------------------------------------------------
# Damped least-squares IK
# ---------------------------------------------------------------------------
def ik(target_pos, target_rot_or_None=None, q_init=None, base_pose=(0.0, 0.0, 0.0),
       max_iters=300, pos_tol=1e-4, rot_tol=1e-3, damping=0.05,
       step_clip=0.35, rot_weight=1.0, restarts=6, seed=0,
       return_info=False):
    """Damped-least-squares (Levenberg-Marquardt) inverse kinematics.

    Parameters
    ----------
    target_pos        : (3,) desired world TCP position
    target_rot_or_None: (3,3) desired world tool rotation, or None for
                        position-only IK
    q_init            : (7,) seed configuration (defaults to HOME_Q)
    base_pose         : (base_x, base_y, base_yaw) of the mobile base
    restarts          : number of randomized restarts if the first solve
                        fails to converge

    Returns
    -------
    q  : (7,) joint angles (joint limits respected)
    or (q, info) if return_info=True, where info has keys
       'success', 'pos_err', 'rot_err', 'iters'.
    """
    bx, by, byaw = base_pose
    target_pos = np.asarray(target_pos, dtype=float).ravel()
    R_t = None if target_rot_or_None is None else np.asarray(
        target_rot_or_None, dtype=float)

    if q_init is None:
        q_init = HOME_Q.copy()
    q_seed = clamp_joints(q_init)

    rng = np.random.default_rng(seed)
    best_q, best_cost = q_seed.copy(), np.inf
    best_info = {"success": False, "pos_err": np.inf, "rot_err": np.inf,
                 "iters": 0}

    n_err = 6 if R_t is not None else 3

    for attempt in range(max(1, restarts)):
        if attempt == 0:
            q = q_seed.copy()
        else:
            q = clamp_joints(q_seed + rng.normal(0.0, 0.6 * attempt,
                                                 size=N_JOINTS))
        lam = damping
        it = 0
        for it in range(1, max_iters + 1):
            e = pose_error(q, target_pos, R_t, bx, by, byaw, rot_weight)[:n_err]
            pos_err = float(np.linalg.norm(e[:3]))
            rot_err = (float(np.linalg.norm(e[3:6])) / max(rot_weight, 1e-9)
                       if R_t is not None else 0.0)
            if pos_err < pos_tol and rot_err < rot_tol:
                break
            J = jacobian(q, bx, by, byaw)[:n_err]
            # DLS: dq = J^T (J J^T + lam^2 I)^-1 e
            JJt = J @ J.T + (lam ** 2) * np.eye(n_err)
            try:
                dq = J.T @ np.linalg.solve(JJt, e)
            except np.linalg.LinAlgError:
                dq = J.T @ np.linalg.pinv(JJt) @ e
            # nullspace pull toward the seed (keeps the elbow sane)
            if n_err == 3 or True:
                Jp = np.linalg.pinv(J, rcond=1e-4)
                N = np.eye(N_JOINTS) - Jp @ J
                dq = dq + 0.02 * (N @ _null_bias(q, q_seed))
            nrm = np.linalg.norm(dq)
            if nrm > step_clip:
                dq *= step_clip / nrm
            q_new = clamp_joints(q + dq)
            e_new = pose_error(q_new, target_pos, R_t, bx, by, byaw,
                               rot_weight)[:n_err]
            if np.linalg.norm(e_new) < np.linalg.norm(e):
                q = q_new
                lam = max(lam * 0.7, 1e-4)
            else:
                lam = min(lam * 2.0, 10.0)
                q = q_new  # still accept; DLS with bigger damping next round

        e = pose_error(q, target_pos, R_t, bx, by, byaw, rot_weight)[:n_err]
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = (float(np.linalg.norm(e[3:6])) / max(rot_weight, 1e-9)
                   if R_t is not None else 0.0)
        cost = pos_err + (rot_err * 0.1 if R_t is not None else 0.0)
        ok = pos_err < pos_tol and rot_err < rot_tol
        if cost < best_cost:
            best_cost = cost
            best_q = q.copy()
            best_info = {"success": bool(ok), "pos_err": pos_err,
                         "rot_err": rot_err, "iters": it}
        if ok:
            break

    if return_info:
        return best_q, best_info
    return best_q


def _null_bias(q, q_seed):
    """Secondary task: stay near the seed / away from joint limits."""
    b = np.zeros(N_JOINTS)
    d = _wrap(q_seed - q)
    b[JOINT_CONTINUOUS] = d[JOINT_CONTINUOUS]
    lim = ~JOINT_CONTINUOUS
    mid = 0.5 * (JOINT_LIMITS_LOWER[lim] + JOINT_LIMITS_UPPER[lim])
    b[lim] = (q_seed[lim] - q[lim]) + 0.5 * (mid - q[lim])
    return b


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------
def _fmt(v):
    return "[" + ", ".join(f"{x: .5f}" for x in np.asarray(v).ravel()) + "]"


def _main():
    np.set_printoptions(precision=5, suppress=True)

    print("=" * 74)
    print("Kinova Gen3 7-DoF on TidyBot-style base -- FK self test")
    print("=" * 74)
    print(f"ARM_MOUNT_XYZ  = {_fmt(ARM_MOUNT_XYZ)}   ARM_MOUNT_YAW = {ARM_MOUNT_YAW}")
    print(f"TOOL_OFFSET_Z  = {TOOL_OFFSET_Z}")
    print()

    for name, q in (("HOME  ", HOME_Q), ("ZEROS ", np.zeros(7))):
        p, R = fk(q)
        pf, _ = fk_flange(q)
        print(f"{name} q = {_fmt(q)}")
        print(f"        tip pos (world, base at origin) = {_fmt(p)}")
        print(f"        flange  (end_effector_link)     = {_fmt(pf)}")
        print(f"        tip rot =\n{R}")
        # arm-local (relative to Gen3 base_link)
        pl = p - ARM_MOUNT_XYZ
        print(f"        tip in ARM base_link frame      = {_fmt(pl)}")
        print()

    # --- DH cross-check -----------------------------------------------------
    print("-" * 74)
    print("Cross-check vs Kinova published DH d-values:")
    frames = fk_all_frames(np.zeros(7))
    exp = dict(d1=0.2848, d3=0.4208, d5=0.3143, d7=0.1674)
    got_d1 = _LINKS[0][2] - _LINKS[1][2]
    got_d3 = -_LINKS[2][1] - _LINKS[3][2]
    got_d5 = -_LINKS[4][1] - _LINKS[5][2]
    got_d7 = -_LINKS[6][1] - _EE_FIXED[2]
    for k, g in (("d1", got_d1), ("d3", got_d3), ("d5", got_d5), ("d7", got_d7)):
        print(f"   {k}: chain={g:.5f}  kinova={exp[k]:.4f}  diff={abs(g-exp[k])*1000:.2f} mm")
    print()

    # --- base transform check ----------------------------------------------
    print("-" * 74)
    bx, by, byaw = 1.5, -0.7, 0.9
    p0, R0 = fk(HOME_Q)
    p1, R1 = fk(HOME_Q, bx, by, byaw)
    Rz = _rotz(byaw)
    p_exp = Rz @ p0 + np.array([bx, by, 0.0])
    print(f"base pose ({bx}, {by}, {byaw}) -> tip {_fmt(p1)}")
    print(f"   expected from rigid transform  {_fmt(p_exp)}  "
          f"err={np.linalg.norm(p1-p_exp):.2e}")
    print(f"   rotation consistent: {np.allclose(R1, Rz @ R0, atol=1e-9)}")
    print()

    # --- Jacobian check -----------------------------------------------------
    print("-" * 74)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        q = clamp_joints(rng.uniform(-2.0, 2.0, 7))
        Ja = jacobian(q, analytic=True)
        Jn = jacobian(q, analytic=False)
        worst = max(worst, float(np.abs(Ja - Jn).max()))
    print(f"analytic vs finite-difference Jacobian, max abs diff = {worst:.3e}")
    print()

    # --- IK round trip ------------------------------------------------------
    print("-" * 74)
    print("IK round-trip test  (ik(fk(q)) must reproduce the pose)")
    rng = np.random.default_rng(0)
    n_ok = 0
    n_trials = 25
    worst_p, worst_r = 0.0, 0.0
    for t in range(n_trials):
        q_true = clamp_joints(np.concatenate([
            rng.uniform(-np.pi, np.pi, 1),
            rng.uniform(-2.2, 2.2, 1),
            rng.uniform(-np.pi, np.pi, 1),
            rng.uniform(-2.4, 2.4, 1),
            rng.uniform(-np.pi, np.pi, 1),
            rng.uniform(-2.0, 2.0, 1),
            rng.uniform(-np.pi, np.pi, 1),
        ]))
        base = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi))
        p_t, R_t = fk(q_true, *base)
        q_sol, info = ik(p_t, R_t, q_init=HOME_Q, base_pose=base,
                         return_info=True)
        p_s, R_s = fk(q_sol, *base)
        ep = np.linalg.norm(p_s - p_t)
        er = np.linalg.norm(_so3_log(R_t @ R_s.T))
        worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
        if ep < 1e-3 and er < 1e-2:
            n_ok += 1
        else:
            print(f"   trial {t:2d} FAILED  pos_err={ep:.2e}  rot_err={er:.2e}")
    print(f"   full 6-DoF pose IK: {n_ok}/{n_trials} converged "
          f"(worst pos {worst_p:.2e} m, worst rot {worst_r:.2e} rad)")

    n_ok = 0
    worst_p = 0.0
    for t in range(n_trials):
        q_true = clamp_joints(rng.uniform(-2.0, 2.0, 7))
        p_t, _ = fk(q_true)
        q_sol = ik(p_t, None, q_init=HOME_Q)
        p_s, _ = fk(q_sol)
        ep = np.linalg.norm(p_s - p_t)
        worst_p = max(worst_p, ep)
        n_ok += ep < 1e-3
    print(f"   position-only IK  : {n_ok}/{n_trials} converged "
          f"(worst pos {worst_p:.2e} m)")
    print()

    # --- joint limits respected --------------------------------------------
    q_bad = np.array([10.0, 5.0, -9.0, -5.0, 8.0, 4.0, -7.0])
    qc = clamp_joints(q_bad)
    lim = ~JOINT_CONTINUOUS
    print("-" * 74)
    print(f"clamp_joints({_fmt(q_bad)})\n           = {_fmt(qc)}")
    print(f"   limited joints within bounds: "
          f"{bool(np.all(qc[lim] >= JOINT_LIMITS_LOWER[lim] - 1e-12) and np.all(qc[lim] <= JOINT_LIMITS_UPPER[lim] + 1e-12))}")
    print("=" * 74)


if __name__ == "__main__":
    _main()
