"""
Forward / inverse kinematics for the Kinova Gen3 7-DoF arm (+ Robotiq 2F-85).

Dependencies: numpy (and math) only.  Import-safe, no side effects.

================================ CONVENTIONS =================================

Base frame ("arm base frame" = URDF ``base_link``)
    Origin at the centre of the mounting flange of the arm base.
    +z points UP, away from the table, along the axis of joint 1.
    +x points "forward" (the direction the arm faces / away from the base
    connector panel), +y completes the right-handed frame.

Tool frame
    The chain returned by :func:`fk` ends at the *fingertip* frame:

        base_link --(7 revolute joints)--> end_effector_link (a.k.a.
        "tool_frame" / interface-module frame, 1.18736 m above the base at
        q = 0) --(translate TOOL_OFFSET_Z along +z_tool)--> fingertips.

    In the tool frame **+z points OUT of the gripper** (the approach /
    grasp-line direction), +x and +y span the gripper's "palm" plane.
    Hence a gripper pointing straight down has EE z-axis = [0, 0, -1].
    Set ``tool_z=0.0`` to get the bare interface (end_effector_link) frame.

Joint angles
    q = (q1 ... q7) in radians, all joints rotate about their own +z.
    q = 0 is the fully-extended "candlestick" pose (arm straight up).
    Positive rotation follows the right-hand rule about the joint z axis,
    matching the Kinova convention.

=========================== DENAVIT-HARTENBERG ================================

Standard (classic / distal) DH, i.e. each link transform is

    A_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)

with a pre-multiplied base rotation ``Rx(pi)`` that maps ``base_link`` into
the DH frame 0 used by Kinova's user guide (Kinova's DH frame 0 has its z
pointing DOWN).  All a_i are zero (no link offsets perpendicular to the
joint axes); the small lateral link offsets of the physical arm appear as
the d_2, d_4 terms.

    i | theta_i      |     d_i [m]                     | a_i |  alpha_i
   ---+--------------+---------------------------------+-----+----------
    1 | q1           | -(0.1564 + 0.1284)  = -0.28481  | 0   |  +pi/2
    2 | q2 + pi      | -(0.0054 + 0.0064)  = -0.01175  | 0   |  +pi/2
    3 | q3 + pi      | -(0.2104 + 0.2104)  = -0.42076  | 0   |  +pi/2
    4 | q4 + pi      | -(0.0064 + 0.0064)  = -0.01275  | 0   |  +pi/2
    5 | q5 + pi      | -(0.2084 + 0.1059)  = -0.31436  | 0   |  +pi/2
    6 | q6 + pi      |   0.0                           | 0   |  +pi/2
    7 | q7 + pi      | -(0.1059 + 0.0615)  = -0.16743  | 0   |  +pi

After A_7 the frame *is* ``end_effector_link`` (no extra correction needed).

Cross-check: the sum |d_1| + |d_3| + |d_5| + |d_7| = 1.18736 m is exactly the
published base-to-interface-module distance of the Gen3 7-DoF.

This module's primary implementation (:func:`fk`) instead walks the URDF
joint-origin chain (``kortex_description`` gen3 7dof), which is numerically
identical but keeps the true, unrounded lateral offsets.  ``_fk_dh`` evaluates
the DH table above; the ``__main__`` block asserts the two agree to < 0.5 mm.

URDF chain used (joint i: translate xyz, rotate about x by rpy_x, then Rz(q_i)):

    j1: ( 0,  0.0,        0.15643  ), rx =  pi
    j2: ( 0,  0.005375,  -0.12838  ), rx =  pi/2
    j3: ( 0, -0.21038,   -0.006375 ), rx = -pi/2
    j4: ( 0,  0.006375,  -0.21038  ), rx =  pi/2
    j5: ( 0, -0.20843,   -0.006375 ), rx = -pi/2
    j6: ( 0,  0.00017505,-0.10593  ), rx =  pi/2
    j7: ( 0, -0.10593,   -0.00017505), rx = -pi/2
    EE: ( 0,  0.0,       -0.0615   ), rx =  pi     (fixed, -> end_effector_link)

============================== JOINT LIMITS ===================================

Gen3 7-DoF: joints 1, 3, 5, 7 are CONTINUOUS (unlimited rotation); joints
2, 4, 6 are limited to about +-2.41, +-2.66, +-2.23 rad respectively.
"""

import math

import numpy as np

__all__ = [
    "TOOL_OFFSET_Z",
    "N_JOINTS",
    "JOINT_LIMITS",
    "CONTINUOUS_JOINTS",
    "RETRACT",
    "HOME",
    "ZERO",
    "fk",
    "fk_frames",
    "jacobian",
    "ik",
    "clip_to_limits",
    "wrap_angles",
]

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

N_JOINTS = 7

#: Distance from the tool/interface frame (``end_effector_link``) to the
#: fingertips of a Robotiq 2F-85 along the tool +z axis, in metres.
#: The 2F-85 coupled to the Gen3 interface module is ~0.130 m from the
#: interface plate to the fully-open fingertip contact point (0.120-0.160 m
#: depending on finger opening / pad choice).  Tune as needed.
TOOL_OFFSET_Z = 0.130

#: Base-to-interface-frame distance at q = 0 (published Gen3 7-DoF value).
ARM_LENGTH_ZERO = 1.18736

#: Per-joint (lower, upper) limits in rad.  ``None`` marks a continuous joint.
JOINT_LIMITS = [
    None,                 # joint 1 - continuous
    (-2.41, 2.41),        # joint 2
    None,                 # joint 3 - continuous
    (-2.66, 2.66),        # joint 4
    None,                 # joint 5 - continuous
    (-2.23, 2.23),        # joint 6
    None,                 # joint 7 - continuous
]
CONTINUOUS_JOINTS = tuple(i for i, lim in enumerate(JOINT_LIMITS) if lim is None)

#: Factory poses (rad).
ZERO = np.zeros(7)
RETRACT = np.array([0.0, -0.35, 3.14, -2.54, 0.0, -0.87, 1.57])
HOME = np.array([0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57])

# URDF joint origins: (xyz translation, rotation about x), then Rz(q_i).
_URDF_JOINTS = (
    ((0.0, 0.0, 0.15643), math.pi),
    ((0.0, 0.005375, -0.12838), math.pi / 2),
    ((0.0, -0.21038, -0.006375), -math.pi / 2),
    ((0.0, 0.006375, -0.21038), math.pi / 2),
    ((0.0, -0.20843, -0.006375), -math.pi / 2),
    ((0.0, 0.00017505, -0.10593), math.pi / 2),
    ((0.0, -0.10593, -0.00017505), -math.pi / 2),
)
#: Fixed transform link_7 -> end_effector_link (interface / tool frame).
_EE_FIXED_XYZ = (0.0, 0.0, -0.0615)
_EE_FIXED_RX = math.pi

# Classic DH table (see module docstring).  alpha, a, d, theta_offset.
DH_TABLE = (
    (math.pi / 2, 0.0, -0.28481, 0.0),
    (math.pi / 2, 0.0, -0.01175, math.pi),
    (math.pi / 2, 0.0, -0.42076, math.pi),
    (math.pi / 2, 0.0, -0.01275, math.pi),
    (math.pi / 2, 0.0, -0.31436, math.pi),
    (math.pi / 2, 0.0, 0.0, math.pi),
    (math.pi, 0.0, -0.16743, math.pi),
)

# ----------------------------------------------------------------------------
# Small SE(3) helpers
# ----------------------------------------------------------------------------


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[1, 1] = c
    T[1, 2] = -s
    T[2, 1] = s
    T[2, 2] = c
    return T


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def _trans(x, y, z):
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


# Pre-compute the constant part of every joint transform: Trans(xyz) @ Rx(rx).
_JOINT_PRE = tuple(_trans(*xyz) @ _rot_x(rx) for xyz, rx in _URDF_JOINTS)
_EE_FIXED = _trans(*_EE_FIXED_XYZ) @ _rot_x(_EE_FIXED_RX)


def _as_q(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != N_JOINTS:
        raise ValueError("q must have %d elements, got %d" % (N_JOINTS, q.size))
    return q


# ----------------------------------------------------------------------------
# Forward kinematics
# ----------------------------------------------------------------------------


def fk_frames(q, tool_z=None):
    """Return the list of cumulative transforms along the chain.

    The returned list has 9 entries, all expressed in ``base_link``:

        [0] base_link (identity)
        [1..7] frame of link_1 .. link_7 after applying joint 1..7
        [8] end_effector_link (interface / tool frame, z out of the gripper)
        [9] fingertip frame (tool frame translated by ``tool_z`` along +z)

    Parameters
    ----------
    q : array_like, shape (7,)
        Joint angles [rad].
    tool_z : float, optional
        Tool offset override; defaults to :data:`TOOL_OFFSET_Z`.
    """
    q = _as_q(q)
    tz = TOOL_OFFSET_Z if tool_z is None else float(tool_z)
    T = np.eye(4)
    frames = [T]
    for i in range(N_JOINTS):
        T = T @ _JOINT_PRE[i] @ _rot_z(q[i])
        frames.append(T)
    T_tool = T @ _EE_FIXED           # end_effector_link (interface frame)
    frames.append(T_tool)
    frames.append(T_tool @ _trans(0.0, 0.0, tz))   # fingertips
    return frames


def fk(q, tool_z=None):
    """Forward kinematics: joint angles -> 4x4 EE pose in the arm base frame.

    The returned frame is the *fingertip* frame: the Gen3 tool frame
    (``end_effector_link``) translated by ``tool_z`` (default
    :data:`TOOL_OFFSET_Z`) along its own +z, which points out of the gripper.
    Pass ``tool_z=0.0`` for the bare interface frame.
    """
    q = _as_q(q)
    tz = TOOL_OFFSET_Z if tool_z is None else float(tool_z)
    T = np.eye(4)
    for i in range(N_JOINTS):
        T = T @ _JOINT_PRE[i] @ _rot_z(q[i])
    T = T @ _EE_FIXED
    if tz:
        T = T @ _trans(0.0, 0.0, tz)
    return T


def _fk_dh(q, tool_z=None):
    """Same FK, evaluated from the classic DH table (cross-check / reference)."""
    q = _as_q(q)
    tz = TOOL_OFFSET_Z if tool_z is None else float(tool_z)
    T = _rot_x(math.pi)                      # base_link -> Kinova DH frame 0
    for i, (alpha, a, d, off) in enumerate(DH_TABLE):
        T = T @ _rot_z(q[i] + off) @ _trans(0.0, 0.0, d) @ _trans(a, 0.0, 0.0) @ _rot_x(alpha)
    if tz:
        T = T @ _trans(0.0, 0.0, tz)
    return T


# ----------------------------------------------------------------------------
# Jacobian
# ----------------------------------------------------------------------------


def jacobian(q, tool_z=None):
    """Analytic 6x7 geometric Jacobian of the fingertip frame, in base frame.

    Rows 0..2 map joint velocities to linear velocity of the fingertip point,
    rows 3..5 to angular velocity.  All quantities are expressed in the arm
    base frame (``base_link``).
    """
    frames = fk_frames(q, tool_z=tool_z)
    p_ee = frames[-1][:3, 3]
    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        Ti = frames[i + 1]          # frame attached to joint i AFTER rotating
        z = Ti[:3, 2]               # joint axis in base coords
        p = Ti[:3, 3]
        J[:3, i] = np.cross(z, p_ee - p)
        J[3:, i] = z
    return J


def jacobian_numeric(q, tool_z=None, eps=1e-6):
    """Finite-difference Jacobian (validation helper for :func:`jacobian`)."""
    q = _as_q(q)
    T0 = fk(q, tool_z=tool_z)
    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        dq = q.copy()
        dq[i] += eps
        T1 = fk(dq, tool_z=tool_z)
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        dR = T1[:3, :3] @ T0[:3, :3].T
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2],
                             dR[0, 2] - dR[2, 0],
                             dR[1, 0] - dR[0, 1]]) / (2.0 * eps)
    return J


# ----------------------------------------------------------------------------
# Joint helpers
# ----------------------------------------------------------------------------


def wrap_angles(q):
    """Wrap the continuous joints (1, 3, 5, 7) into [-pi, pi]."""
    q = _as_q(q).copy()
    for i in CONTINUOUS_JOINTS:
        q[i] = (q[i] + math.pi) % (2.0 * math.pi) - math.pi
    return q


def clip_to_limits(q):
    """Clip the limited joints (2, 4, 6) to their travel range; wrap the rest."""
    q = wrap_angles(q)
    for i, lim in enumerate(JOINT_LIMITS):
        if lim is not None:
            q[i] = min(max(q[i], lim[0]), lim[1])
    return q


def _rot_log(R):
    """Rotation matrix -> rotation vector (axis * angle), numerically safe."""
    cos_t = (np.trace(R) - 1.0) * 0.5
    cos_t = min(1.0, max(-1.0, cos_t))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    if theta > math.pi - 1e-6:
        # near-pi: use the symmetric part to recover the axis robustly
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 1e-12:
            axis = A[:, k] / axis[k]
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return np.zeros(3)
        return axis / n * theta
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) * (theta / (2.0 * math.sin(theta)))


# ----------------------------------------------------------------------------
# Inverse kinematics (damped least squares)
# ----------------------------------------------------------------------------


def ik(target_pos,
       target_R=None,
       q0=None,
       tool_z=None,
       orient_weight=1.0,
       max_iters=200,
       pos_tol=1e-4,
       rot_tol=1e-3,
       damping=0.05,
       step_clip=0.35,
       respect_limits=True,
       nullspace_target=None,
       nullspace_gain=0.0,
       return_info=False):
    """Damped-least-squares inverse kinematics for the fingertip frame.

    Parameters
    ----------
    target_pos : array_like, shape (3,)
        Desired fingertip position in the arm base frame [m].
    target_R : array_like, shape (3, 3), optional
        Desired fingertip orientation (tool +z points out of the gripper).
        Ignored when ``orient_weight == 0`` (position-only mode).
    q0 : array_like, shape (7,), optional
        Seed configuration; defaults to :data:`HOME`.
    tool_z : float, optional
        Tool offset override (see :data:`TOOL_OFFSET_Z`).
    orient_weight : float
        Weight on the orientation residual.  ``0`` -> position-only IK.
    max_iters, pos_tol, rot_tol, damping, step_clip :
        Solver knobs.  ``damping`` is the Levenberg-Marquardt lambda,
        ``step_clip`` the max per-joint step per iteration [rad].
    respect_limits : bool
        Clip joints 2/4/6 to their travel range every iteration.
    nullspace_target, nullspace_gain :
        Optional secondary objective pulling q toward ``nullspace_target``
        through the Jacobian null space.
    return_info : bool
        If True, return ``(q, info)`` with convergence diagnostics.

    Returns
    -------
    q : ndarray, shape (7,)
        Best configuration found (the one with the lowest error seen).
    info : dict, only if ``return_info``
        ``success``, ``pos_err``, ``rot_err`` [rad], ``iters``.
    """
    p_des = np.asarray(target_pos, dtype=float).reshape(3)
    use_orient = orient_weight > 0.0 and target_R is not None
    R_des = np.asarray(target_R, dtype=float).reshape(3, 3) if use_orient else None

    q = HOME.copy() if q0 is None else _as_q(q0).copy()
    if respect_limits:
        q = clip_to_limits(q)

    best_q = q.copy()
    best_cost = np.inf
    best_pe = np.inf
    best_re = np.inf
    iters = 0
    success = False

    for iters in range(1, max_iters + 1):
        T = fk(q, tool_z=tool_z)
        e_p = p_des - T[:3, 3]
        pe = float(np.linalg.norm(e_p))
        if use_orient:
            e_r = _rot_log(R_des @ T[:3, :3].T)
            re = float(np.linalg.norm(e_r))
        else:
            e_r = np.zeros(3)
            re = 0.0

        cost = pe + (orient_weight * re if use_orient else 0.0)
        if cost < best_cost:
            best_cost, best_q, best_pe, best_re = cost, q.copy(), pe, re

        if pe < pos_tol and (not use_orient or re < rot_tol):
            success = True
            break

        J = jacobian(q, tool_z=tool_z)
        if use_orient:
            err = np.concatenate([e_p, orient_weight * e_r])
            Ju = J.copy()
            Ju[3:, :] *= orient_weight
        else:
            err = e_p
            Ju = J[:3, :]

        # DLS: dq = J^T (J J^T + lambda^2 I)^-1 e
        m = Ju.shape[0]
        A = Ju @ Ju.T + (damping ** 2) * np.eye(m)
        dq = Ju.T @ np.linalg.solve(A, err)

        if nullspace_gain > 0.0 and nullspace_target is not None:
            qn = _as_q(nullspace_target)
            Jpinv = Ju.T @ np.linalg.inv(A)
            dq += (np.eye(N_JOINTS) - Jpinv @ Ju) @ (nullspace_gain * (qn - q))

        n = float(np.max(np.abs(dq)))
        if n > step_clip:
            dq *= step_clip / n
        q = q + dq
        if respect_limits:
            q = clip_to_limits(q)
        else:
            q = wrap_angles(q)

    if success:
        best_q, best_pe, best_re = q.copy(), pe, re

    best_q = clip_to_limits(best_q) if respect_limits else wrap_angles(best_q)
    if return_info:
        return best_q, {"success": bool(success), "pos_err": best_pe,
                        "rot_err": best_re, "iters": iters}
    return best_q


def ik_multistart(target_pos, target_R=None, seeds=None, n_random=8, rng=None, **kw):
    """Try several seeds and return the first/best converged IK solution."""
    if rng is None:
        rng = np.random.default_rng(0)
    if seeds is None:
        seeds = [HOME, RETRACT, np.zeros(7)]
    seeds = [np.asarray(s, dtype=float) for s in seeds]
    for _ in range(n_random):
        s = rng.uniform(-np.pi, np.pi, N_JOINTS)
        seeds.append(clip_to_limits(s))
    best, best_info = None, None
    for s in seeds:
        q, info = ik(target_pos, target_R, q0=s, return_info=True, **kw)
        if info["success"]:
            return (q, info) if kw.get("return_info") else q
        if best_info is None or info["pos_err"] < best_info["pos_err"]:
            best, best_info = q, info
    return (best, best_info) if kw.get("return_info") else best


# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------

def _fmt(v):
    return "[" + ", ".join("%8.5f" % x for x in v) + "]"


if __name__ == "__main__":
    import time

    np.set_printoptions(precision=5, suppress=True)

    print("=" * 74)
    print("Kinova Gen3 7-DoF forward kinematics -- sanity checks")
    print("TOOL_OFFSET_Z = %.4f m (interface frame -> 2F-85 fingertips)" % TOOL_OFFSET_Z)
    print("=" * 74)

    # --- (0) URDF chain vs. DH table -------------------------------------
    rng = np.random.default_rng(0)
    dmax = 0.0
    for _ in range(200):
        qr = rng.uniform(-np.pi, np.pi, 7)
        dmax = max(dmax, float(np.max(np.abs(fk(qr) - _fk_dh(qr)))))
    print("\n[check] max |fk(URDF chain) - fk(DH table)| over 200 random q : %.2e m"
          % dmax)
    assert dmax < 5e-4, "URDF chain and DH table disagree"

    # --- (0b) analytic vs numeric Jacobian --------------------------------
    jmax = 0.0
    for _ in range(50):
        qr = rng.uniform(-np.pi, np.pi, 7)
        jmax = max(jmax, float(np.max(np.abs(jacobian(qr) - jacobian_numeric(qr)))))
    print("[check] max |J_analytic - J_numeric| over 50 random q        : %.2e" % jmax)
    assert jmax < 1e-4, "Jacobian mismatch"

    # --- (a)(b)(c) named poses -------------------------------------------
    print("\n%-9s %-26s %-26s %s" % ("pose", "fingertip pos [m]", "EE z-axis (approach)", "|p|"))
    print("-" * 84)
    for name, qn in (("ZERO", ZERO), ("RETRACT", RETRACT), ("HOME", HOME)):
        T = fk(qn)
        Ti = fk(qn, tool_z=0.0)
        print("%-9s %-26s %-26s %6.4f" % (name, _fmt(T[:3, 3]), _fmt(T[:3, 2]),
                                          np.linalg.norm(T[:3, 3])))
        print("%-9s %-26s %-26s %6.4f   (interface frame, tool_z=0)"
              % ("", _fmt(Ti[:3, 3]), _fmt(Ti[:3, 2]), np.linalg.norm(Ti[:3, 3])))
    print("-" * 84)
    print("ZERO pose: arm straight up along +z.  Reach to interface frame = %.5f m,"
          % np.linalg.norm(fk(ZERO, tool_z=0.0)[:3, 3]))
    print("           reach to fingertips = %.5f m (published Gen3 value 1.18736 m"
          % np.linalg.norm(fk(ZERO)[:3, 3]))
    print("           base->interface, + the 2F-85 tool offset).")

    # --- (d) gripper straight down at (0.45, 0, -0.15) --------------------
    # Tool +z must point along -z_base.  Take x_tool = +x_base (gripper's
    # "x" forward), then y_tool = z_tool x x_tool = -y_base.
    R_down = np.array([[1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]])
    target = np.array([0.45, 0.0, -0.15])

    t0 = time.perf_counter()
    q_sol, info = ik(target, R_down, q0=HOME, return_info=True)
    dt = (time.perf_counter() - t0) * 1e3

    T = fk(q_sol)
    print("\n[IK] gripper straight DOWN, fingertips at (0.45, 0.00, -0.15):")
    print("     converged = %s   iters = %d   time = %.2f ms" % (info["success"], info["iters"], dt))
    print("     q       = %s" % _fmt(q_sol))
    print("     pos     = %s   (err %.2e m)" % (_fmt(T[:3, 3]), info["pos_err"]))
    print("     z-axis  = %s   (err %.2e rad)" % (_fmt(T[:3, 2]), info["rot_err"]))
    print("     within limits: %s" % np.allclose(q_sol, clip_to_limits(q_sol), atol=1e-9))
    assert info["success"], "IK failed on the straight-down target"
    assert np.allclose(T[:3, 2], [0, 0, -1], atol=1e-3)

    # --- (e) table of downward poses along +x -----------------------------
    print("\n[IK] downward-pointing configurations, fingertips at (r, 0, -0.15):")
    hdr = "%5s  %-58s %7s %8s %6s %5s" % ("r [m]", "q1..q7 [rad]", "pos_err", "rot_err",
                                          "t[ms]", "ok")
    print(hdr)
    print("-" * len(hdr))
    q_seed = HOME.copy()
    for r in (0.3, 0.4, 0.5, 0.6):
        tgt = np.array([r, 0.0, -0.15])
        t0 = time.perf_counter()
        qs, inf = ik(tgt, R_down, q0=q_seed, return_info=True)
        dt = (time.perf_counter() - t0) * 1e3
        print("%5.2f  %-58s %7.1e %8.1e %6.2f %5s"
              % (r, ", ".join("%6.3f" % x for x in qs),
                 inf["pos_err"], inf["rot_err"], dt, inf["success"]))
        if inf["success"]:
            q_seed = qs          # warm-start the next target

    # --- (f) position-only IK ---------------------------------------------
    t0 = time.perf_counter()
    qp, infp = ik(np.array([0.5, 0.2, 0.30]), None, q0=HOME,
                  orient_weight=0.0, return_info=True)
    dt = (time.perf_counter() - t0) * 1e3
    print("\n[IK] position-only (orient_weight=0) target (0.50, 0.20, 0.30):")
    print("     converged = %s  iters = %d  pos_err = %.2e m  time = %.2f ms"
          % (infp["success"], infp["iters"], infp["pos_err"], dt))
    print("     pos = %s" % _fmt(fk(qp)[:3, 3]))

    # --- (g) IK round-trip on random reachable poses ----------------------
    ok, tot, tsum = 0, 0, 0.0
    for _ in range(50):
        qr = clip_to_limits(rng.uniform(-np.pi, np.pi, 7))
        Tr = fk(qr)
        t0 = time.perf_counter()
        _, i2 = ik(Tr[:3, 3], Tr[:3, :3], q0=clip_to_limits(qr + rng.normal(0, 0.25, 7)),
                   return_info=True)
        tsum += (time.perf_counter() - t0) * 1e3
        tot += 1
        ok += int(i2["success"])
    print("\n[IK] round-trip from perturbed seeds: %d/%d converged, %.2f ms avg"
          % (ok, tot, tsum / tot))
    print("=" * 74)
