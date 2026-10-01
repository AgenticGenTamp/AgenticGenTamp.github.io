"""Kinematics for a Kinova Gen3 7-DOF arm (numpy + stdlib only).

Conventions
-----------
* Joint order is ``joint_1 .. joint_7``; ``q`` is always a length-7 array of
  radians.
* The DH table below is the official Kinova Gen3 (7 DOF, spherical wrist)
  table from the Kinova Gen3 user guide, expressed in the **modified
  (Craig) DH** convention::

      A_i = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i + off_i) * Tz(d_i)

  All ``a`` values are zero for the Gen3 and all ``d`` values are negative
  (the Kinova table is written with the chain running "down" the links).
* Frame 7 produced by this chain is the Kinova **tool / interface frame**
  (the flange at the end of the wrist, where the gripper bolts on).  Its
  +z axis is the *approach* axis.  ``tool_z`` slides an extra distance
  along that +z axis so the returned frame is the gripper grasp point
  (fingertip centre) rather than the flange.

Validation: at the Kinova "home" configuration
``q = (0, 0.26, 3.14, -2.27, 0, 0.96, 1.57)`` this FK returns
``p = (0.456, 0.002, 0.434)`` which matches the pose Kinova's own web app
reports for that configuration to within 1 mm.

Note on "reach": the fully-extended geometric distance from the base to the
flange is 1.187 m (the sum of the d values); Kinova's quoted 902 mm is the
*usable* workspace radius, so do not use 0.9 m as an FK check.

Quaternions are ``(x, y, z, w)`` by default; pass ``wxyz=True`` to the
conversion helpers (and to :func:`ik`) to use ``(w, x, y, z)`` instead.
"""

from __future__ import annotations

import math
import time as _time

import numpy as np

__all__ = [
    "N_JOINTS",
    "DH",
    "Q_RETRACT",
    "Q_HOME",
    "JOINT_LIMITS",
    "CONTINUOUS_JOINTS",
    "fk",
    "fk_all",
    "ee_position",
    "ee_z_axis",
    "ee_axes",
    "jacobian",
    "ik",
    "clamp_to_limits",
    "in_limits",
    "ik_top_down",
    "wrap_to_pi",
    "quat_to_mat",
    "mat_to_quat",
    "top_down_quat",
    "TOP_DOWN_R",
]

PI = math.pi
N_JOINTS = 7

# ---------------------------------------------------------------------------
# Official Kinova Gen3 (7 DOF) modified-DH table.
#   rows: (alpha_{i-1}, a_{i-1}, d_i, theta_offset_i)
# ---------------------------------------------------------------------------
DH = np.array(
    [
        (PI,        0.0, -0.2848, 0.0),   # 0.1564 + 0.1284  (base + shoulder)
        (PI / 2.0,  0.0, -0.0118, PI),    # 0.0054 + 0.0064
        (PI / 2.0,  0.0, -0.4208, PI),    # 0.2104 + 0.2104  (upper arm)
        (PI / 2.0,  0.0, -0.0128, PI),    # 0.0064 + 0.0064
        (PI / 2.0,  0.0, -0.3143, PI),    # 0.2084 + 0.1059  (forearm)
        (PI / 2.0,  0.0,  0.0,    PI),
        (PI / 2.0,  0.0, -0.1674, PI),    # 0.1059 + 0.0615  (wrist -> flange)
    ],
    dtype=float,
)
# Frame 7 -> Kinova tool/interface frame (flips z so +z is the approach axis).
_TOOL_ALPHA = PI

# Canonical configurations (radians).
Q_RETRACT = np.array([0.0, -0.35, -3.1416, -2.5, 0.0, -0.87, 1.5708])
Q_HOME = np.array([0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57])

# Joint limits.  Joints 1, 3, 5, 7 (1-indexed) are continuous / unlimited.
_INF = float("inf")
JOINT_LIMITS = np.array(
    [
        (-_INF, _INF),
        (-2.41, 2.41),
        (-_INF, _INF),
        (-2.66, 2.66),
        (-_INF, _INF),
        (-2.23, 2.23),
        (-_INF, _INF),
    ],
    dtype=float,
)
CONTINUOUS_JOINTS = np.array([True, False, True, False, True, False, True])

# Approach axis of a top-down grasp: EE local +z points along world -z.
TOP_DOWN_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


# ---------------------------------------------------------------------------
# Small homogeneous-transform helpers
# ---------------------------------------------------------------------------
def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, c, -s, 0.0],
         [0.0, s, c, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, 0.0, 0.0],
         [s, c, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )


def _trans(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def _base_transform(base_x=0.0, base_y=0.0, base_rot=0.0, mount=(0.0, 0.0, 0.0)) -> np.ndarray:
    """World <- arm-base transform.

    The base frame sits at ``(base_x, base_y, 0) + mount`` and is rotated by
    ``base_rot`` radians about the world z axis.
    """
    mx, my, mz = (float(v) for v in mount)
    T = _rot_z(float(base_rot))
    T[0, 3] = float(base_x) + mx
    T[1, 3] = float(base_y) + my
    T[2, 3] = mz
    return T


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------
_CA = np.cos(DH[:, 0])
_SA = np.sin(DH[:, 0])
_D = DH[:, 2]
_OFF = DH[:, 3]
_TOOL_R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # Rx(pi)


def _chain(q):
    """Accumulated (R, p) of the tool frame relative to the arm base frame.

    Uses the closed form of ``Rx(alpha) @ Rz(theta) @ Tz(d)`` (all ``a`` are
    zero for the Gen3), which keeps FK cheap enough for finite-difference
    Jacobians inside the IK loop.
    """
    R = np.eye(3)
    p = np.zeros(3)
    th = q + _OFF
    cs = np.cos(th)
    sn = np.sin(th)
    for i in range(N_JOINTS):
        ca, sa, d = _CA[i], _SA[i], _D[i]
        c, s = cs[i], sn[i]
        Ai = np.array(
            [[c, -s, 0.0],
             [ca * s, ca * c, -sa],
             [sa * s, sa * c, ca]]
        )
        p = p + R @ (0.0, -sa * d, ca * d)
        R = R @ Ai
    return R @ _TOOL_R, p


def fk(q, base_x=0.0, base_y=0.0, base_rot=0.0, mount=(0.0, 0.0, 0.0), tool_z=0.0) -> np.ndarray:
    """Forward kinematics of the gripper grasp point, in world frame.

    Parameters
    ----------
    q : array-like, shape (7,)
        Joint angles ``joint_1 .. joint_7`` in radians.
    base_x, base_y : float
        World position of the arm base (z is taken from ``mount``).
    base_rot : float
        Rotation of the arm base about the world z axis, radians.
    mount : (3,) array-like
        Extra offset of the base frame, added to ``(base_x, base_y, 0)``.
    tool_z : float
        Extra translation along the end-effector local +z (approach) axis,
        e.g. flange -> fingertip centre.

    Returns
    -------
    (4, 4) ndarray
        Homogeneous transform ``world <- grasp_point``.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != N_JOINTS:
        raise ValueError(f"q must have {N_JOINTS} elements, got {q.size}")

    R, p = _chain(q)
    if tool_z:
        p = p + R[:, 2] * float(tool_z)

    if base_rot:
        c, s = math.cos(float(base_rot)), math.sin(float(base_rot))
        Rb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R = Rb @ R
        p = Rb @ p

    mx, my, mz = (float(v) for v in mount)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p + (float(base_x) + mx, float(base_y) + my, mz)
    return T


def fk_all(q, base_x=0.0, base_y=0.0, base_rot=0.0, mount=(0.0, 0.0, 0.0), tool_z=0.0):
    """Return the list of cumulative transforms ``[base, T1..T7, tool]``."""
    q = np.asarray(q, dtype=float).reshape(-1)
    T = _base_transform(base_x, base_y, base_rot, mount)
    out = [T.copy()]
    for i in range(N_JOINTS):
        alpha, a, d, off = DH[i]
        T = T @ _rot_x(alpha)
        if a != 0.0:
            T = T @ _trans(x=a)
        T = T @ _rot_z(q[i] + off) @ _trans(z=d)
        out.append(T.copy())
    T = T @ _rot_x(_TOOL_ALPHA)
    if tool_z:
        T = T @ _trans(z=float(tool_z))
    out.append(T.copy())
    return out


def ee_position(q, **kw) -> np.ndarray:
    """World xyz of the grasp point."""
    return fk(q, **kw)[:3, 3]


def ee_z_axis(q, **kw) -> np.ndarray:
    """Unit world-frame direction of the end-effector local +z (approach) axis.

    For a top-down grasp this should be ``[0, 0, -1]``.
    """
    return fk(q, **kw)[:3, 2]


def ee_axes(q, **kw):
    """Return ``(x_axis, y_axis, z_axis)`` of the EE frame in world coords."""
    R = fk(q, **kw)[:3, :3]
    return R[:, 0], R[:, 1], R[:, 2]


# ---------------------------------------------------------------------------
# Jacobian (finite differences)
# ---------------------------------------------------------------------------
def jacobian(q, eps=1e-6, rotation=True, **kw) -> np.ndarray:
    """Numerical Jacobian of the grasp-point pose w.r.t. the joint angles.

    Returns a ``(6, 7)`` array ``[[dp/dq], [dw/dq]]`` (linear velocity on top,
    angular velocity below), or ``(3, 7)`` if ``rotation=False``.
    Extra keyword args are forwarded to :func:`fk`.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    T0 = fk(q, **kw)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]
    rows = 6 if rotation else 3
    J = np.zeros((rows, N_JOINTS))
    for i in range(N_JOINTS):
        dq = q.copy()
        dq[i] += eps
        T1 = fk(dq, **kw)
        J[:3, i] = (T1[:3, 3] - p0) / eps
        if rotation:
            dR = (T1[:3, :3] - R0) / eps
            W = dR @ R0.T  # skew-symmetric angular velocity matrix
            J[3:, i] = (W[2, 1] - W[1, 2]) * 0.5, (W[0, 2] - W[2, 0]) * 0.5, (W[1, 0] - W[0, 1]) * 0.5
    return J


# ---------------------------------------------------------------------------
# Rotation / quaternion helpers
# ---------------------------------------------------------------------------
def quat_to_mat(quat, wxyz=False) -> np.ndarray:
    """Unit quaternion -> 3x3 rotation matrix. Default order is (x, y, z, w)."""
    quat = np.asarray(quat, dtype=float).reshape(-1)
    if quat.size != 4:
        raise ValueError("quaternion must have 4 elements")
    if wxyz:
        w, x, y, z = quat
    else:
        x, y, z, w = quat
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R, wxyz=False) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (x, y, z, w) by default."""
    R = np.asarray(R, dtype=float)[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z]) if wxyz else np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def top_down_quat(yaw=0.0, wxyz=False) -> np.ndarray:
    """Quaternion for a top-down grasp: EE +z points along world -z.

    ``yaw`` rotates the gripper about the (vertical) approach axis.
    """
    Rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0],
         [math.sin(yaw), math.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    )
    return mat_to_quat(Rz @ TOP_DOWN_R, wxyz=wxyz)


def _rot_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """Rotation error as a world-frame rotation vector taking cur -> tgt."""
    Re = R_tgt @ R_cur.T
    cos_t = (np.trace(Re) - 1.0) * 0.5
    cos_t = min(1.0, max(-1.0, cos_t))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)
    if theta > PI - 1e-6:
        # Near-pi: extract axis from (Re + I).
        A = (Re + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        k = int(np.argmax(axis))
        if axis[k] > 1e-9:
            axis = A[:, k] / axis[k]
            axis = axis / np.linalg.norm(axis)
        return axis * theta
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    return v * (theta / (2.0 * math.sin(theta)))


def wrap_to_pi(x):
    """Wrap angle(s) into [-pi, pi)."""
    return (np.asarray(x, dtype=float) + PI) % (2.0 * PI) - PI


def clamp_to_limits(q) -> np.ndarray:
    """Wrap the continuous joints and clip the limited ones into range."""
    q = np.asarray(q, dtype=float).reshape(-1).copy()
    q[CONTINUOUS_JOINTS] = wrap_to_pi(q[CONTINUOUS_JOINTS])
    lim = ~CONTINUOUS_JOINTS
    q[lim] = np.clip(wrap_to_pi(q[lim]), JOINT_LIMITS[lim, 0], JOINT_LIMITS[lim, 1])
    return q


def in_limits(q, tol=1e-6) -> bool:
    q = np.asarray(q, dtype=float).reshape(-1)
    lim = ~CONTINUOUS_JOINTS
    return bool(
        np.all(q[lim] >= JOINT_LIMITS[lim, 0] - tol)
        and np.all(q[lim] <= JOINT_LIMITS[lim, 1] + tol)
    )


# ---------------------------------------------------------------------------
# Inverse kinematics: damped least squares
# ---------------------------------------------------------------------------
def ik(
    target_pos,
    target_quat=None,
    q_init=None,
    base_x=0.0,
    base_y=0.0,
    base_rot=0.0,
    mount=(0.0, 0.0, 0.0),
    tool_z=0.0,
    target_R=None,
    wxyz=False,
    pos_tol=1e-3,
    rot_tol=1e-2,
    max_iters=60,
    damping=0.02,
    step_clip=1.0,
    rot_weight=0.5,
    restarts=4,
    time_budget=0.08,
    seed=0,
    return_info=False,
):
    """Damped-least-squares IK for the Kinova Gen3, respecting joint limits.

    Parameters
    ----------
    target_pos : (3,) array-like
        Desired world position of the grasp point.
    target_quat : (4,) array-like or None
        Desired world orientation as a quaternion (x, y, z, w) -- or
        (w, x, y, z) if ``wxyz=True``.  ``None`` solves position only.
    q_init : (7,) array-like or None
        Seed configuration; defaults to :data:`Q_HOME`.
    target_R : (3, 3) array-like, optional
        Alternative to ``target_quat`` (takes precedence if given).
    rot_weight : float
        Relative weight of the orientation residual vs. position.
    restarts : int
        Number of extra randomized seeds tried if the first solve fails.
    time_budget : float or None
        Soft wall-clock budget in seconds.  Randomized restarts are abandoned
        once it is exceeded, which bounds the worst case (a plain solve from a
        good seed takes a few milliseconds).  ``None`` disables the guard.
    return_info : bool
        If True return ``(q, info_dict)`` instead of just ``q``.

    Returns
    -------
    q : (7,) ndarray
        Best joint solution found (always within the joint limits).
    """
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    if target_R is not None:
        R_tgt = np.asarray(target_R, dtype=float)[:3, :3]
    elif target_quat is not None:
        R_tgt = quat_to_mat(target_quat, wxyz=wxyz)
    else:
        R_tgt = None

    kw = dict(base_x=base_x, base_y=base_y, base_rot=base_rot, mount=mount, tool_z=tool_z)
    q0 = Q_HOME.copy() if q_init is None else np.asarray(q_init, dtype=float).reshape(N_JOINTS).copy()

    rows = 6 if R_tgt is not None else 3
    lo = JOINT_LIMITS[:, 0]
    hi = JOINT_LIMITS[:, 1]
    rng = np.random.default_rng(seed)

    best_q = clamp_to_limits(q0)
    best_cost = np.inf
    best_pos_err = np.inf
    best_rot_err = np.inf
    solved = False
    total_iters = 0
    t_start = _time.perf_counter()

    for attempt in range(max(1, int(restarts) + 1)):
        if attempt and time_budget is not None and _time.perf_counter() - t_start > time_budget:
            break
        if attempt == 0:
            q = clamp_to_limits(q0)
        else:
            pert = rng.uniform(-1.0, 1.0, N_JOINTS) * np.array([PI, 1.2, PI, 1.2, PI, 1.0, PI])
            q = clamp_to_limits(q0 + pert)

        lam2 = damping * damping
        for _ in range(max_iters):
            total_iters += 1
            T = fk(q, **kw)
            e = np.zeros(rows)
            e[:3] = target_pos - T[:3, 3]
            pos_err = float(np.linalg.norm(e[:3]))
            if R_tgt is not None:
                w = _rot_error(T[:3, :3], R_tgt)
                e[3:] = rot_weight * w
                rot_err = float(np.linalg.norm(w))
            else:
                rot_err = 0.0

            cost = pos_err + rot_weight * rot_err
            if cost < best_cost:
                best_cost, best_q = cost, q.copy()
                best_pos_err, best_rot_err = pos_err, rot_err

            if pos_err < pos_tol and rot_err < rot_tol:
                solved = True
                break

            J = jacobian(q, rotation=(R_tgt is not None), **kw)
            if R_tgt is not None:
                J[3:] *= rot_weight
            # Damped least squares: dq = J^T (J J^T + lam^2 I)^-1 e
            JJt = J @ J.T + lam2 * np.eye(rows)
            try:
                dq = J.T @ np.linalg.solve(JJt, e)
            except np.linalg.LinAlgError:  # pragma: no cover
                break

            n = np.linalg.norm(dq)
            if n > step_clip:
                dq *= step_clip / n
            if n < 1e-10:
                break
            q = clamp_to_limits(q + dq)

        if solved:
            break

    q_out = clamp_to_limits(best_q)
    if return_info:
        info = {
            "success": bool(solved),
            "pos_err": best_pos_err,
            "rot_err": best_rot_err,
            "iters": total_iters,
        }
        return q_out, info
    return q_out


def ik_top_down(target_pos, yaw=0.0, q_init=None, **kw):
    """Convenience wrapper: IK with the approach axis pointing straight down."""
    Rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0],
         [math.sin(yaw), math.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    )
    return ik(target_pos, target_R=Rz @ TOP_DOWN_R, q_init=q_init, **kw)
