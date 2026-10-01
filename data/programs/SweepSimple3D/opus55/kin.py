"""Kinematics for TidyBot++ (holonomic base + Kinova Gen3 7-DOF + Robotiq 2F-85).

numpy only.  Calibrated empirically against the environment (see calib_*.py).

fk(base_x, base_y, base_rot, q7) -> (pos_world(3,), R_world(3x3))
    Grasp point (between fingertips) and its orientation.  R[:,2] is the
    approach direction (points from wrist toward fingertips), R[:,1] is the
    finger-closing axis (fingers move along +-R[:,1]).
ik(base_pose, q_init, target_pos, target_R_or_down=None, yaw=None, ...) -> q7
    Damped-least-squares IK.  target_R_or_down may be
        None      -> position only
        'down'    -> approach axis pointing straight down, free yaw
                     (or fixed yaw if yaw=float is given: yaw is the world
                      angle of the finger-closing axis R[:,1])
        3x3 array -> full orientation.
"""
import numpy as np

# ---------------------------------------------------------------- params
# Arm mount in base frame (x forward, y left, z up) and yaw offset.
MOUNT_XYZ = np.array([0.12, 0.0, 0.3943])   # fitted (0.1199, 0.0004, 0.3943), yaw 2e-4
MOUNT_YAW = 0.0
TIP_LEN = 0.155       # grasp point (held-cube center) beyond bracelet tool frame; closed fingertip end at ~0.1625

# Joint limits (Menagerie kinova gen3); continuous joints given wide range.
Q_LO = np.array([-1e9, -2.41, -1e9, -2.66, -1e9, -2.23, -1e9])
Q_HI = -Q_LO
Q_HOME = np.array([0.0, -0.349, 3.142, -2.548, 0.0, -0.873, 1.571])


def _qmat(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# (pos, rot) of each joint body relative to its parent (Menagerie kinova_gen3)
_CHAIN = [
    (np.array([0, 0, 0.15643]), _qmat(0, 1, 0, 0)),
    (np.array([0, 0.005375, -0.12838]), _qmat(1, 1, 0, 0)),
    (np.array([0, -0.21038, -0.006375]), _qmat(1, -1, 0, 0)),
    (np.array([0, 0.006375, -0.21038]), _qmat(1, 1, 0, 0)),
    (np.array([0, -0.20843, -0.006375]), _qmat(1, -1, 0, 0)),
    (np.array([0, 0.00017505, -0.10593]), _qmat(1, 1, 0, 0)),
    (np.array([0, -0.10593, -0.00017505]), _qmat(1, -1, 0, 0)),
]
_TOOL_P = np.array([0, 0, -0.0615])
_TOOL_R = _qmat(0, 1, 0, 0)


def set_params(mount_xyz=None, mount_yaw=None, tip_len=None):
    global MOUNT_XYZ, MOUNT_YAW, TIP_LEN
    if mount_xyz is not None:
        MOUNT_XYZ = np.asarray(mount_xyz, float)
    if mount_yaw is not None:
        MOUNT_YAW = float(mount_yaw)
    if tip_len is not None:
        TIP_LEN = float(tip_len)


def fk_arm(q, mount_xyz=None, mount_yaw=None, tip_len=None):
    """Grasp point pose in the robot BASE frame."""
    mxyz = MOUNT_XYZ if mount_xyz is None else mount_xyz
    myaw = MOUNT_YAW if mount_yaw is None else mount_yaw
    tl = TIP_LEN if tip_len is None else tip_len
    R = _rz(myaw)
    p = np.array(mxyz, float)
    for (pp, pr), qi in zip(_CHAIN, q):
        p = p + R @ pp
        R = R @ pr @ _rz(qi)
    p = p + R @ _TOOL_P
    R = R @ _TOOL_R
    p = p + R[:, 2] * tl
    return p, R


def fk(base_x, base_y, base_rot, q7, **kw):
    p, R = fk_arm(np.asarray(q7, float), **kw)
    Rb = _rz(base_rot)
    return np.array([base_x, base_y, 0.0]) + Rb @ p, Rb @ R


def _err(p, R, tp, mode, tR, yaw):
    e = [tp - p]
    if mode == 'down':
        z = R[:, 2]
        e.append(np.cross(z, np.array([0, 0, -1.0])))
        if yaw is not None:
            y_des = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            y = R[:, 1]
            e.append(np.array([0, 0, np.cross(y, y_des)[2]]))
    elif mode == 'full':
        e.append(0.5 * (np.cross(R[:, 0], tR[:, 0]) + np.cross(R[:, 1], tR[:, 1])
                        + np.cross(R[:, 2], tR[:, 2])))
    return np.concatenate(e)


def ik(base_pose, q_init, target_pos, target_R_or_down=None, yaw=None,
       iters=200, tol=1e-4, yaw_sym=np.pi, damping=0.05, rot_weight=0.5, step_max=0.3):
    """Damped least-squares IK. Returns (q, ok, err_norm).

    base_pose = (x, y, rot).  yaw: if 'down', optional fixed world yaw of the
    finger-closing axis (gripper yaw symmetric mod pi).
    """
    bx, by, br = base_pose
    tp = np.asarray(target_pos, float)
    if target_R_or_down is None:
        mode, tR = 'pos', None
    elif isinstance(target_R_or_down, str):
        mode, tR = 'down', None
    else:
        mode, tR = 'full', np.asarray(target_R_or_down, float)
    q = np.array(q_init, float).copy()
    if mode == 'down' and yaw is not None and yaw_sym:
        # gripper is symmetric: choose yaw (mod pi, or mod yaw_sym) closest to current
        _, R0 = fk(bx, by, br, q)
        cur = np.arctan2(R0[1, 1], R0[0, 1])
        yaw = cur + (yaw - cur + yaw_sym / 2) % yaw_sym - yaw_sym / 2
    eps = 1e-6
    for it in range(iters):
        p, R = fk(bx, by, br, q)
        e = _err(p, R, tp, mode, tR, yaw)
        w = np.ones_like(e)
        w[3:] = rot_weight
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 10 * tol:
            break
        J = np.zeros((len(e), 7))
        for i in range(7):
            dq = q.copy()
            dq[i] += eps
            p2, R2 = fk(bx, by, br, dq)
            J[:, i] = -(_err(p2, R2, tp, mode, tR, yaw) - e) / eps
        Jw = J * w[:, None]
        ew = e * w
        lam = damping ** 2
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(len(e)), ew)
        # small pull toward home for joints 2,4,6 away from limits (nullspace)
        n = np.linalg.norm(dq)
        if n > step_max:
            dq *= step_max / n
        q = np.clip(q + dq, Q_LO, Q_HI)
    p, R = fk(bx, by, br, q)
    e = _err(p, R, tp, mode, tR, yaw)
    ok = np.linalg.norm(e[:3]) < 2e-3 and np.linalg.norm(e[3:]) < 2e-2
    return q, ok, float(np.linalg.norm(e))


def ik_multi(base_pose, q_init, target_pos, target_R_or_down=None, yaw=None,
             tries=8, seed=0, **kw):
    """ik() with restarts from q_init, Q_HOME and random perturbations."""
    rng = np.random.default_rng(seed)
    best = None
    inits = [np.asarray(q_init, float), Q_HOME]
    for t in range(tries):
        q0 = inits[t] if t < 2 else Q_HOME + rng.uniform(-0.7, 0.7, 7)
        q, ok, e = ik(base_pose, q0, target_pos, target_R_or_down, yaw, **kw)
        if ok:
            return wrap_to(q, q_init), True, e
        if best is None or e < best[2]:
            best = (q, ok, e)
    return wrap_to(best[0], q_init), best[1], best[2]


def wrap_to(q, q_ref):
    """Wrap continuous joints (1,3,5,7) to be nearest to q_ref."""
    q = np.array(q, float)
    for i in (0, 2, 4, 6):
        q[i] = q_ref[i] + (q[i] - q_ref[i] + np.pi) % (2 * np.pi) - np.pi
    return q
