"""Kinematics for Kinova Gen3 on a holonomic base (numpy only)."""
import numpy as np

# (pos, quat wxyz) of each joint body relative to parent (MuJoCo menagerie gen3)
_S = np.sqrt(0.5)
_LINKS = [
    ((0.0, 0.0, 0.15643), (0.0, 1.0, 0.0, 0.0)),
    ((0.0, 0.005375, -0.12838), (_S, _S, 0.0, 0.0)),
    ((0.0, -0.21038, -0.006375), (_S, -_S, 0.0, 0.0)),
    ((0.0, 0.006375, -0.21038), (_S, _S, 0.0, 0.0)),
    ((0.0, -0.20843, -0.006375), (_S, -_S, 0.0, 0.0)),
    ((0.0, 0.00017505, -0.10593), (_S, _S, 0.0, 0.0)),
    ((0.0, -0.10593, -0.00017505), (_S, -_S, 0.0, 0.0)),
]
# bracelet -> tool (pinch point) along -z of bracelet frame
TOOL_OFFSET = -0.061525 - 0.13  # tunable

# Arm mount on base (base frame): tunable
MOUNT = np.array([0.1195, -0.001, 0.3944])


def quat2mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


_LR = [(np.array(p), quat2mat(q)) for p, q in _LINKS]


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def fk_arm(q, tool=None):
    """Tool pose in arm-base frame. Returns (pos, R)."""
    if tool is None:
        tool = TOOL_OFFSET
    R = np.eye(3)
    p = np.zeros(3)
    for i, (lp, lr) in enumerate(_LR):
        p = p + R @ lp
        R = R @ lr @ rotz(q[i])
    p = p + R @ np.array([0, 0, tool])
    return p, R


def fk_world(base, q, tool=None, mount=None):
    if mount is None:
        mount = MOUNT
    bx, by, bt = base
    Rb = rotz(bt)
    p, R = fk_arm(q, tool)
    pw = np.array([bx, by, 0.0]) + Rb @ (mount + p)
    return pw, Rb @ R


def rot_err(Rd, R):
    """Axis-angle error vector (world frame) rotating R to Rd."""
    Re = Rd @ R.T
    c = (np.trace(Re) - 1) / 2
    c = min(1.0, max(-1.0, c))
    ang = np.arccos(c)
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    s = np.sin(ang)
    if s < 1e-6:
        if ang < 1e-3:
            return 0.5 * v
        # near pi
        w, V = np.linalg.eigh((Re + Re.T) / 2)
        ax = V[:, np.argmax(w)]
        return ax * ang
    return v * ang / (2 * s)


QLO = np.array([-1e9, -2.41, -1e9, -2.66, -1e9, -2.23, -1e9])
QHI = np.array([1e9, 2.41, 1e9, 2.66, 1e9, 2.23, 1e9])


def ik_arm(p_des, R_des, q0, iters=100, wrot=0.3, tol=1e-4, tool=None, lam=0.02):
    """Damped least squares IK in arm-base frame. R_des may be None (position only)."""
    q = np.array(q0, dtype=float)
    eps = 1e-5
    for _ in range(iters):
        p, R = fk_arm(q, tool)
        e = p_des - p
        if R_des is not None:
            e = np.concatenate([e, wrot * rot_err(R_des, R)])
        if np.linalg.norm(e) < tol:
            break
        J = np.zeros((len(e), 7))
        for j in range(7):
            dq = q.copy(); dq[j] += eps
            pj, Rj = fk_arm(dq, tool)
            col = (pj - p) / eps
            if R_des is not None:
                col = np.concatenate([col, wrot * rot_err(Rj, R) / eps])
            J[:, j] = col
        JJt = J @ J.T + lam * lam * np.eye(len(e))
        dq = J.T @ np.linalg.solve(JJt, e)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        q = np.clip(q + dq, QLO, QHI)
    p, R = fk_arm(q, tool)
    err = np.linalg.norm(p_des - p)
    if R_des is not None:
        err = max(err, np.linalg.norm(rot_err(R_des, R)) * 0.1)
    return q, err


def world_to_arm(pw, base, mount=None):
    if mount is None:
        mount = MOUNT
    bx, by, bt = base
    return rotz(-bt) @ (np.asarray(pw) - np.array([bx, by, 0.0])) - mount
