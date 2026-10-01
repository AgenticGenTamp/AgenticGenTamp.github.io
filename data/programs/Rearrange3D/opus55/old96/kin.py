"""Kinova Gen3 kinematics (MuJoCo menagerie body chain) + mobile base."""
import numpy as np

def _qmat(w, x, y, z):
    q = np.array([w, x, y, z], float); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

# (pos, quat) of each joint body relative to parent; joint about local z
CHAIN = [
    ((0, 0, 0.15643), (0, 1, 0, 0)),
    ((0, 0.005375, -0.12838), (1, 1, 0, 0)),
    ((0, -0.21038, -0.006375), (1, -1, 0, 0)),
    ((0, 0.006375, -0.21038), (1, 1, 0, 0)),
    ((0, -0.20843, -0.006375), (1, -1, 0, 0)),
    ((0, 0.00017505, -0.10593), (1, 1, 0, 0)),
    ((0, -0.10593, -0.00017505), (1, -1, 0, 0)),
]
EE = ((0, 5.4e-05, -0.061525), (0, 1, 0, 0))
_CH = [(np.array(p, float), _qmat(*q)) for p, q in CHAIN]
_EE = (np.array(EE[0], float), _qmat(*EE[1]))

# calibratable params
MOUNT = np.array([0.12, 0.0, 0.3645])  # arm base in robot base frame
TOOL = 0.1325  # closed fingertip distance along ee z (table contact); open tips ~1.3cm higher


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk_arm(q, tool=None):
    """Return (pos, R) of grasp point in arm-base frame, plus joint frames."""
    if tool is None:
        tool = TOOL
    p = np.zeros(3); R = np.eye(3)
    frames = []
    for i, (pp, rr) in enumerate(_CH):
        p = p + R @ pp
        R = R @ rr @ rotz(q[i])
        frames.append((p.copy(), R.copy()))
    p = p + R @ _EE[0]; R = R @ _EE[1]
    p = p + R @ np.array([0, 0, tool])
    return p, R, frames


def fk_world(base, q, tool=None, mount=None):
    if mount is None:
        mount = MOUNT
    p, R, _ = fk_arm(q, tool)
    Rb = rotz(base[2])
    pw = np.array([base[0], base[1], 0.0]) + Rb @ (mount + p)
    return pw, Rb @ R


LIM = np.array([np.inf, 2.24, np.inf, 2.57, np.inf, 2.09, np.inf])
READY = np.array([0.0, 0.35, 3.14, -2.0, 0.0, -0.9, 1.57])


def jac_arm(q, tool=None):
    """Analytic 6x7 jacobian (pos, rot) in arm base frame."""
    p, R, frames = fk_arm(q, tool)
    J = np.zeros((6, 7))
    for i, (pi, Ri) in enumerate(frames):
        z = Ri[:, 2]
        d = p - pi
        J[0, i] = z[1] * d[2] - z[2] * d[1]
        J[1, i] = z[2] * d[0] - z[0] * d[2]
        J[2, i] = z[0] * d[1] - z[1] * d[0]
        J[3:, i] = z
    return J, p, R


def rot_err(Rt, R):
    dR = Rt @ R.T
    ang = np.array([dR[2, 1]-dR[1, 2], dR[0, 2]-dR[2, 0], dR[1, 0]-dR[0, 1]]) / 2
    return ang


def ik_arm(p_t, R_t, q0, tool=None, iters=100, wrot=0.5, rest=None, wnull=0.05):
    q = np.array(q0, float)
    lim = LIM - 0.05
    e = np.zeros(6)
    for it in range(iters):
        J, p, R = jac_arm(q, tool)
        e = np.concatenate([p_t - p, wrot * rot_err(R_t, R)])
        en = np.linalg.norm(e)
        if np.abs(e[:3]).max() < 2e-4 and np.abs(e[3:]).max() < 1e-3:
            break
        J[3:] *= wrot
        lam = 1e-4 + 0.01 * min(en, 0.1)
        JJ = J @ J.T + lam * np.eye(6)
        Jp = J.T @ np.linalg.inv(JJ)
        dq = Jp @ e
        if rest is not None:
            N = np.eye(7) - Jp @ J
            dq += N @ (wnull * (rest - q))
        n = np.max(np.abs(dq))
        if n > 0.3:
            dq *= 0.3 / n
        q += dq
        q = np.clip(q, -lim, lim)
    return q, np.linalg.norm(e)
