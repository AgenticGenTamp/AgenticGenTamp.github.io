"""Kinova Gen3 kinematics (numpy only)."""
import numpy as np

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])

def _T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T

_JOINTS = [
    _T([0, 0, 0.15643], [np.pi, 0, 0]),
    _T([0, 0.005375, -0.12838], [np.pi/2, 0, 0]),
    _T([0, -0.21038, -0.006375], [-np.pi/2, 0, 0]),
    _T([0, 0.006375, -0.21038], [np.pi/2, 0, 0]),
    _T([0, -0.20843, -0.006375], [-np.pi/2, 0, 0]),
    _T([0, 0.00017505, -0.10593], [np.pi/2, 0, 0]),
    _T([0, -0.10593, -0.00017505], [-np.pi/2, 0, 0]),
]
_EE = _T([0, 0, -0.061525], [np.pi, 0, 0])

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c; return T

def fk_all(q, tool=0.0):
    """Return list of joint frames (in arm base frame) and final tool frame."""
    T = np.eye(4); frames = []
    for i in range(7):
        T = T @ _JOINTS[i] @ _rz(q[i])
        frames.append(T.copy())
    T = T @ _EE
    Tt = T.copy(); Tt[:3, 3] = T[:3, 3] + T[:3, 2] * tool
    return frames, Tt

def fk(q, tool=0.0):
    return fk_all(q, tool)[1]

JOINT_LIMITS = np.array([[-1e9, 1e9], [-2.24, 2.24], [-1e9, 1e9], [-2.57, 2.57],
                         [-1e9, 1e9], [-2.09, 2.09], [-1e9, 1e9]])

def jacobian(q, tool=0.0):
    frames, Tt = fk_all(q, tool)
    p = Tt[:3, 3]
    J = np.zeros((6, 7))
    for i, F in enumerate(frames):
        z = F[:3, 2]; o = F[:3, 3]
        J[:3, i] = np.cross(z, p - o); J[3:, i] = z
    return J, Tt

def rot_err(Rc, Rd):
    """Axis-angle vector rotating Rc to Rd (world frame)."""
    Re = Rd @ Rc.T
    ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    s = np.sin(ang)
    if s < 1e-6:
        # ang ~ pi
        w, V = np.linalg.eigh((Re + Re.T) / 2 + np.eye(3))
        ax = V[:, np.argmax(w)]
        return ax * ang
    return v / (2 * s) * ang

def ik(q0, p_des, R_des, tool=0.0, iters=200, tol=1e-4, w_rot=0.5, q_nom=None):
    """Damped least squares IK in the arm base frame. Returns (q, pos_err, rot_err)."""
    q = np.array(q0, dtype=float)
    lam = 0.05
    for it in range(iters):
        J, Tt = jacobian(q, tool)
        ep = p_des - Tt[:3, 3]
        er = rot_err(Tt[:3, :3], R_des)
        e = np.concatenate([ep, w_rot * er])
        if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
            break
        Jw = J.copy(); Jw[3:] *= w_rot
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam**2 * np.eye(6), e)
        if q_nom is not None:
            N = np.eye(7) - np.linalg.pinv(Jw) @ Jw
            dq += N @ (0.05 * (q_nom - q))
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        q = q + dq
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    Tt = fk(q, tool)
    return q, np.linalg.norm(p_des - Tt[:3, 3]), np.linalg.norm(rot_err(Tt[:3, :3], R_des))

SAFE_LIMITS = np.array([[-1e9, 1e9], [-2.0, 2.0], [-1e9, 1e9], [-2.45, 2.45],
                        [-1e9, 1e9], [-1.95, 1.95], [-1e9, 1e9]])
_SEEDS = [np.array(s) for s in [
    [0, -0.35, np.pi, -2.55, 0, -0.87, np.pi / 2],
    [0, 0.6, np.pi, -1.6, 0, -1.0, np.pi / 2],
    [0, 0.3, np.pi, -1.2, 0, 1.0, np.pi / 2],
    [0, 1.0, 0, 1.5, 0, 0.8, 0],
    [0, 0.8, np.pi, -1.0, 0, -1.3, -np.pi / 2],
]]

def wrap_diff(a, b):
    d = a - b
    for i in (0, 2, 4, 6):
        d[i] = (d[i] + np.pi) % (2 * np.pi) - np.pi
    return d

def ik_best(q_cur, p_des, R_des, tool=0.0, extra_seeds=(), pos_tol=2e-3, rot_tol=2e-2, n_random=0):
    """Multi-start IK; returns solution closest to q_cur (in wrapped joint space) that
    stays within safe limits. Returns (q, ok)."""
    best, bestc = None, 1e18
    seeds = [np.asarray(q_cur, float)] + list(extra_seeds) + _SEEDS
    rng = np.random.default_rng(0)
    seeds += list(rng.uniform(-2.0, 2.0, (n_random, 7)))
    for s in seeds:
        q, ep, er = ik(s, p_des, R_des, tool=tool, iters=150)
        if ep > pos_tol or er > rot_tol:
            continue
        # express revolute continuous joints as nearest to q_cur
        d = wrap_diff(q, q_cur)
        qn = np.asarray(q_cur) + d
        if np.any(qn[[1, 3, 5]] < SAFE_LIMITS[[1, 3, 5], 0]) or np.any(qn[[1, 3, 5]] > SAFE_LIMITS[[1, 3, 5], 1]):
            continue
        c = np.sum(np.abs(d) * np.array([1.5, 1.5, 1, 1, 1, 1, 0.5]))
        if c < bestc:
            best, bestc = qn, c
    if best is None:
        q, ep, er = ik(q_cur, p_des, R_des, tool=tool, iters=150)
        return np.asarray(q_cur) + wrap_diff(q, q_cur), False
    return best, True
