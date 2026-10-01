"""Self-contained calibrated kinematics for the Kinova Gen3 in this env.

Rotational chain from the gen3 URDF (exact); link translations calibrated
empirically (A) so that `linpos(q)` gives the world position (robot-base
frame) of a part grasped with the standard top-down grasp.
"""
import numpy as np

def _rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])

def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

_RPYS = [_rpy(-np.pi, 0, 0), _rpy(np.pi/2, 0, 0), _rpy(-np.pi/2, 0, 0),
         _rpy(np.pi/2, 0, 0), _rpy(-np.pi/2, 0, 0), _rpy(np.pi/2, 0, 0),
         _rpy(-np.pi/2, 0, 0)]
_RTOOL = _rpy(np.pi, 0, 0)

A = np.array([[0.1199, 0.0, 0.13981],
              [0.0, 0.00588, -0.13981],
              [0.0, -0.21038, -0.00588],
              [0.0, 0.00638, -0.21038],
              [0.0, -0.15718, -0.00638],
              [0.0, 0.00017, -0.15718],
              [0.0, -0.18362, -0.00017],
              [-0.01989, 0.0, -0.18362]])

LIM_LO = np.array([-np.inf, -2.40, -np.inf, -2.64, -np.inf, -2.20, -np.inf])
LIM_HI = np.array([np.inf, 2.40, np.inf, 2.64, np.inf, 2.20, np.inf])

RDOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])


def rots(q):
    """Rotation of each frame: Rs[0]=I, Rs[i] after joint i; plus tool R."""
    Rs = [np.eye(3)]
    R = np.eye(3)
    for i in range(7):
        R = R @ _RPYS[i] @ rotz(q[i])
        Rs.append(R)
    return Rs, R @ _RTOOL


def linpos(q):
    Rs, _ = rots(q)
    p = np.zeros(3)
    for i in range(8):
        p = p + Rs[i] @ A[i]
    return p


def pose(q):
    Rs, Rt = rots(q)
    p = np.zeros(3)
    for i in range(8):
        p = p + Rs[i] @ A[i]
    return p, Rt


def jac(q):
    """(position jacobian, angular jacobian, p, Rtool) - analytic."""
    Rs, Rt = rots(q)
    terms = [Rs[i] @ A[i] for i in range(8)]
    p = np.sum(terms, axis=0)
    o = np.zeros(3)
    Jp = np.zeros((3, 7))
    Jw = np.zeros((3, 7))
    o = terms[0].copy()  # origin of frame 1 = Rs[0]@A[0]
    for i in range(1, 8):
        z = Rs[i][:, 2]
        Jp[:, i-1] = np.cross(z, p - o)
        Jw[:, i-1] = z
        o = o + terms[i]
    return Jp, Jw, p, Rt


def roterr(Rc, Rt):
    Re = Rt @ Rc.T
    w = np.array([Re[2, 1]-Re[1, 2], Re[0, 2]-Re[2, 0], Re[1, 0]-Re[0, 1]])
    s = np.linalg.norm(w)
    c = (np.trace(Re) - 1) / 2
    ang = np.arctan2(s/2, c)
    if s > 1e-9:
        return w / s * ang
    return np.zeros(3)


def _solve(q0, tp, tR, iters=120):
    q = np.array(q0, float)
    for _ in range(iters):
        Jp, Jw, p, R = jac(q)
        ep = tp - p
        er = roterr(R, tR)
        if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
            break
        e = np.concatenate([ep, 0.5*er])
        J = np.vstack([Jp, Jw])
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam*lam*np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq = dq / n * 0.3
        q = np.clip(q + dq, LIM_LO + 1e-6, LIM_HI - 1e-6)
    Jp, Jw, p, R = jac(q)
    return q, np.linalg.norm(tp - p), np.linalg.norm(roterr(R, tR))


def ik(q0, tp, tR, restarts=8, rng=None, pos_tol=1.5e-3, rot_tol=1.5e-2):
    """IK on the calibrated model. Returns (q, ok)."""
    q, pe, re = _solve(q0, tp, tR)
    if pe < pos_tol and re < rot_tol:
        return q, True
    if rng is None:
        rng = np.random.default_rng(0)
    best = (pe + 0.05*re, q)
    for _ in range(restarts):
        seed = np.clip(np.array(q0, float) + rng.normal(0, 1.0, 7),
                       LIM_LO + 1e-6, LIM_HI - 1e-6)
        qq, pe, re = _solve(seed, tp, tR)
        if pe < pos_tol and re < rot_tol:
            return qq, True
        if pe + 0.05*re < best[0]:
            best = (pe + 0.05*re, qq)
    return best[1], False
