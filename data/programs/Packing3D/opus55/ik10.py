"""10-DOF (base x,y,rot + 7 joints) IK minimizing the max per-DOF change."""
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation
from fk import fk

# bounds on absolute config (base x, base y, base rot, q1..q7)
QLO = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -2.40, -np.inf, -2.65, -np.inf, -2.22, -np.inf])
QHI = np.array([-0.08, np.inf, np.inf, np.inf, 2.40, np.inf, 2.65, np.inf, 2.22, np.inf])


def tcp(c):
    M, _ = fk(c[3:10], base=c[0:3])
    return M


def _pose_res(c, target_p, target_R, yaw_free):
    M = tcp(c)
    rp = M[:3, 3] - target_p
    if yaw_free:
        rr = M[:2, 2]  # z-axis x,y components must be 0 (pointing down)
    else:
        rr = Rotation.from_matrix(target_R.T @ M[:3, :3]).as_rotvec()
    return np.concatenate([rp, 0.3 * rr])


def _bound_pen(c):
    return 30.0 * (np.maximum(0.0, c - (QHI - 0.003)) + np.maximum(0.0, (QLO + 0.003) - c))


def solve(c0, target_p, target_R, yaw_free=False, use_base=True, t_hint=None):
    """Return (c, maxdelta, err). c0: current 10-config."""
    c0 = np.asarray(c0, float)
    free = np.ones(10, bool)
    if not use_base:
        free[:3] = False
    idx = np.where(free)[0]

    def full(x):
        c = c0.copy(); c[idx] = x[:len(idx)]; return c

    # stage 1: L2-regularized least squares
    w = np.ones(10) * 0.02
    w[:3] = 0.03

    def r1(x):
        c = full(x)
        return np.concatenate([_pose_res(c, target_p, target_R, yaw_free) * 10.0,
                               (w * (c - c0))[idx], _bound_pen(c)[idx]])
    lo = np.maximum(QLO[idx], -1e9); hi = np.minimum(QHI[idx], 1e9)
    x0 = np.clip(c0[idx], lo + 1e-6, hi - 1e-6)
    s1 = least_squares(r1, x0, method='lm', xtol=1e-10, ftol=1e-10, max_nfev=400)
    x1 = s1.x
    # stage 2: minimize t s.t. |c - c0| <= t and pose residual = 0
    n = len(idx)
    t1 = np.max(np.abs(full(x1) - c0))
    z0 = np.concatenate([x1, [t1]])
    cons = [
        {'type': 'eq', 'fun': lambda z: _pose_res(full(z[:n]), target_p, target_R, yaw_free)},
        {'type': 'ineq', 'fun': lambda z: np.concatenate([z[n] - (z[:n] - c0[idx]), z[n] + (z[:n] - c0[idx])])},
    ]
    bnds = [(l if np.isfinite(l) else None, h if np.isfinite(h) else None) for l, h in zip(QLO[idx], QHI[idx])] + [(0, None)]
    try:
        s2 = minimize(lambda z: z[n], z0, method='SLSQP', constraints=cons, bounds=bnds,
                      options={'maxiter': 60, 'ftol': 1e-7})
        c2 = full(s2.x[:n])
        e2 = np.linalg.norm(_pose_res(c2, target_p, target_R, yaw_free))
    except Exception:
        e2 = np.inf
    c1 = full(x1)
    e1 = np.linalg.norm(_pose_res(c1, target_p, target_R, yaw_free))
    if e2 < 2e-4 and np.max(np.abs(c2 - c0)) <= t1 + 1e-9:
        c = c2
    else:
        c = c1
    # polish exactly (small LS from c)
    def r3(x):
        cc = full(x)
        return np.concatenate([_pose_res(cc, target_p, target_R, yaw_free) * 10.0, 1e-4 * (cc - c)[idx], _bound_pen(cc)[idx]])
    s3 = least_squares(r3, c[idx], method='lm', xtol=1e-12, ftol=1e-12, max_nfev=200)
    c = full(s3.x)
    err = np.linalg.norm(_pose_res(c, target_p, target_R, yaw_free)) + np.sum(np.maximum(0, c - QHI) + np.maximum(0, QLO - c))
    return c, np.max(np.abs(c - c0)), err
