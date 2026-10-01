import numpy as np
from scipy.optimize import minimize
from approach import fk, ik, down_R, _LIM, ARM_OFFSET_X, _base_ext
H = 0.3947
DEPTH = 0.16
# collision probe points in flange frame (fingertips + gripper body corners)
PTS = np.array([[sx*0.05, sy*0.012, 0.15] for sx in (-1,1) for sy in (-1,1)] +
               [[sx*0.045, sy*0.035, 0.09] for sx in (-1,1) for sy in (-1,1)])

def world_T(x):
    bx, by, br = x[0], x[1], x[2]
    T = fk(x[3:10])
    c, s = np.cos(br), np.sin(br)
    Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    p = Rb @ (T[:3, 3] + np.array([ARM_OFFSET_X, 0, 0])) + np.array([bx, by, H])
    return p, Rb @ T[:3, :3]

def solve_relaxed(x0, p_cube, yaw, tilt_max=0.5, slack=(0.004, 0.012, 0.012), table_z=0.4, bmargin=0.01, xi=None):
    sl = np.asarray(slack)
    ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    perp = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    ct = np.cos(tilt_max)
    def cin(v):
        p, R = world_T(v[:10])
        e = R.T @ (p_cube - p) - np.array([0, 0, DEPTH])
        pts = p + PTS @ R.T
        d = v[:10] - x0; s = v[10]
        return np.concatenate([s - d, s + d, sl - e, sl + e, [-R[2, 2] - ct],
                               pts[:, 2] - (table_z + 0.004),
                               [0.4 - bmargin - v[0] - _base_ext(v[2])],
                               v[3:10] - _LIM[:, 0], _LIM[:, 1] - v[3:10]])
    def ceq(v):
        p, R = world_T(v[:10])
        return np.array([R[:, 0] @ perp])
    if xi is None:
        # warm start from down-orientation IK
        bx, by, br = x0[:3]
        c, s_ = np.cos(br), np.sin(br)
        dxw, dyw = p_cube[0] - bx, p_cube[1] - by
        pa = np.array([c*dxw + s_*dyw - ARM_OFFSET_X, -s_*dxw + c*dyw, p_cube[2] - H + DEPTH])
        qi, ei = ik(x0[3:10], pa, down_R(yaw - br))
        for j in (0, 2, 4, 6):
            qi[j] = x0[3+j] + ((qi[j] - x0[3+j] + np.pi) % (2*np.pi) - np.pi)
        xi = np.concatenate([x0[:3], qi]) if ei < 1e-3 else x0.copy()
    v0 = np.concatenate([xi, [np.abs(xi - x0).max() + 1e-3]])
    try:
        res = minimize(lambda v: v[10] + 1e-3*np.sum((v[:10]-x0)**2), v0, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': cin}, {'type': 'eq', 'fun': ceq}],
                       options={'maxiter': 200, 'ftol': 1e-10})
        v = res.x
    except Exception:
        return None
    if not np.all(np.isfinite(v)) or cin(v).min() < -1e-5 or abs(ceq(v)[0]) > 1e-4:
        return None
    s = np.abs(v[:10] - x0).max()
    if s > 3: return None
    return v[:10], s
