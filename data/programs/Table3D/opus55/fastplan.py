import numpy as np
from scipy.optimize import minimize
from approach import fk, down_R, _LIM, ARM_OFFSET_X, GRASP_DZ

BX_MAX = 0.17

def world_flange(x):
    bx, by, br = x[0], x[1], x[2]
    T = fk(x[3:10])
    c, s = np.cos(br), np.sin(br)
    Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    p = Rb @ (T[:3, 3] + np.array([ARM_OFFSET_X, 0, 0])) + np.array([bx, by, GRASP_DZ])
    return p, Rb @ T[:3, :3]

def solve(x0, p_cube, yaw):
    Rd = down_R(yaw)
    def cons_eq(v):
        p, R = world_flange(v[:10])
        return np.concatenate([p - p_cube, (R[:, 2] - Rd[:, 2])[:2], [R[:, 0] @ Rd[:, 1]]])
    def cons_in(v):
        d = v[:10] - x0
        s = v[10]
        bxw = v[0]
        return np.concatenate([s - d, s + d, [BX_MAX - bxw], v[3:10] - _LIM[:, 0], _LIM[:, 1] - v[3:10]])
    v0 = np.concatenate([x0, [0.5]])
    res = minimize(lambda v: v[10] + 1e-3 * np.sum((v[:10] - x0) ** 2), v0, method='SLSQP',
                   constraints=[{'type': 'eq', 'fun': cons_eq}, {'type': 'ineq', 'fun': cons_in}],
                   options={'maxiter': 200, 'ftol': 1e-9})
    v = res.x
    ce = np.abs(cons_eq(v)).max()
    ci = cons_in(v).min()
    return v[:10], v[10], ce, ci
