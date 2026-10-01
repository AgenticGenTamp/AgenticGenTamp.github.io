"""PR2 left-arm forward kinematics and IK (numpy/scipy only)."""
import numpy as np
from scipy.optimize import least_squares

TORSO_Z = 0.790675  # base_footprint->torso_lift_link z at torso=0 (calibrated below)
TORSO_LIFT = 0.20
TOOL_LEN = 0.18
JLOW = np.array([-0.7146, -0.5236, -0.65, -2.1213, -np.inf, -2.0, -np.inf])
JHIGH = np.array([2.2854, 1.3963, 3.75, -0.15, np.inf, 0.0, np.inf])

def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])
def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0, 0], [0, c, -s], [0, s, c]])

def fk_base(q, torso=None):
    """Tool frame pose in robot base frame. Returns (p, R)."""
    if torso is None:
        torso = TORSO_LIFT
    p = np.array([-0.05, 0.188, TORSO_Z + torso])
    R = rz(q[0])
    p = p + R @ np.array([0.1, 0, 0])
    R = R @ ry(q[1]) @ rx(q[2])
    p = p + R @ np.array([0.4, 0, 0])
    R = R @ ry(q[3]) @ rx(q[4])
    p = p + R @ np.array([0.321, 0, 0])
    R = R @ ry(q[5]) @ rx(q[6])
    p = p + R @ np.array([TOOL_LEN, 0, 0])
    return p, R

def fk_world(base, q, torso=None):
    x, y, th = base
    p, R = fk_base(q, torso)
    Rb = rz(th)
    return Rb @ p + np.array([x, y, 0.]), Rb @ R

ROBOT_BOX = (0.0, -0.045, 0.335, 0.405)  # centre offset (x,y) in robot frame, half extents
BASE_XLIM = 4.97


def rect_pen(b, half, table, margin=None):
    """SAT penetration depth between robot base box (pose b) and axis-aligned table
    (cx, cy, hx, hy). 'half' is used as extra margin above the nominal footprint (half-0.33)."""
    import math
    cx, cy, hx, hy = table
    m = half - 0.33
    if m < 0.0:
        m = 0.0
    ox, oy, bhx, bhy = ROBOT_BOX
    bhx += m; bhy += m
    th = float(b[2])
    c = math.cos(th); s = math.sin(th)
    ac = abs(c); as_ = abs(s)
    dx = float(b[0]) + c * ox - s * oy - cx
    dy = float(b[1]) + s * ox + c * oy - cy
    p1 = hx + bhx * ac + bhy * as_ - abs(dx)
    p2 = hy + bhx * as_ + bhy * ac - abs(dy)
    p3 = bhx + hx * ac + hy * as_ - abs(dx * c + dy * s)
    p4 = bhy + hx * as_ + hy * ac - abs(-dx * s + dy * c)
    r = min(p1, p2, p3, p4)
    return r if r > 0.0 else 0.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def ik_single_old(target_p, approach, base0, q0, torso=None, free_base=True, base_nom=None,
       zsign=1.0, w_reg=1e-3, w_base=1e-2, elbow_min=None, zaxis=None, table=None, base_half=0.38):
    """Solve for (base, q) so tool at target_p, tool x-axis = approach (unit, world),
    tool z-axis = zsign*world up. Returns (base, q, err)."""
    target_p = np.asarray(target_p, float)
    d = np.asarray(approach, float); d = d / np.linalg.norm(d)
    zt = np.array([0, 0, zsign]) if zaxis is None else np.asarray(zaxis, float)
    if base_nom is None:
        base_nom = np.array(base0, float)
    base0 = np.array(base0, float); q0 = np.array(q0, float)
    lo = np.concatenate([[-BASE_XLIM, -np.inf, -np.inf], JLOW + 1e-4])
    hi = np.concatenate([[BASE_XLIM, np.inf, np.inf], JHIGH - 1e-4])
    lo = np.where(np.isinf(lo), -1e3, lo); hi = np.where(np.isinf(hi), 1e3, hi)

    def res(v):
        b = v[:3] if free_base else base0
        q = v[3:]
        p, R = fk_world(b, q, torso)
        r = [p - target_p, 0.5 * (R[:, 0] - d), 0.5 * (R[:, 2] - zt),
             w_reg * (q - q0)]
        if free_base:
            r.append(w_base * (b - base_nom))
        if table is not None and free_base:
            r.append([5.0 * rect_pen(b, base_half, table)])
        if elbow_min is not None:
            pts = fk_points(b, q, torso)
            r.append([max(0.0, elbow_min - pts[1][2])])
        return np.concatenate(r)
    x0 = np.concatenate([base0, q0])
    x0 = np.clip(x0, lo, hi)
    sol = least_squares(res, x0, bounds=(lo, hi), max_nfev=400, xtol=1e-10, ftol=1e-10)
    v = sol.x
    b = v[:3] if free_base else base0
    q = v[3:].copy()
    q[4] = wrap(q[4]); q[6] = wrap(q[6])
    p, R = fk_world(b, q, torso)
    err = np.linalg.norm(p - target_p) + np.linalg.norm(R[:, 0] - d) + np.linalg.norm(R[:, 2] - zt)
    return np.array(b), q, err

def fk_points(base, q, torso=None):
    """World positions of shoulder, elbow, wrist, tool."""
    if torso is None:
        torso = TORSO_LIFT
    x, y, th = base
    Rb = rz(th); t = np.array([x, y, 0.])
    pts = []
    p = np.array([-0.05, 0.188, TORSO_Z + torso])
    R = rz(q[0]); p = p + R @ np.array([0.1, 0, 0]); pts.append(p)
    R = R @ ry(q[1]) @ rx(q[2]); p = p + R @ np.array([0.4, 0, 0]); pts.append(p)
    R = R @ ry(q[3]) @ rx(q[4]); p = p + R @ np.array([0.321, 0, 0]); pts.append(p)
    R = R @ ry(q[5]) @ rx(q[6]); p = p + R @ np.array([TOOL_LEN, 0, 0]); pts.append(p)
    return [Rb @ pp + t for pp in pts]


SEEDS = [np.array(s_) for s_ in [
    [0.39, 0.33, 0.0, -1.52, 2.72, -1.22, -2.99],
    [0.0, 0.3, 0.0, -1.5, 0.0, -0.5, 0.0],
    [0.5, 0.0, 1.5, -1.5, 0.0, -1.0, 0.0],
    [-0.3, 0.5, 0.0, -1.8, 0.0, -0.8, 0.0],
    [0.8, 0.2, 0.5, -1.0, -1.0, -1.0, 1.0],
    [0.0, 0.8, 0.0, -1.2, 3.1, -0.5, 3.1],
]]

def ik(target_p, approach, base0, q0, n_restarts=6, tol=2e-3, **kw):
    if kw.get('zaxis') is None:
        a = np.asarray(approach, float)
        a = a / np.linalg.norm(a)
        if abs(a[2]) > 1e-6:
            zt = np.array([0.0, 0.0, 1.0]) - a[2] * a
            kw['zaxis'] = zt / np.linalg.norm(zt)
    best = None
    seeds = [np.array(q0, float)] + SEEDS[:n_restarts]
    for i, s0 in enumerate(seeds):
        b, q, err = ik_single(target_p, approach, base0, s0, **kw)
        dist = np.abs(np.r_[q[:4] - q0[:4], wrap(q[4] - q0[4]), q[5] - q0[5], wrap(q[6] - q0[6])]).max()
        score = (err > tol, err if err > tol else dist)
        if best is None or score < best[0]:
            best = (score, b, q, err)
        if i == 0 and err < tol:
            break
    return best[1], best[2], best[3]


def ik_single(target_p, approach, base0, q0, torso=None, free_base=True, base_nom=None,
              zsign=1.0, w_reg=1e-3, w_base=1e-2, elbow_min=None, zaxis=None, table=None, base_half=0.38):
    from kinpy import ik_lm
    if zaxis is None:
        zaxis = (0.0, 0.0, zsign)
    pen_fn = None
    if table is not None:
        pen_fn = lambda b: rect_pen(b, base_half, table)
    return ik_lm(target_p, approach, base0, q0, free_base=free_base, base_nom=base_nom, zaxis=zaxis,
                 w_reg=w_reg, w_base=w_base, elbow_min=elbow_min, pen_fn=pen_fn)
