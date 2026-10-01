"""Exact PR2 left-arm kinematics (fitted to env) + fast DLS IK. numpy only."""
import math
import numpy as np

SX, SY, SZ = -0.05, 0.188, 0.990675
L_SH, L_UP, L_FO, L_TOOL = 0.1, 0.4, 0.321, 0.18
LO = np.array([-0.7146, -0.5236, -0.8, -2.3213, -np.pi, -2.0943, -np.pi])
HI = np.array([2.2854, 1.3963, 3.9, 0.0, np.pi, 0.0, np.pi])
CONT = np.array([False, False, False, False, True, False, True])
Q0 = np.array([0.6772, -0.3431, 1.2, -1.4669, 1.2422, -1.9544, 2.2225])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fk_full(base, q):
    bx, by, bt = base
    R = _rz(bt)
    t = np.array([bx, by, 0.0]) + R @ np.array([SX, SY, SZ])
    axes = np.zeros((7, 3)); orig = np.zeros((7, 3))
    shoulder = t.copy()
    axes[0] = R[:, 2]; orig[0] = t
    R = R @ _rz(q[0])
    t = t + R[:, 0] * L_SH
    upper = t.copy()
    axes[1] = R[:, 1]; orig[1] = t
    R = R @ _ry(q[1])
    axes[2] = R[:, 0]; orig[2] = t
    R = R @ _rx(q[2])
    t = t + R[:, 0] * L_UP
    elbow = t.copy()
    axes[3] = R[:, 1]; orig[3] = t
    R = R @ _ry(q[3])
    axes[4] = R[:, 0]; orig[4] = t
    R = R @ _rx(q[4])
    t = t + R[:, 0] * L_FO
    wrist = t.copy()
    axes[5] = R[:, 1]; orig[5] = t
    R = R @ _ry(q[5])
    axes[6] = R[:, 0]; orig[6] = t
    R = R @ _rx(q[6])
    tool = t + R[:, 0] * L_TOOL
    pts = (shoulder, upper, elbow, wrist, tool)
    return pts, axes, orig, tool, R


def fk(base, q):
    _, _, _, t, R = fk_full(base, q)
    return t, R


def shoulder_xy(base):
    bx, by, bt = base
    c, s = math.cos(bt), math.sin(bt)
    return np.array([bx + c * SX - s * SY, by + s * SX + c * SY])


def ik_down_slow(base, pos, q_init, yaw=None, p_off=None, iters=80, tol=1e-6):
    """Tool x-axis straight down; point p_off (tool frame) placed at pos.
    If yaw given, tool y-axis = (cos yaw, sin yaw, 0). Returns (q, ok)."""
    q = np.array(q_init, dtype=float).copy()
    q = np.where(CONT, wrap(q), np.clip(q, LO + 1e-4, HI - 1e-4))
    down = np.array([0., 0., -1.])
    P = np.eye(3) - np.outer(down, down)
    po = np.zeros(3) if p_off is None else np.asarray(p_off, float)
    lam = 1e-3
    J = np.zeros((6, 7))
    ydes = None if yaw is None else np.array([math.cos(yaw), math.sin(yaw), 0.])
    for it in range(iters):
        _, axes, orig, t, R = fk_full(base, q)
        pe = t + R @ po
        ep = pos - pe
        r0 = R[:, 0]
        ex = np.array([-r0[1], r0[0], 0.0])  # cross(r0, down)
        if ydes is not None:
            r1 = R[:, 1]
            eo = ex + _cross(r1, ydes)
        else:
            eo = ex
        D = pe - orig
        A = axes
        J[0] = A[:, 1] * D[:, 2] - A[:, 2] * D[:, 1]
        J[1] = A[:, 2] * D[:, 0] - A[:, 0] * D[:, 2]
        J[2] = A[:, 0] * D[:, 1] - A[:, 1] * D[:, 0]
        J[3:] = A.T
        if ydes is None:
            Jr = J.copy(); Jr[3:] = P @ J[3:]; eo = P @ eo
        else:
            Jr = J
        e = np.concatenate([ep, eo])
        if e @ e < tol * tol:
            break
        dq = Jr.T @ np.linalg.solve(Jr @ Jr.T + lam * np.eye(6), e)
        n = np.abs(dq).max()
        if n > 0.4:
            dq *= 0.4 / n
        q = q + dq
        q = np.where(CONT, wrap(q), np.clip(q, LO, HI))
    _, _, _, t, R = fk_full(base, q)
    pe = t + R @ po
    ok = np.linalg.norm(pos - pe) < 1e-3 and R[2, 0] < -0.9995
    if ydes is not None:
        ok = ok and (R[:, 1] @ ydes) > 0.9995
    return q, bool(ok)


# ---------------------------------------------------------------- fast path
def _fk_fast(bx, by, bt, q):
    """Pure-python FK. Returns (axes list of 7 tuples, orig list of 7 tuples, tool tuple, cols (c0,c1,c2))."""
    cb, sb = math.cos(bt), math.sin(bt)
    c0 = (cb, sb, 0.0); c1 = (-sb, cb, 0.0); c2 = (0.0, 0.0, 1.0)
    t = (bx + cb * SX - sb * SY, by + sb * SX + cb * SY, SZ)
    axes = [None] * 7; orig = [None] * 7
    axes[0] = c2; orig[0] = t
    # Rz(q0)
    c, s = math.cos(q[0]), math.sin(q[0])
    c0, c1 = (c * c0[0] + s * c1[0], c * c0[1] + s * c1[1], c * c0[2] + s * c1[2]), \
             (-s * c0[0] + c * c1[0], -s * c0[1] + c * c1[1], -s * c0[2] + c * c1[2])
    t = (t[0] + c0[0] * L_SH, t[1] + c0[1] * L_SH, t[2] + c0[2] * L_SH)
    axes[1] = c1; orig[1] = t
    # Ry(q1)
    c, s = math.cos(q[1]), math.sin(q[1])
    c0, c2 = (c * c0[0] - s * c2[0], c * c0[1] - s * c2[1], c * c0[2] - s * c2[2]), \
             (s * c0[0] + c * c2[0], s * c0[1] + c * c2[1], s * c0[2] + c * c2[2])
    axes[2] = c0; orig[2] = t
    # Rx(q2)
    c, s = math.cos(q[2]), math.sin(q[2])
    c1, c2 = (c * c1[0] + s * c2[0], c * c1[1] + s * c2[1], c * c1[2] + s * c2[2]), \
             (-s * c1[0] + c * c2[0], -s * c1[1] + c * c2[1], -s * c1[2] + c * c2[2])
    t = (t[0] + c0[0] * L_UP, t[1] + c0[1] * L_UP, t[2] + c0[2] * L_UP)
    axes[3] = c1; orig[3] = t
    # Ry(q3)
    c, s = math.cos(q[3]), math.sin(q[3])
    c0, c2 = (c * c0[0] - s * c2[0], c * c0[1] - s * c2[1], c * c0[2] - s * c2[2]), \
             (s * c0[0] + c * c2[0], s * c0[1] + c * c2[1], s * c0[2] + c * c2[2])
    axes[4] = c0; orig[4] = t
    # Rx(q4)
    c, s = math.cos(q[4]), math.sin(q[4])
    c1, c2 = (c * c1[0] + s * c2[0], c * c1[1] + s * c2[1], c * c1[2] + s * c2[2]), \
             (-s * c1[0] + c * c2[0], -s * c1[1] + c * c2[1], -s * c1[2] + c * c2[2])
    t = (t[0] + c0[0] * L_FO, t[1] + c0[1] * L_FO, t[2] + c0[2] * L_FO)
    axes[5] = c1; orig[5] = t
    # Ry(q5)
    c, s = math.cos(q[5]), math.sin(q[5])
    c0, c2 = (c * c0[0] - s * c2[0], c * c0[1] - s * c2[1], c * c0[2] - s * c2[2]), \
             (s * c0[0] + c * c2[0], s * c0[1] + c * c2[1], s * c0[2] + c * c2[2])
    axes[6] = c0; orig[6] = t
    # Rx(q6)
    c, s = math.cos(q[6]), math.sin(q[6])
    c1, c2 = (c * c1[0] + s * c2[0], c * c1[1] + s * c2[1], c * c1[2] + s * c2[2]), \
             (-s * c1[0] + c * c2[0], -s * c1[1] + c * c2[1], -s * c1[2] + c * c2[2])
    tool = (t[0] + c0[0] * L_TOOL, t[1] + c0[1] * L_TOOL, t[2] + c0[2] * L_TOOL)
    return axes, orig, tool, (c0, c1, c2)


STALL = [60, 1e-3, 0.95, 40]
_PI = math.pi
_LO = [float(v) for v in LO]
_HI = [float(v) for v in HI]
_CONT = [bool(v) for v in CONT]


def _wrapf(a):
    return (a + _PI) % (2 * _PI) - _PI


def ik_down_fast(base, pos, q_init, yaw=None, p_off=None, iters=80, tol=1e-4, stall=True):
    hist = []
    bx, by, bt = float(base[0]), float(base[1]), float(base[2])
    q = [float(v) for v in q_init]
    for i in range(7):
        q[i] = _wrapf(q[i]) if _CONT[i] else min(max(q[i], _LO[i] + 1e-4), _HI[i] - 1e-4)
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    if p_off is None:
        o0 = o1 = o2 = 0.0
    else:
        o0, o1, o2 = float(p_off[0]), float(p_off[1]), float(p_off[2])
    # reachability pre-check (tool vertical): wrist = pos + (0,0,o0+L_TOOL) +- lateral
    cb, sb = math.cos(bt), math.sin(bt)
    shx, shy = bx + cb * SX - sb * SY, by + sb * SX + cb * SY
    dh = math.hypot(px - shx, py - shy)
    dz = pz + o0 + L_TOOL - SZ
    slack = math.hypot(o1, o2) + 0.005
    if math.hypot(max(dh - L_SH - slack, 0.0), dz) > L_UP + L_FO:
        return np.array(q), False
    has_yaw = yaw is not None
    if has_yaw:
        yx, yy = math.cos(yaw), math.sin(yaw)
    lam = 1e-3
    tol2 = tol * tol
    eye = np.eye(6) * lam
    J = np.empty((6, 7))
    nrow = 6 if has_yaw else 5
    for it in range(iters):
        axes, orig, t, (c0, c1, c2) = _fk_fast(bx, by, bt, q)
        pe0 = t[0] + c0[0] * o0 + c1[0] * o1 + c2[0] * o2
        pe1 = t[1] + c0[1] * o0 + c1[1] * o1 + c2[1] * o2
        pe2 = t[2] + c0[2] * o0 + c1[2] * o1 + c2[2] * o2
        e0, e1, e2 = px - pe0, py - pe1, pz - pe2
        ex0, ex1 = -c0[1], c0[0]
        if has_yaw:
            # cross(c1, ydes) with ydes=(yx,yy,0)
            e3 = ex0 + (-c1[2] * yy)
            e4 = ex1 + (c1[2] * yx)
            e5 = c1[0] * yy - c1[1] * yx
        else:
            e3, e4, e5 = ex0, ex1, 0.0
        err2 = e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4 + e5 * e5
        if err2 < tol2:
            break
        if stall:
            hist.append(err2)
            if it >= STALL[0] and err2 > STALL[1] and err2 > STALL[2] * hist[it - STALL[3]]:
                break
        for j in range(7):
            a = axes[j]; o = orig[j]
            d0, d1, d2 = pe0 - o[0], pe1 - o[1], pe2 - o[2]
            J[0, j] = a[1] * d2 - a[2] * d1
            J[1, j] = a[2] * d0 - a[0] * d2
            J[2, j] = a[0] * d1 - a[1] * d0
            J[3, j] = a[0]; J[4, j] = a[1]; J[5, j] = a[2] if has_yaw else 0.0
        e = np.array((e0, e1, e2, e3, e4, e5))
        dq = J.T @ np.linalg.solve(J @ J.T + eye, e)
        n = float(np.abs(dq).max())
        sc = 0.4 / n if n > 0.4 else 1.0
        for i in range(7):
            v = q[i] + float(dq[i]) * sc
            q[i] = _wrapf(v) if _CONT[i] else min(max(v, _LO[i]), _HI[i])
    axes, orig, t, (c0, c1, c2) = _fk_fast(bx, by, bt, q)
    pe0 = t[0] + c0[0] * o0 + c1[0] * o1 + c2[0] * o2
    pe1 = t[1] + c0[1] * o0 + c1[1] * o1 + c2[1] * o2
    pe2 = t[2] + c0[2] * o0 + c1[2] * o1 + c2[2] * o2
    err = math.sqrt((px - pe0) ** 2 + (py - pe1) ** 2 + (pz - pe2) ** 2)
    ok = err < 1e-3 and c0[2] < -0.9995
    if has_yaw:
        ok = ok and (c1[0] * yx + c1[1] * yy) > 0.9995
    return np.array(q), bool(ok)


ik_down = ik_down_fast


def ik_refine(base, pos, q, q_pref, yaw=None, p_off=None, iters=25, gain=0.5):
    """Null-space pull of solution q toward q_pref (minimise max |dq|), keeping the task.
    Returns refined q if still valid and max|q-q_pref| improved, else q."""
    bx, by, bt = float(base[0]), float(base[1]), float(base[2])
    qa = np.array(q, float)
    qp = np.array(q_pref, float)
    po = np.zeros(3) if p_off is None else np.asarray(p_off, float)
    has_yaw = yaw is not None
    if has_yaw:
        yx, yy = math.cos(yaw), math.sin(yaw)
    J = np.empty((6, 7))
    eye = np.eye(6) * 1e-6
    lo = LO; hi = HI
    cur = qa.copy()

    def diffv(a):
        d = qp - a
        d[CONT] = (d[CONT] + np.pi) % (2 * np.pi) - np.pi
        return d
    for it in range(iters):
        axes, orig, t, (c0, c1, c2) = _fk_fast(bx, by, bt, cur)
        pe = np.array(t) + np.array(c0) * po[0] + np.array(c1) * po[1] + np.array(c2) * po[2]
        ep = np.asarray(pos, float) - pe
        e3, e4 = -c0[1], c0[0]
        if has_yaw:
            e3 += -c1[2] * yy; e4 += c1[2] * yx
            e5 = c1[0] * yy - c1[1] * yx
        else:
            e5 = 0.0
        e = np.array((ep[0], ep[1], ep[2], e3, e4, e5))
        for j in range(7):
            a = axes[j]; o = orig[j]
            d0, d1, d2 = pe[0] - o[0], pe[1] - o[1], pe[2] - o[2]
            J[0, j] = a[1] * d2 - a[2] * d1
            J[1, j] = a[2] * d0 - a[0] * d2
            J[2, j] = a[0] * d1 - a[1] * d0
            J[3, j] = a[0]; J[4, j] = a[1]; J[5, j] = a[2] if has_yaw else 0.0
        d = diffv(cur)
        w = np.abs(d) ** 2 + 1e-6
        z = gain * d * (w / w.max())
        dq = J.T @ np.linalg.solve(J @ J.T + eye, e - J @ z) + z
        m = np.abs(dq).max()
        if m > 0.2:
            dq *= 0.2 / m
        cur = cur + dq
        cur = np.where(CONT, (cur + np.pi) % (2 * np.pi) - np.pi, np.clip(cur, lo, hi))
    # validate
    axes, orig, t, (c0, c1, c2) = _fk_fast(bx, by, bt, cur)
    pe = np.array(t) + np.array(c0) * po[0] + np.array(c1) * po[1] + np.array(c2) * po[2]
    ok = np.linalg.norm(np.asarray(pos, float) - pe) < 1e-3 and c0[2] < -0.9995
    if has_yaw:
        ok = ok and (c1[0] * yx + c1[1] * yy) > 0.9995
    if ok and np.abs(diffv(cur)).max() < np.abs(diffv(qa)).max() - 1e-6:
        return cur
    return qa
