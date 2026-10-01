"""Pure-python fast FK + analytic-Jacobian LM IK for the PR2 left arm."""
import math
import numpy as np

TORSO_ZT = 0.790675 + 0.20
TOOL_LEN = 0.18
LINKS = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.0), (0.4, 0.0), (0.0, 0.0), (0.321, 0.0), (0.0, 0.0)]
AXT = [2, 1, 0, 1, 0, 1, 0]  # z y x y x y x
JLOW = [-0.7146, -0.5236, -0.65, -2.1213, -1e3, -2.0, -1e3]
JHIGH = [2.2854, 1.3963, 3.75, -0.15, 1e3, 0.0, 1e3]
PI = math.pi


def _wrap(a):
    return (a + PI) % (2 * PI) - PI


def fk(b, q):
    """Returns tool p (list), cols (c0,c1,c2), axes, origins, elbow."""
    cy, sy = math.cos(b[2]), math.sin(b[2])
    c0 = [cy, sy, 0.0]; c1 = [-sy, cy, 0.0]; c2 = [0.0, 0.0, 1.0]
    p = [b[0] + cy * -0.05 - sy * 0.188, b[1] + sy * -0.05 + cy * 0.188, TORSO_ZT]
    axes = []; origins = []
    elbow = None
    for i in range(7):
        lx = LINKS[i][0]
        if lx:
            p = [p[0] + c0[0] * lx, p[1] + c0[1] * lx, p[2] + c0[2] * lx]
        if i == 3:
            elbow = p
        t = AXT[i]
        axes.append((c0, c1, c2)[t]); origins.append(p)
        c, s = math.cos(q[i]), math.sin(q[i])
        if t == 0:
            n1 = [c * c1[k] + s * c2[k] for k in range(3)]
            n2 = [-s * c1[k] + c * c2[k] for k in range(3)]
            c1, c2 = n1, n2
        elif t == 1:
            n0 = [c * c0[k] - s * c2[k] for k in range(3)]
            n2 = [s * c0[k] + c * c2[k] for k in range(3)]
            c0, c2 = n0, n2
        else:
            n0 = [c * c0[k] + s * c1[k] for k in range(3)]
            n1 = [-s * c0[k] + c * c1[k] for k in range(3)]
            c0, c1 = n0, n1
    p = [p[0] + c0[0] * TOOL_LEN, p[1] + c0[1] * TOOL_LEN, p[2] + c0[2] * TOOL_LEN]
    return p, (c0, c1, c2), axes, origins, elbow


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def ik_lm(target_p, approach, base0, q0, free_base=False, base_nom=None, zaxis=None,
          w_reg=1e-3, w_base=1e-2, elbow_min=None, pen_fn=None, iters=150, xlim=4.97):
    d = np.asarray(approach, float); d = d / np.linalg.norm(d)
    zt = (0.0, 0.0, 1.0) if zaxis is None else tuple(float(v) for v in zaxis)
    tp = [float(v) for v in target_p]
    dl = [float(v) for v in d]
    x = [float(v) for v in base0] + [float(v) for v in q0]
    bn = list(x[:3]) if base_nom is None else [float(v) for v in base_nom]
    q0r = list(x[3:])
    lo = [-xlim, -1e3, -1e3] + JLOW
    hi = [xlim, 1e3, 1e3] + JHIGH
    for k in range(10):
        x[k] = min(max(x[k], lo[k] + 1e-6), hi[k] - 1e-6)
    act = list(range(10)) if free_base else list(range(3, 10))

    def evaluate(x):
        b = x[:3]; q = x[3:]
        p, (c0, c1, c2), axes, origins, elbow = fk(b, q)
        r = [p[0] - tp[0], p[1] - tp[1], p[2] - tp[2],
             0.5 * (c0[0] - dl[0]), 0.5 * (c0[1] - dl[1]), 0.5 * (c0[2] - dl[2]),
             0.5 * (c2[0] - zt[0]), 0.5 * (c2[1] - zt[1]), 0.5 * (c2[2] - zt[2])]
        J = [[0.0] * 10 for _ in range(9)]
        for i in range(7):
            a = axes[i]; o = origins[i]
            v = _cross(a, (p[0] - o[0], p[1] - o[1], p[2] - o[2]))
            u = _cross(a, c0); w = _cross(a, c2)
            col = 3 + i
            J[0][col] = v[0]; J[1][col] = v[1]; J[2][col] = v[2]
            J[3][col] = 0.5 * u[0]; J[4][col] = 0.5 * u[1]; J[5][col] = 0.5 * u[2]
            J[6][col] = 0.5 * w[0]; J[7][col] = 0.5 * w[1]; J[8][col] = 0.5 * w[2]
        for i in range(7):
            r.append(w_reg * (q[i] - q0r[i]))
            row = [0.0] * 10; row[3 + i] = w_reg; J.append(row)
        if free_base:
            J[0][0] = 1.0; J[1][1] = 1.0
            # yaw: rotation about world z through base origin
            J[0][2] = -(p[1] - b[1]); J[1][2] = (p[0] - b[0])
            J[3][2] = -0.5 * c0[1]; J[4][2] = 0.5 * c0[0]
            J[6][2] = -0.5 * c2[1]; J[7][2] = 0.5 * c2[0]
            dyaw = _wrap(b[2] - bn[2])
            for k, val in enumerate((b[0] - bn[0], b[1] - bn[1], dyaw)):
                r.append(w_base * val)
                row = [0.0] * 10; row[k] = w_base; J.append(row)
            if pen_fn is not None:
                pen = pen_fn(b)
                row = [0.0] * 10
                if pen > 0:
                    for k in range(3):
                        bb = list(b); bb[k] += 1e-4
                        row[k] = 5.0 * (pen_fn(bb) - pen) / 1e-4
                r.append(5.0 * pen); J.append(row)
        if elbow_min is not None:
            v = elbow_min - elbow[2]
            row = [0.0] * 10
            if v > 0:
                for i in range(3):
                    a = axes[i]; o = origins[i]
                    row[3 + i] = -(a[0] * (elbow[1] - o[1]) - a[1] * (elbow[0] - o[0]))
            r.append(max(0.0, v)); J.append(row)
        return r, J

    r, J = evaluate(x)
    cost = sum(v * v for v in r)
    lam = 1e-3
    for it in range(iters):
        Ja = np.array(J)[:, act]
        rv = np.array(r)
        H = Ja.T @ Ja
        g = Ja.T @ rv
        try:
            dx = -np.linalg.solve(H + lam * np.eye(len(act)), g)
        except np.linalg.LinAlgError:
            break
        xn = list(x)
        for k, j in enumerate(act):
            xn[j] = min(max(x[j] + float(dx[k]), lo[j]), hi[j])
        rn, Jn = evaluate(xn)
        cn = sum(v * v for v in rn)
        if cn < cost:
            step = max(abs(xn[j] - x[j]) for j in act)
            x, r, J, cost = xn, rn, Jn, cn
            lam = max(lam * 0.3, 1e-9)
            if step < 2e-5:
                break
        else:
            lam *= 10.0
            if lam > 1e7:
                break
    b = np.array(x[:3]); q = np.array(x[3:])
    q[4] = _wrap(q[4]); q[6] = _wrap(q[6])
    p, (c0, c1, c2), _, _, _ = fk(b, q)
    err = math.sqrt(sum((p[k] - tp[k]) ** 2 for k in range(3))) + \
        math.sqrt(sum((c0[k] - dl[k]) ** 2 for k in range(3))) + \
        math.sqrt(sum((c2[k] - zt[k]) ** 2 for k in range(3)))
    return b, q, err
