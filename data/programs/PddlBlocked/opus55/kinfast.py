"""Fast IK with analytic Jacobian (LM)."""
import numpy as np
from kin import rz, ry, rx, TORSO_Z, TOOL_LEN, JLOW, JHIGH, wrap, rect_pen

TORSO = 0.20
OFFS = [np.array([0.1, 0, 0]), np.zeros(3), np.array([0.4, 0, 0]), np.zeros(3), np.array([0.321, 0, 0]), np.zeros(3)]
AX = ['z', 'y', 'x', 'y', 'x', 'y', 'x']
ROT = {'x': rx, 'y': ry, 'z': rz}
EVEC = {'x': 0, 'y': 1, 'z': 2}


def fk_full(b, q):
    """Returns tool p, R, joint axes (world), joint origins (world), elbow point."""
    Rb = rz(b[2]); tb = np.array([b[0], b[1], 0.0])
    p = Rb @ np.array([-0.05, 0.188, TORSO_Z + TORSO]) + tb
    R = Rb
    axes = []; origins = []
    elbow = None
    # joint 1 at shoulder pan (origin p)
    links = [np.zeros(3), np.array([0.1, 0, 0]), np.zeros(3), np.array([0.4, 0, 0]), np.zeros(3), np.array([0.321, 0, 0]), np.zeros(3)]
    for i in range(7):
        p = p + R @ links[i]
        if i == 3:
            elbow = p.copy()
        axes.append(R[:, EVEC[AX[i]]].copy()); origins.append(p.copy())
        R = R @ ROT[AX[i]](q[i])
    p = p + R @ np.array([TOOL_LEN, 0, 0])
    return p, R, axes, origins, elbow


def ik_fast(target_p, approach, base0, q0, free_base=False, base_nom=None, zaxis=None,
            w_reg=1e-3, w_base=1e-2, elbow_min=None, table=None, base_half=0.38, iters=80):
    d = np.asarray(approach, float); d = d / np.linalg.norm(d)
    zt = np.array([0, 0, 1.0]) if zaxis is None else np.asarray(zaxis, float)
    tp = np.asarray(target_p, float)
    b = np.array(base0, float); q = np.array(q0, float)
    bn = b.copy() if base_nom is None else np.asarray(base_nom, float)
    q0r = q.copy()
    nv = 10 if free_base else 7
    lo = np.r_[[-1e3] * 3, JLOW + 1e-4]; hi = np.r_[[1e3] * 3, JHIGH - 1e-4]
    lo = np.where(np.isinf(lo), -1e3, lo); hi = np.where(np.isinf(hi), 1e3, hi)
    x = np.clip(np.r_[b, q], lo, hi)

    def resjac(x):
        b, q = x[:3], x[3:]
        p, R, axes, origins, elbow = fk_full(b, q)
        r = [p - tp, 0.5 * (R[:, 0] - d), 0.5 * (R[:, 2] - zt), w_reg * (q - q0r)]
        J = np.zeros((12, 10))
        for i in range(7):
            a = axes[i]
            J[0:3, 3 + i] = np.cross(a, p - origins[i])
            J[3:6, 3 + i] = 0.5 * np.cross(a, R[:, 0])
            J[6:9, 3 + i] = 0.5 * np.cross(a, R[:, 2])
        J[9:12, :] = 0  # placeholder rows not used
        Jl = [J[0:9]]
        Jreg = np.zeros((7, 10)); Jreg[:, 3:] = np.eye(7) * w_reg
        Jl.append(Jreg)
        if free_base:
            ez = np.array([0, 0, 1.0]); ob = np.array([b[0], b[1], 0.0])
            Jb = np.zeros((9, 3))
            Jb[0:3, 0] = [1, 0, 0]; Jb[0:3, 1] = [0, 1, 0]
            Jb[0:3, 2] = np.cross(ez, p - ob)
            Jb[3:6, 2] = 0.5 * np.cross(ez, R[:, 0]); Jb[6:9, 2] = 0.5 * np.cross(ez, R[:, 2])
            Jl[0][:, 0:3] = Jb
            r.append(w_base * np.r_[b[:2] - bn[:2], wrap(b[2] - bn[2])])
            Jbn = np.zeros((3, 10)); Jbn[:, 0:3] = np.eye(3) * w_base
            Jl.append(Jbn)
            if table is not None:
                pen = rect_pen(b, base_half, table)
                r.append([5.0 * pen])
                g = np.zeros(10)
                if pen > 0:
                    for k in range(3):
                        e = np.zeros(3); e[k] = 1e-4
                        g[k] = 5.0 * (rect_pen(b + e, base_half, table) - pen) / 1e-4
                Jl.append(g[None, :])
        if elbow_min is not None:
            v = elbow_min - elbow[2]
            g = np.zeros(10)
            if v > 0:
                for i in range(3):
                    g[3 + i] = -np.cross(axes[i], elbow - origins[i])[2]
                if free_base:
                    pass
            r.append([max(0.0, v)])
            Jl.append(g[None, :])
        return np.concatenate([np.ravel(a) for a in r]), np.vstack(Jl)

    lam = 1e-3
    r, J = resjac(x)
    cost = r @ r
    act = slice(0, 10) if free_base else slice(3, 10)
    for it in range(iters):
        Ja = J[:, act]
        H = Ja.T @ Ja; g = Ja.T @ r
        dx = -np.linalg.solve(H + lam * np.eye(H.shape[0]), g)
        xn = x.copy(); xn[act] = np.clip(x[act] + dx, lo[act], hi[act])
        rn, Jn = resjac(xn)
        cn = rn @ rn
        if cn < cost:
            x, r, J, cost = xn, rn, Jn, cn
            lam = max(lam * 0.3, 1e-7)
            if np.abs(dx).max() < 1e-7:
                break
        else:
            lam *= 10
            if lam > 1e6:
                break
    b, q = x[:3], x[3:].copy()
    q[4] = wrap(q[4]); q[6] = wrap(q[6])
    p, R, _, _, _ = fk_full(b, q)
    err = np.linalg.norm(p - tp) + np.linalg.norm(R[:, 0] - d) + np.linalg.norm(R[:, 2] - zt)
    return np.array(b), q, err
