"""Approach for ClutteredRetrieval2D (variable obstruction count).

Plans in robot configuration space (x, y, theta, arm_joint) with a fast custom
collision checker and RRT-Connect, using a small state machine:
  grasp target -> carry to region -> release; if target grasp is infeasible,
  remove blocking obstructions first (grasp, carry away, release).
"""

import math
import time

import numpy as np
from scipy import ndimage
import os
DEBUG = bool(os.environ.get('APDEBUG'))

WORLD_MIN, WORLD_MAX = 0.0, 2.5
LIM = np.array([0.05, 0.05, np.pi / 16, 0.1])
STEP = LIM * 0.995
ARM_MIN, ARM_MAX = 0.1, 0.2
MARGIN = 0.003
GRASP_GAP = 0.007  # gap between gripper front face and object face


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- geometry
def rect_verts(x, y, th, w, h):
    c, s = math.cos(th), math.sin(th)
    return np.array([[x, y], [x + c * w, y + s * w],
                     [x + c * w - s * h, y + s * w + c * h], [x - s * h, y + c * h]])


def rect_from_center(cx, cy, w, h, th):
    c, s = math.cos(th), math.sin(th)
    hw, hh = w / 2, h / 2
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([cx, cy])


class Obstacles:
    """Vectorized set of rectangle obstacles."""

    def __init__(self, verts_list, names):
        self.names = list(names)
        if len(verts_list) == 0:
            self.V = np.zeros((0, 4, 2))
        else:
            self.V = np.array(verts_list)
        self.n = len(self.V)
        if self.n:
            self.C = self.V.mean(axis=1)
            self.R = np.linalg.norm(self.V - self.C[:, None, :], axis=2).max(axis=1)
            e1 = self.V[:, 1] - self.V[:, 0]
            e2 = self.V[:, 3] - self.V[:, 0]
            self.A = np.stack([e1 / np.linalg.norm(e1, axis=1)[:, None],
                               e2 / np.linalg.norm(e2, axis=1)[:, None]], axis=1)  # (n,2,2)
            # local frame projections of own verts
            self.own = np.einsum('nkd,nvd->nkv', self.A, self.V)  # (n,2,4)
            self.omin = self.own.min(axis=2)
            self.omax = self.own.max(axis=2)
            self.origin = self.V[:, 0]
            self.len = np.stack([np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)], 1)

    def rect_hits(self, P, margin=MARGIN):
        """Boolean array of obstacles colliding with polygon P (4,2)."""
        if self.n == 0:
            return np.zeros(0, bool)
        pc = P.mean(axis=0)
        pr = np.linalg.norm(P - pc, axis=1).max()
        d = np.linalg.norm(self.C - pc, axis=1)
        cand = d < self.R + pr + margin
        if not cand.any():
            return cand
        idx = np.nonzero(cand)[0]
        hit = np.ones(len(idx), bool)
        # obstacle axes
        A = self.A[idx]  # (k,2,2)
        proj = np.einsum('kad,vd->kav', A, P)  # (k,2,4)
        pmin, pmax = proj.min(axis=2), proj.max(axis=2)
        omin, omax = self.omin[idx], self.omax[idx]
        sep = (pmin > omax + margin) | (pmax < omin - margin)
        hit &= ~sep.any(axis=1)
        # polygon axes
        e1 = P[1] - P[0]
        e2 = P[3] - P[0]
        ax = np.array([e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)])
        pp = P @ ax.T  # (4,2)
        ppmin, ppmax = pp.min(axis=0), pp.max(axis=0)
        op = np.einsum('kvd,ad->kav', self.V[idx], ax)  # (k,2,4)
        sep2 = (op.min(axis=2) > ppmax + margin) | (op.max(axis=2) < ppmin - margin)
        hit &= ~sep2.any(axis=1)
        out = np.zeros(self.n, bool)
        out[idx] = hit
        return out

    def circle_hits(self, cx, cy, r, margin=MARGIN):
        if self.n == 0:
            return np.zeros(0, bool)
        p = np.array([cx, cy])
        d = np.linalg.norm(self.C - p, axis=1)
        cand = d < self.R + r + margin
        if not cand.any():
            return cand
        idx = np.nonzero(cand)[0]
        rel = p - self.origin[idx]
        loc = np.einsum('kad,kd->ka', self.A[idx], rel)
        cl = np.clip(loc, 0, self.len[idx])
        dist = np.linalg.norm(loc - cl, axis=1)
        out = np.zeros(self.n, bool)
        out[idx] = dist < r + margin
        return out


def in_world(P, margin=MARGIN):
    return (P.min() >= WORLD_MIN + margin) and (P.max() <= WORLD_MAX - margin)


# ---------------------------------------------------------------- robot model
class Robot:
    def __init__(self, base_r, gw, gh):
        self.r = base_r
        self.gw = gw
        self.gh = gh

    def bodies(self, q):
        x, y, th, aj = q
        c, s = math.cos(th), math.sin(th)
        gx, gy = x + c * aj, y + s * aj
        grip = rect_from_center(gx, gy, self.gw, self.gh, th)
        arm = rect_from_center((x + gx) / 2, (y + gy) / 2, aj, 0.5 * self.gw, th)
        return grip, arm

    def suction(self, q):
        x, y, th, aj = q
        c, s = math.cos(th), math.sin(th)
        d = aj + self.gw + self.gw / 2
        return rect_from_center(x + c * d, y + s * d, self.gw, self.gh, th)

    def tooltip(self, q):
        x, y, th, aj = q
        d = aj + self.gw / 2
        return np.array([x + math.cos(th) * d, y + math.sin(th) * d, th])


def se2_mul(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def se2_inv(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap(-a[2])])


class Checker:
    def __init__(self, robot, obstacles, held=(), extra_suction_check=False):
        """held: list of (rel_pose (3,), w, h) of rects held relative to tooltip."""
        self.robot = robot
        self.obs = obstacles
        self.held = list(held)
        self.suction_check = extra_suction_check

    def held_polys(self, q):
        tt = self.robot.tooltip(q)
        out = []
        for rel, w, h in self.held:
            p = se2_mul(tt, rel)
            out.append(rect_verts(p[0], p[1], p[2], w, h))
        return out

    def free(self, q, margin=MARGIN):
        return fast_free(self, q, margin)

    def free_scalar(self, q, margin=MARGIN):
        x, y, th, aj = q
        r = self.robot.r
        if x - r < WORLD_MIN + margin or x + r > WORLD_MAX - margin:
            return False
        if y - r < WORLD_MIN + margin or y + r > WORLD_MAX - margin:
            return False
        if self.obs.circle_hits(x, y, r, margin).any():
            return False
        polys = list(self.robot.bodies(q)) + self.held_polys(q)
        if self.suction_check and self.held:
            polys.append(self.robot.suction(q))
        for P in polys:
            if not in_world(P, margin):
                return False
            if self.obs.rect_hits(P, margin).any():
                return False
        return True

    def colliders(self, q, margin=MARGIN):
        """Indices of obstacles colliding with config q (and world violation flag)."""
        x, y, th, aj = q
        hits = self.obs.circle_hits(x, y, self.robot.r, margin).copy()
        ok_world = (x - self.robot.r >= WORLD_MIN + margin and x + self.robot.r <= WORLD_MAX - margin
                    and y - self.robot.r >= WORLD_MIN + margin and y + self.robot.r <= WORLD_MAX - margin)
        for P in list(self.robot.bodies(q)) + self.held_polys(q):
            if not in_world(P, margin):
                ok_world = False
            hits |= self.obs.rect_hits(P, margin)
        return hits, ok_world


# ---------------------------------------------------------------- batch checking
def _poly_templates(robot, held, suction):
    """Local polygons as A + aj * B, A,B: (k,4,2)."""
    gw, gh = robot.gw, robot.gh
    As, Bs = [], []
    def box(x0, x1, y0, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    onesx = np.array([[1.0, 0.0]] * 4)
    As.append(box(-gw / 2, gw / 2, -gh / 2, gh / 2)); Bs.append(onesx)
    As.append(box(0, 0, -gw / 4, gw / 4)); Bs.append(np.array([[0, 0], [1, 0], [1, 0], [0, 0]], float))
    if suction and held:
        As.append(box(gw, 2 * gw, -gh / 2, gh / 2)); Bs.append(onesx)
    for rel, w, h in held:
        V = rect_verts(rel[0], rel[1], rel[2], w, h).copy()
        V[:, 0] += gw / 2
        As.append(V); Bs.append(onesx)
    return np.array(As), np.array(Bs)


def batch_free(chk, Q, margin=MARGIN):
    Q = np.atleast_2d(np.asarray(Q, float))
    m = len(Q)
    x, y, th, aj = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    r = chk.robot.r
    ok = (x - r >= WORLD_MIN + margin) & (x + r <= WORLD_MAX - margin) & \
         (y - r >= WORLD_MIN + margin) & (y + r <= WORLD_MAX - margin)
    if getattr(chk, '_tpl', None) is None:
        chk._tpl = _poly_templates(chk.robot, chk.held, chk.suction_check)
    TA, TB = chk._tpl
    L = TA[None] + aj[:, None, None, None] * TB[None]  # (m,k,4,2)
    c, s = np.cos(th), np.sin(th)
    W = np.empty_like(L)
    W[..., 0] = c[:, None, None] * L[..., 0] - s[:, None, None] * L[..., 1] + x[:, None, None]
    W[..., 1] = s[:, None, None] * L[..., 0] + c[:, None, None] * L[..., 1] + y[:, None, None]
    ok &= (W.reshape(m, -1).min(axis=1) >= WORLD_MIN + margin) & (W.reshape(m, -1).max(axis=1) <= WORLD_MAX - margin)
    ob = chk.obs
    if ob.n == 0:
        return ok
    # circle
    rel = np.stack([x, y], 1)[:, None, :] - ob.origin[None]  # (m,n,2)
    loc = np.matmul(rel[:, :, None, :], ob.A.transpose(0, 2, 1)[None])[:, :, 0, :]
    cl = np.clip(loc, 0, ob.len[None])
    dist = np.linalg.norm(loc - cl, axis=2)
    ok &= ~(dist < r + margin).any(axis=1)
    # polys SAT (broadphase on bounding circles, then SAT on candidate pairs)
    k = W.shape[1]
    P = W.reshape(m * k, 4, 2)
    pc = P.mean(axis=1)
    pr = np.sqrt(((P - pc[:, None]) ** 2).sum(-1)).max(axis=1)
    dx = pc[:, None, 0] - ob.C[None, :, 0]
    dy = pc[:, None, 1] - ob.C[None, :, 1]
    lim = pr[:, None] + ob.R[None] + margin
    cand = dx * dx + dy * dy < lim * lim
    hit = np.zeros((m * k, ob.n), bool)
    if cand.any():
        Mi, Ni = np.nonzero(cand)
        Pp = P[Mi]  # (q,4,2)
        Ao = ob.A[Ni]  # (q,2,2)
        proj = np.matmul(Pp, Ao.transpose(0, 2, 1))  # (q,4,2)
        sep = (proj.min(axis=1) > ob.omax[Ni] + margin) | (proj.max(axis=1) < ob.omin[Ni] - margin)
        e1 = Pp[:, 1] - Pp[:, 0]
        e2 = Pp[:, 3] - Pp[:, 0]
        e1 = e1 / np.maximum(np.sqrt((e1 ** 2).sum(-1)), 1e-12)[:, None]
        e2 = e2 / np.maximum(np.sqrt((e2 ** 2).sum(-1)), 1e-12)[:, None]
        Ap = np.stack([e1, e2], 2)  # (q,2(d),2(a))
        pp = np.matmul(Pp, Ap)  # (q,4,2)
        op = np.matmul(ob.V[Ni], Ap)
        sep2 = (op.min(axis=1) > pp.max(axis=1) + margin) | (op.max(axis=1) < pp.min(axis=1) - margin)
        h = ~(sep.any(axis=1) | sep2.any(axis=1))
        hit[Mi, Ni] = h
    ok &= ~hit.reshape(m, k * ob.n).any(axis=1)
    return ok


def _prep_fast(chk):
    ob = chk.obs
    lst = []
    for i in range(ob.n):
        V = ob.V[i]
        A = ob.A[i]
        lst.append((float(ob.C[i, 0]), float(ob.C[i, 1]), float(ob.R[i]),
                    float(V[0, 0]), float(V[0, 1]),
                    float(A[0, 0]), float(A[0, 1]), float(A[1, 0]), float(A[1, 1]),
                    float(ob.len[i, 0]), float(ob.len[i, 1]),
                    tuple((float(v[0]), float(v[1])) for v in V)))
    TA, TB = _poly_templates(chk.robot, chk.held, chk.suction_check)
    tpl = [([(float(a[0]), float(a[1])) for a in TA[j]], [(float(b[0]), float(b[1])) for b in TB[j]])
           for j in range(len(TA))]
    chk._fast = (lst, tpl)


def fast_free(chk, q, margin=MARGIN):
    if getattr(chk, '_fast', None) is None:
        _prep_fast(chk)
    lst, tpl = chk._fast
    x, y, th, aj = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    r = chk.robot.r
    lo, hi = WORLD_MIN + margin, WORLD_MAX - margin
    if x - r < lo or x + r > hi or y - r < lo or y + r > hi:
        return False
    rr = r + margin
    for (cx, cy, R, ox, oy, a1x, a1y, a2x, a2y, l1, l2, V) in lst:
        dx, dy = x - cx, y - cy
        lim = R + rr
        if dx * dx + dy * dy >= lim * lim:
            continue
        px, py = x - ox, y - oy
        u = px * a1x + py * a1y
        v = px * a2x + py * a2y
        cu = 0.0 if u < 0 else (l1 if u > l1 else u)
        cv = 0.0 if v < 0 else (l2 if v > l2 else v)
        if (u - cu) ** 2 + (v - cv) ** 2 < rr * rr:
            return False
    c, s = math.cos(th), math.sin(th)
    for TA, TB in tpl:
        P = []
        sx = sy = 0.0
        for (ax, ay), (bx, by) in zip(TA, TB):
            lx = ax + aj * bx
            ly = ay + aj * by
            wx = c * lx - s * ly + x
            wy = s * lx + c * ly + y
            if wx < lo or wx > hi or wy < lo or wy > hi:
                return False
            P.append((wx, wy))
            sx += wx
            sy += wy
        pcx, pcy = sx / 4, sy / 4
        pr = 0.0
        for (wx, wy) in P:
            d = (wx - pcx) ** 2 + (wy - pcy) ** 2
            if d > pr:
                pr = d
        pr = math.sqrt(pr)
        # poly axes
        e1x, e1y = P[1][0] - P[0][0], P[1][1] - P[0][1]
        e2x, e2y = P[3][0] - P[0][0], P[3][1] - P[0][1]
        n1 = math.hypot(e1x, e1y) or 1e-12
        n2 = math.hypot(e2x, e2y) or 1e-12
        e1x, e1y, e2x, e2y = e1x / n1, e1y / n1, e2x / n2, e2y / n2
        pp1 = [wx * e1x + wy * e1y for (wx, wy) in P]
        pp2 = [wx * e2x + wy * e2y for (wx, wy) in P]
        p1min, p1max, p2min, p2max = min(pp1), max(pp1), min(pp2), max(pp2)
        for (cx, cy, R, ox, oy, a1x, a1y, a2x, a2y, l1, l2, V) in lst:
            dx, dy = pcx - cx, pcy - cy
            lim = R + pr + margin
            if dx * dx + dy * dy >= lim * lim:
                continue
            # obstacle axes (obstacle spans [o, o+l] along its own axes)
            o1 = ox * a1x + oy * a1y
            pr1 = [wx * a1x + wy * a1y for (wx, wy) in P]
            if min(pr1) > o1 + l1 + margin or max(pr1) < o1 - margin:
                continue
            o2 = ox * a2x + oy * a2y
            pr2 = [wx * a2x + wy * a2y for (wx, wy) in P]
            if min(pr2) > o2 + l2 + margin or max(pr2) < o2 - margin:
                continue
            q1 = [vx * e1x + vy * e1y for (vx, vy) in V]
            if min(q1) > p1max + margin or max(q1) < p1min - margin:
                continue
            q2 = [vx * e2x + vy * e2y for (vx, vy) in V]
            if min(q2) > p2max + margin or max(q2) < p2min - margin:
                continue
            return False
    return True


def free_prefix(chk, Q, margin=MARGIN):
    """Number of leading configs in Q that are free."""
    for i in range(len(Q)):
        if not fast_free(chk, Q[i], margin):
            return i
    return len(Q)


# ---------------------------------------------------------------- planning
def cfg_diff(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    d[2] = wrap(d[2])
    return d


def nsteps(a, b):
    d = np.abs(cfg_diff(a, b)) / STEP
    return int(math.ceil(d.max() - 1e-9))


def interp(a, b):
    """Waypoints from a (exclusive) to b (inclusive) with per-step limits."""
    n = max(1, nsteps(a, b))
    d = cfg_diff(a, b)
    out = []
    for i in range(1, n + 1):
        q = np.asarray(a, float) + d * (i / n)
        q[2] = wrap(q[2])
        out.append(q)
    out[-1] = np.array(b, float)
    out[-1][2] = wrap(out[-1][2])
    return out


def interp_arr(a, b):
    n = max(1, nsteps(a, b))
    d = cfg_diff(a, b)
    Q = np.asarray(a, float)[None] + d[None] * (np.arange(1, n + 1) / n)[:, None]
    Q[-1] = b
    Q[:, 2] = wrap(Q[:, 2])
    return Q


def edge_free(chk, a, b):
    Q = interp_arr(a, b)
    return free_prefix(chk, Q) == len(Q)


def wdist(A, q):
    d = A - q
    d[:, 2] = wrap(d[:, 2])
    d = d / STEP
    return np.sqrt((d ** 2).sum(axis=1))


def base_reach_labels(obs, r, res=0.01, W=2.5):
    n = int(round(W / res))
    xs = (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    reff = r - res * 0.75
    free = (X > reff) & (X < W - reff) & (Y > reff) & (Y < W - reff)
    for k in range(obs.n):
        c = obs.C[k]; R = obs.R[k]
        i0 = max(0, int((c[0] - R - r) / res) - 1); i1 = min(n, int((c[0] + R + r) / res) + 2)
        j0 = max(0, int((c[1] - R - r) / res) - 1); j1 = min(n, int((c[1] + R + r) / res) + 2)
        if i0 >= i1 or j0 >= j1: continue
        px = X[i0:i1, j0:j1] - obs.origin[k][0]; py = Y[i0:i1, j0:j1] - obs.origin[k][1]
        u = obs.A[k][0][0] * px + obs.A[k][0][1] * py
        v = obs.A[k][1][0] * px + obs.A[k][1][1] * py
        du = np.maximum(0, np.maximum(-u, u - obs.len[k][0]))
        dv = np.maximum(0, np.maximum(-v, v - obs.len[k][1]))
        free[i0:i1, j0:j1] &= (du * du + dv * dv) >= reff * reff
    lab, _ = ndimage.label(free, structure=np.ones((3, 3)))
    return lab, res
def cell(lab, res, p):
    n = lab.shape[0]
    i = min(n - 1, max(0, int(p[0] / res))); j = min(n - 1, max(0, int(p[1] / res)))
    return lab[i, j]


LAST_FAIL = [None]


def rrt_connect(chk, start, goals, rng, time_limit=2.0, max_nodes=8000):
    start = np.array(start, float)
    goals = [np.array(g, float) for g in goals]
    t0 = time.time()
    cap = max_nodes + 200 + len(goals)
    trees = []
    for roots in ([start], goals):
        N = np.zeros((cap, 4))
        N[:len(roots)] = roots
        trees.append({'N': N, 'n': len(roots), 'par': [-1] * len(roots), 'start': roots is not goals})
    ext = 8.0
    pool = []

    def sample():
        while not pool:
            m = 64
            S = np.stack([rng.uniform(0.1, 2.4, m), rng.uniform(0.1, 2.4, m),
                          rng.uniform(-np.pi, np.pi, m),
                          np.where(rng.random(m) < 0.5, ARM_MIN, rng.uniform(ARM_MIN, ARM_MAX, m))], 1)
            pool.extend([q for q in S if fast_free(chk, q)])
        return pool.pop()

    def extend(tree, q, full):
        N, n = tree['N'], tree['n']
        dd = wdist(N[:n], q)
        i = int(np.argmin(dd))
        cur = N[i]
        dist = dd[i]
        if dist < 1e-9:
            return i, True
        target = q
        if not full and dist > ext:
            target = cur + cfg_diff(cur, q) * (ext / dist)
            target[2] = wrap(target[2])
        Q = interp_arr(cur, target)
        nf = free_prefix(chk, Q)
        nf = min(nf, cap - n)
        if nf > 0:
            N[n:n + nf] = Q[:nf]
            tree['par'].extend([i] + list(range(n, n + nf - 1)))
            tree['n'] = n + nf
            last = n + nf - 1
        else:
            last = i
        reached = (nf == len(Q)) and (target is q)
        return last, reached

    while time.time() - t0 < time_limit and trees[0]['n'] + trees[1]['n'] < max_nodes:
        q = sample()
        ia, _ = extend(trees[0], q, False)
        qa = trees[0]['N'][ia].copy()
        ib, ok = extend(trees[1], qa, True)
        if ok:
            def trace(tree, i):
                out = []
                while i != -1:
                    out.append(tree['N'][i].copy())
                    i = tree['par'][i]
                return out
            pa = trace(trees[0], ia)[::-1]
            pb = trace(trees[1], ib)
            path = pa + pb[1:]
            if not trees[0]['start']:
                path = path[::-1]
            return path
        trees.reverse()
    for t in trees:
        if t['start']:
            LAST_FAIL[0] = (t['N'][:t['n']].copy(), list(t['par']))
    return None


def escape(chk, start, rng, tries=300):
    """Find a short sequence of steps (valid with zero margin) to a config
    that is free with the normal margin."""
    start = np.array(start, float)
    best = None
    for k in range(tries):
        d = rng.uniform(-1, 1, size=4) * STEP
        if k < 40:
            d[2] *= 0.2
        seq = []
        q = start.copy()
        for _ in range(4):
            q = q + d
            q[2] = wrap(q[2])
            q[3] = min(max(q[3], ARM_MIN), ARM_MAX)
            if not chk.free(q, margin=0.0):
                break
            seq.append(q.copy())
            if chk.free(q):
                if best is None or len(seq) < len(best):
                    best = list(seq)
                break
        if best is not None and len(best) == 1:
            break
    return best


def shortcut(chk, path, rng, iters=60):
    path = [np.array(p, float) for p in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i = rng.integers(0, len(path) - 1)
        j = rng.integers(0, len(path) - 1)
        if abs(i - j) < 2:
            continue
        i, j = min(i, j), max(i, j)
        if edge_free(chk, path[i], path[j]):
            path = path[:i + 1] + path[j:]
    # greedy pass from start
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not edge_free(chk, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def densify(path):
    out = [np.array(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        out.extend(interp(a, b))
    return out


# ---------------------------------------------------------------- approach
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.rng = np.random.default_rng(0)

    # ---- state parsing
    def _parse(self, state):
        objs = {}
        robot = None
        for o in state:
            tn = o.type.name
            if tn == 'crv_robot':
                robot = o
            else:
                objs[o.name] = o
        g = lambda o, f: float(state.get(o, f))
        r = {f: g(robot, f) for f in ['x', 'y', 'theta', 'base_radius', 'arm_joint', 'arm_length',
                                       'vacuum', 'gripper_height', 'gripper_width']}
        rects = {}
        for n, o in objs.items():
            rects[n] = dict(x=g(o, 'x'), y=g(o, 'y'), th=g(o, 'theta'), w=g(o, 'width'),
                            h=g(o, 'height'), static=g(o, 'static'), type=o.type.name)
        return r, rects

    def reset(self, state, info):
        self.plan = []  # list of (target_cfg, vac)
        self.t_start = time.time()
        self.rng = np.random.default_rng(0)
        self.mode = None
        self.fail_count = 0
        self.last_q = None
        self.removed = 0
        self.expect = None
        self.place_fail = 0
        self.force_region_clear = False
        self.extract_fail = 0
        self.stuck = {}
        self.path_blockers = []
        self.last_drop_path = None
        self.protect = None
        self.path_circles = None

    def _q(self, r):
        return np.array([r['x'], r['y'], r['theta'], r['arm_joint']])

    # ---- helpers
    def _robot_model(self, r):
        return Robot(r['base_radius'], r['gripper_width'], r['gripper_height'])

    def _held(self, r, rects, model):
        """Names of objects currently in the suction zone (if vacuum on)."""
        if r['vacuum'] <= 0.5:
            return []
        q = self._q(r)
        S = model.suction(q)
        names = []
        for n, d in rects.items():
            if d['static'] > 0.5 or d['type'] == 'target_region':
                continue
            V = rect_verts(d['x'], d['y'], d['th'], d['w'], d['h'])
            ob = Obstacles([V], [n])
            if ob.rect_hits(S, margin=0.0)[0]:
                names.append(n)
        return names

    def _obstacles(self, rects, exclude):
        vs, ns = [], []
        for n, d in rects.items():
            if n in exclude or d['type'] == 'target_region':
                continue
            vs.append(rect_verts(d['x'], d['y'], d['th'], d['w'], d['h']))
            ns.append(n)
        return Obstacles(vs, ns)

    def _grasp_cfgs(self, d, model, ajs=(ARM_MIN, 0.15, ARM_MAX), n_off=5):
        """Robot configs grasping rect d on each face."""
        V = rect_verts(d['x'], d['y'], d['th'], d['w'], d['h'])
        ctr = V.mean(axis=0)
        cfgs = []
        for k in range(4):
            a, b = V[k], V[(k + 1) % 4]
            mid = (a + b) / 2
            t = b - a
            L = np.linalg.norm(t)
            t = t / L
            nrm = mid - ctr
            nrm = nrm - t * (nrm @ t)
            nrm = nrm / np.linalg.norm(nrm)
            th = math.atan2(-nrm[1], -nrm[0])
            maxoff = max(0.0, L / 2 - 0.02)
            offs = [0.0] if maxoff < 1e-6 else list(np.linspace(-maxoff, maxoff, n_off))
            offs.sort(key=abs)
            for aj in ajs:
                dist = aj + model.gw / 2 + GRASP_GAP
                for s in offs:
                    p = mid + t * s + nrm * dist
                    cfgs.append(np.array([p[0], p[1], th, aj]))
        return cfgs

    def _plan_to_any(self, chk, start, goals, time_budget, per_goal=1.0, max_goals=12, extra=None):
        pre = []
        if not chk.free(start):
            esc = escape(chk, start, self.rng)
            if esc is None:
                return None
            pre = [np.array(start, float)] + esc[:-1]
            start = esc[-1]
        res = self._plan_to_any_inner(chk, start, goals, time_budget, per_goal, max_goals, extra)
        if res is None:
            return None
        return pre + res

    def _plan_to_any_inner(self, chk, start, goals, time_budget, per_goal=1.0, max_goals=12, extra=None):
        if extra is not None:
            goals = [g + 0 for g in goals]
            ex = {id(g): e for g, e in zip(goals, extra)}
        goals = [g for g in goals if chk.free(g)]
        if not goals:
            return None
        # necessary condition: base circle must be able to reach the goal base position
        try:
            if getattr(chk, '_lab', None) is None:
                chk._lab = base_reach_labels(chk.obs, chk.robot.r)
            lab, res = chk._lab
            ls = cell(lab, res, start)
            if ls > 0:
                g2 = [g for g in goals if cell(lab, res, g) == ls]
                if not g2:
                    if chk.held:
                        # explore anyway (tree is used for partial drops)
                        rrt_connect(chk, start, goals[:20], self.rng, time_limit=0.4)
                    return None
                goals = g2
        except Exception:
            pass
        if extra is not None:
            goals.sort(key=lambda g: nsteps(start, g) + ex[id(g)])
        else:
            goals.sort(key=lambda g: nsteps(start, g))
        goals = goals[:60]
        for g in goals[:30]:
            if edge_free(chk, start, g):
                return densify([np.array(start, float), g])
        t0 = time.time()
        best = None
        t_first = None
        for attempt in range(6):
            remaining = time_budget - (time.time() - t0)
            if best is not None:
                el = time.time() - self.t_start
                extra = 0.4 if el < 25 else (0.15 if el < 45 else 0.0)
                remaining = min(remaining, extra - (time.time() - t_first))
            if remaining <= 0.02:
                break
            path = rrt_connect(chk, start, goals, self.rng, time_limit=remaining)
            if path is None:
                if best is None:
                    break
                continue
            path = shortcut(chk, path, self.rng, iters=100)
            dense = densify(path)
            if best is None or len(dense) < len(best):
                best = dense
            if t_first is None:
                t_first = time.time()
        return best

    # ---- high level planning
    def _make_plan(self, state):
        r, rects = self._parse(state)
        model = self._robot_model(r)
        q0 = self._q(r)
        held = self._held(r, rects, model)
        tt = model.tooltip(q0)
        held_specs = []
        for n in held:
            d = rects[n]
            rel = se2_mul(se2_inv(tt), np.array([d['x'], d['y'], d['th']]))
            held_specs.append((rel, d['w'], d['h']))
        obs = self._obstacles(rects, set(held))
        chk = Checker(model, obs, held_specs, extra_suction_check=True)
        tgt = rects['target_block']
        reg = rects['target_region']
        elapsed = time.time() - self.t_start
        budget = 6.0 if elapsed < 40 else 2.0

        if 'target_block' in held:
            # carry to region
            spec = held_specs[held.index('target_block')]
            goals = self._place_cfgs(model, spec, tgt, reg)
            tp0 = time.time()
            path = self._plan_to_any(chk, q0, goals, budget * 0.5)
            if path is not None:
                self.plan = self._mk_plan(path, 1.0, 0.0)
                return True
            # cannot place: put the target down away from the region, clear region
            self.place_fail += 1
            self.force_region_clear = True
            try:
                self._compute_path_blockers(model, rects, held, held_specs, q0, goals, reg)
            except Exception:
                self.path_blockers = []
            tp1 = time.time()
            goals = self._drop_cfgs(chk, q0, rects, held, target_mode=True)
            path = self._plan_to_any(chk, q0, goals, min(2.5, budget * 0.5))
            if DEBUG:
                print('DBG place-fail times pb %.2f drop %.2f total %.2f ok %s' % (
                    tp1 - tp0, time.time() - tp1, time.time() - tp0, path is not None))
            if path is None:
                # stuck holding the target: release it and clear around it
                self.extract_fail += 1
                self.plan = [(q0, 0.0)]
                return True
            self.plan = self._mk_plan(path, 1.0, 0.0)
            return True
        if held:
            # carry obstruction away and release
            goals = self._drop_cfgs(chk, q0, rects, held)
            path = self._plan_to_any(chk, q0, goals, min(2.0, budget * 0.5))
            if path is None:
                path = self._partial_drop(chk, q0, rects, held)
                if path is not None:
                    if DEBUG:
                        print('DBG partial-drop', held, len(path))
                    self.plan = self._mk_plan(path, 1.0, 0.0)
                    self.last_drop_path = [np.array(p, float) for p in path]
                    return True
            if path is None:
                if DEBUG:
                    print('DBG held-fail', held, 'free', chk.free(q0), 'free0', chk.free(q0, 0.0),
                          'ngoals', len(goals), 'nfree', sum(chk.free(g) for g in goals))
                for n in held:
                    self.stuck[n] = self.stuck.get(n, 0) + 1
                self.plan = [(q0, 0.0)]
                return True
            self.plan = self._mk_plan(path, 1.0, 0.0)
            self.last_drop_path = [np.array(p, float) for p in path]
            for m in list(self.stuck):
                if m not in held and self.stuck[m] >= 3:
                    self.stuck[m] = 2
            return True
        chk0 = Checker(model, self._obstacles(rects, set()), [])
        area = 9.0
        try:
            chk0._lab = base_reach_labels(chk0.obs, chk0.robot.r)
            lab, lres = chk0._lab
            ls = cell(lab, lres, q0)
            if ls > 0:
                area = float((lab == ls).sum()) * lres * lres
        except Exception:
            pass
        t_free = time.time()
        ok = self._make_plan_free(r, rects, model, q0, chk0, tgt, reg, budget)
        if DEBUG:
            print('DBG area %.3f ok %s t %.2f' % (area, ok, time.time() - t_free))
        if not ok:
            ok = self._retrace_drop(chk0, q0)
        if not ok:
            ok = self._unstick(model, rects, chk0, q0)
        return ok

    def _unstick(self, model, rects, chk0, q0):
        """Robot seems boxed in: grab the nearest graspable obstruction."""
        near = []
        for n, d in rects.items():
            if d['type'] != 'rectangle' or n == 'target_block' or d.get('static', 0) > 0.5:
                continue
            c = rect_verts(d['x'], d['y'], d['th'], d['w'], d['h']).mean(axis=0)
            dist = np.linalg.norm(c - q0[:2])
            if dist < 0.7:
                near.append((dist, n))
        near.sort()
        t0 = time.time()
        for _, n in near[:6]:
            if time.time() - t0 > 3.0:
                break
            path = self._grasp_obstruction_path(model, rects, chk0, q0, n, False, force=True)
            if path is None:
                path = self._grasp_obstruction_path(model, rects, chk0, q0, n, True, force=True)
            if path is not None:
                if DEBUG:
                    print('DBG unstick', n)
                self.plan = self._mk_plan(path, 0.0, 1.0)
                return True
        return False

    def _retrace_drop(self, chk0, q0):
        """Back out along the path used for the last drop (escape from pockets)."""
        p = self.last_drop_path
        self.last_drop_path = None
        if not p or np.abs(cfg_diff(p[-1], q0)).max() > 1e-3:
            if DEBUG:
                print('DBG retrace-none', None if not p else np.round(p[-1], 3))
            return False
        rev = p[::-1][1:]
        out = [np.array(q0, float)]
        for q in rev:
            if not chk0.free(q, margin=0.0):
                break
            out.append(q)
        if len(out) < 3:
            if DEBUG:
                print('DBG retrace-short', len(out), len(p))
            return False
        if DEBUG:
            print('DBG retrace', len(out))
        self.plan = self._mk_plan(out, 0.0, 0.0)
        return True

    def _make_plan_free(self, r, rects, model, q0, chk0, tgt, reg, budget, _rec=False):
        if r['vacuum'] > 0.5:
            # turn off vacuum first
            self.plan = [(q0, 0.0)]
            return True
        self.protect = None
        # obstructions found to block the path to the region
        pb = [(n, x, y) for n, x, y in self.path_blockers
              if n in rects and abs(rects[n]['x'] - x) < 1e-6 and abs(rects[n]['y'] - y) < 1e-6]
        self.path_blockers = pb
        if pb:
            t0 = time.time()
            for n, _, _ in pb:
                if time.time() - t0 > budget:
                    break
                path, _ = self._try_remove(model, rects, chk0, q0, n, 1, t0 + budget)
                if path is not None:
                    if DEBUG:
                        print('DBG path-clear', n)
                    self.plan = self._mk_plan(path, 0.0, 1.0)
                    return True
            self.path_blockers = []
        if _rec:
            return False
        # is the region (statically) blocked?
        place_sets = []
        for spec in self._hyp_specs(model, tgt):
            pg = self._place_cfgs(model, spec, tgt, reg, n_th=8, dense=False)
            place_sets += self._blocker_sets(model, rects, pg, spec)
        order_p, min_p = self._rank(rects, place_sets, reg)
        if (min_p > 0 or self.force_region_clear) and order_p:
            t0 = time.time()
            for n in sorted(order_p, key=lambda n: self.stuck.get(n, 0) >= 2):
                if time.time() - t0 > budget:
                    break
                path, _ = self._try_remove(model, rects, chk0, q0, n, 1 if n == order_p[0] else 0,
                                           t0 + budget)
                if path is not None:
                    if DEBUG:
                        print('DBG region-clear', n, 'min_p', min_p, 'order', order_p[:5], 'force', self.force_region_clear)
                    self.force_region_clear = False
                    self.plan = self._mk_plan(path, 0.0, 1.0)
                    return True
        # try grasp target
        goals = self._grasp_cfgs(tgt, model)
        obs_nt = self._obstacles(rects, {'target_block'})
        good, bad = [], []
        for g in goals:
            if not chk0.free(g) or not self._suction_clean(model, obs_nt, g):
                continue
            (good if self._retreat_ok(model, obs_nt, g, tgt) else bad).append(g)
        try:
            extra = [self._carry_estimate(model, g, tgt, reg) for g in good]
        except Exception:
            extra = None
        path = self._plan_to_any(chk0, q0, good, min(2.0, budget * 0.5), extra=extra)
        if path is not None:
            if not _rec and time.time() - self.t_start < 40:
                # verify the carry before grasping; otherwise clear the carry path first
                try:
                    g = np.array(path[-1], float)
                    spec = self._spec_for(model, g, tgt)
                    chk_h = Checker(model, obs_nt, [spec], extra_suction_check=True)
                    pl = self._place_cfgs(model, spec, tgt, reg)
                    if self._plan_to_any(chk_h, g, pl, 1.0) is None:
                        self._compute_path_blockers(model, rects, ['target_block'], [spec], g, pl, reg)
                        if DEBUG:
                            print('DBG pre-carry fail', [n for n, _, _ in self.path_blockers])
                        if self.path_blockers:
                            if self._make_plan_free(r, rects, model, q0, chk0, tgt, reg, budget, _rec=True):
                                return True
                except Exception as e:
                    if DEBUG:
                        print('DBG pre-carry exc', repr(e))
            if DEBUG:
                print('DBG grasp target good', _rec)
            self.plan = self._mk_plan(path, 0.0, 1.0)
            return True
        # bad target grasps whose carry to the region is verified by planning
        if self.extract_fail == 0 and bad and time.time() - self.t_start < 40:
            try:
                ex = [self._carry_estimate(model, g, tgt, reg) for g in bad]
                order = sorted(range(len(bad)), key=lambda i: nsteps(q0, bad[i]) + ex[i])
            except Exception:
                order = list(range(len(bad)))
            tv = time.time()
            for i in order[:4]:
                if time.time() - tv > 1.5:
                    break
                g = bad[i]
                p_in = self._plan_to_any(chk0, q0, [g], 0.4)
                if p_in is None:
                    continue
                spec = self._spec_for(model, g, tgt)
                chk_h = Checker(model, obs_nt, [spec], extra_suction_check=True)
                pl = self._place_cfgs(model, spec, tgt, reg)
                p_out = self._plan_to_any(chk_h, g, pl, 0.5)
                if p_out is not None:
                    if DEBUG:
                        print('DBG bad-grasp verified', len(p_in), len(p_out))
                    self.plan = self._mk_plan(p_in, 0.0, 1.0)
                    return True
        # choose obstruction to remove
        sets = self._blocker_sets(model, rects, goals, None, retreat_tgt=tgt)
        scored, _ = self._rank(rects, sets, tgt)
        scored = sorted(scored, key=lambda n: self.stuck.get(n, 0) >= 2)
        allr, _ = self._rank(rects, sets, tgt, include_all=True)
        rest = [n for n in allr if n not in scored and n != 'target_block']
        t0 = time.time()
        dl = t0 + budget

        def done(path, n, tag):
            if DEBUG:
                print('DBG tgt-clear', tag, n, 'scored', scored[:5], 'ngood', len(good), 'nbad', len(bad))
            self.plan = self._mk_plan(path, 0.0, 1.0)
            return True
        for n in scored:
            if time.time() > dl:
                break
            path = self._grasp_obstruction_path(model, rects, chk0, q0, n, False)
            if path is not None:
                return done(path, n, 'A')
        for n in scored[:4]:
            if time.time() > dl:
                break
            path, m = self._try_remove(model, rects, chk0, q0, n, 1, dl)
            if path is not None:
                return done(path, m, 'C')
        for allow_bad in (False, True):
            for n in rest[:12]:
                if time.time() > dl:
                    break
                path = self._grasp_obstruction_path(model, rects, chk0, q0, n, allow_bad)
                if path is not None:
                    return done(path, n, 'D')
        return False

    def _compute_path_blockers(self, model, rects, held, held_specs, q0, goals, reg):
        """Plan ignoring obstructions near the region; obstructions hit by that
        path are the ones to remove (in order of encounter)."""
        rc = rect_verts(reg['x'], reg['y'], reg['th'], reg['w'], reg['h']).mean(axis=0)
        self.path_blockers = []
        sc = np.array(q0[:2])
        t_pb = time.time()
        for rr, rs in ((0.45, 0.0), (0.7, 0.0), (0.0, 0.5), (0.5, 0.5), (0.8, 0.8), (9.0, 9.0)):
            if time.time() - t_pb > 2.5:
                break
            keep = {}
            for n, d in rects.items():
                if n in held:
                    continue
                c = rect_verts(d['x'], d['y'], d['th'], d['w'], d['h']).mean(axis=0)
                if d['type'] == 'rectangle' and n != 'target_block' and \
                        (np.linalg.norm(c - rc) < rr or np.linalg.norm(c - sc) < rs):
                    continue
                keep[n] = d
            chk_rel = Checker(model, self._obstacles(keep, set()), held_specs)
            obs_full = self._obstacles(rects, set(held))
            chk_full = Checker(model, obs_full, held_specs)
            best = None
            t_lvl = time.time()
            for k in range(5):
                if k > 0 and (time.time() - t_lvl > 0.8 or time.time() - t_pb > 2.5):
                    break
                path = self._plan_to_any(chk_rel, q0, goals, 0.4)
                if path is None:
                    break
                order = []
                for q in densify(path):
                    h, _ = chk_full.colliders(q, margin=0.01)
                    for i in np.nonzero(h)[0]:
                        n = obs_full.names[i]
                        if n not in order and n != 'target_block':
                            order.append(n)
                if best is None or len(order) < len(best[0]):
                    best = (order, path)
                if len(best[0]) <= 1:
                    break
            if best is None:
                continue
            order, bpath = best
            try:
                dp = densify(bpath)[::2]
                self.path_circles = self._cfg_circles(
                    model, dp, [chk_rel.held_polys(q)[0].mean(axis=0) for q in dp])
            except Exception:
                self.path_circles = None
            self.path_blockers = [(n, rects[n]['x'], rects[n]['y']) for n in order]
            if DEBUG:
                print('DBG path_blockers', rr, rs, order)
            return

    def _carry_estimate(self, model, g, tgt, reg):
        spec = self._spec_for(model, g, tgt)
        P = np.array(self._place_cfgs(model, spec, tgt, reg, n_th=16, dense=False))
        if len(P) == 0:
            return 0.0
        d = P - g
        d[:, 2] = (d[:, 2] + np.pi) % (2 * np.pi) - np.pi
        return float(np.ceil((np.abs(d) / STEP).max(axis=1)).min())

    def _try_remove(self, model, rects, chk0, q0, n, depth, deadline):
        """Path to grasp obstruction n; if impossible, recursively clear its blockers."""
        path = self._grasp_obstruction_path(model, rects, chk0, q0, n, False)
        if path is None:
            path = self._grasp_obstruction_path(model, rects, chk0, q0, n, True)
        if path is not None or depth <= 0 or time.time() > deadline:
            return path, n
        d = rects[n]
        sets = self._blocker_sets(model, rects, self._grasp_cfgs(d, model), None,
                                  retreat_tgt=d, exclude={n, 'target_block'})
        sub, _ = self._rank(rects, sets, d)
        sub = sorted(sub[:5], key=lambda m: self.stuck.get(m, 0) >= 2)
        for m in sub:
            if time.time() > deadline:
                break
            path, mm = self._try_remove(model, rects, chk0, q0, m, depth - 1, deadline)
            if path is not None:
                if DEBUG:
                    print('DBG sub-clear', m, 'for', n)
                if self.protect is None:
                    try:
                        gc = [g for g in self._grasp_cfgs(d, model) if g[3] <= ARM_MIN + 1e-6]
                        self.protect = self._cfg_circles(model, gc)
                    except Exception:
                        self.protect = None
                return path, mm
        return None, n

    def _grasp_obstruction_path(self, model, rects, chk0, q0, n, allow_bad=True, force=False):
        if self.stuck.get(n, 0) >= 3 and not force:
            return None
        d = rects[n]
        goals = self._grasp_cfgs(d, model)
        obs_n = self._obstacles(rects, {n})
        good, bad = [], []
        for g in goals:
            if not chk0.free(g) or not self._suction_clean(model, obs_n, g):
                continue
            (good if self._retreat_ok(model, obs_n, g, d) else bad).append(g)
        path = None
        if not allow_bad:
            path = self._plan_to_any(chk0, q0, good, 0.8)
        elif (self.stuck.get(n, 0) == 0 or force) and bad:
            path = self._plan_to_any(chk0, q0, bad, 0.4)
        return path

    def _hyp_specs(self, model, tgt):
        """Hypothetical held specs of the target for each face grasp."""
        specs = []
        for g in self._grasp_cfgs(tgt, model, ajs=(ARM_MIN,), n_off=1):
            tt = model.tooltip(g)
            rel = se2_mul(se2_inv(tt), np.array([tgt['x'], tgt['y'], tgt['th']]))
            specs.append((rel, tgt['w'], tgt['h']))
        return specs

    def _spec_for(self, model, g, tgt):
        tt = model.tooltip(g)
        rel = se2_mul(se2_inv(tt), np.array([tgt['x'], tgt['y'], tgt['th']]))
        return (rel, tgt['w'], tgt['h'])

    def _suction_clean(self, model, obs_other, g):
        """Suction zone at g must not touch any object other than the intended one."""
        S = model.suction(g)
        return not obs_other.rect_hits(S, margin=0.004).any()

    def _retreat_ok(self, model, obs_nt, g, tgt):
        chk = Checker(model, obs_nt, [self._spec_for(model, g, tgt)])
        u = np.array([math.cos(g[2]), math.sin(g[2])])
        for back in [0.04, 0.1, 0.17, 0.25, 0.33]:
            q = g.copy()
            q[:2] -= u * back
            if not chk.free(q, margin=0.0):
                return False
        return True

    def _blocker_sets(self, model, rects, goals, spec, retreat_tgt=None, exclude=()):
        obst_names = [n for n, d in rects.items() if d['type'] == 'rectangle' and n not in exclude]
        obs = self._obstacles({n: rects[n] for n in obst_names}, set())
        chk = Checker(model, obs, [spec] if spec is not None else [])
        sets = []
        for g in goals:
            hits = np.zeros(obs.n, bool)
            ok = True
            u = np.array([math.cos(g[2]), math.sin(g[2])])
            chk_r = chk
            if retreat_tgt is not None:
                chk_r = Checker(model, obs, [self._spec_for(model, g, retreat_tgt)])
            for back in [0.0, 0.08, 0.16, 0.25, 0.35]:
                c = chk if back == 0 else chk_r
                q = g.copy()
                q[:2] -= u * back
                h, okw = c.colliders(q, margin=0.01)
                if back == 0.0 and not okw:
                    ok = False
                if back == 0.0 or okw:
                    hits |= h
            if ok:
                sets.append(set(obs.names[i] for i in np.nonzero(hits)[0]))
        return sets

    def _rank(self, rects, sets, ref, include_all=False):
        rc = rect_verts(ref['x'], ref['y'], ref['th'], ref['w'], ref['h']).mean(axis=0)
        if not sets:
            min_s = 0
            score = {}
        else:
            min_s = min(len(s) for s in sets)
            score = {}
            for s in sets:
                if not s:
                    continue
                w = 1.0 if len(s) == min_s else (0.3 if len(s) <= min_s + 1 else 0.05)
                for n in s:
                    score[n] = score.get(n, 0.0) + w / len(s)
        def key(n):
            d = rects[n]
            c = rect_verts(d['x'], d['y'], d['th'], d['w'], d['h']).mean(axis=0)
            return (-score.get(n, 0.0), np.linalg.norm(c - rc))
        names = list(score.keys())
        if include_all:
            names = [n for n, d in rects.items() if d['type'] == 'rectangle']
        return sorted(names, key=key), min_s

    def _blocker_order(self, model, rects, goals):
        obst_names = [n for n, d in rects.items() if d['type'] == 'rectangle']
        if not obst_names:
            return []
        obs = self._obstacles({n: rects[n] for n in obst_names}, set())
        chk = Checker(model, obs, [])
        tgt = rects['target_block']
        tc = rect_verts(tgt['x'], tgt['y'], tgt['th'], tgt['w'], tgt['h']).mean(axis=0)
        # for each goal, blockers = obstruction hits at goal and along backing-off approach
        best_sets = []
        for g in goals:
            hits = np.zeros(obs.n, bool)
            ok = True
            u = np.array([math.cos(g[2]), math.sin(g[2])])
            for back in [0.0, 0.08, 0.16, 0.25, 0.35]:
                q = g.copy()
                q[:2] -= u * back
                h, okw = chk.colliders(q, margin=0.01)
                if back == 0.0 and not okw:
                    ok = False
                hits |= h
            # also target block itself (we can't move it) - checked at goal: ignore
            if ok:
                best_sets.append(hits)
        score = {}
        if best_sets:
            m = min(s.sum() for s in best_sets)
            for s in best_sets:
                w = 1.0 if s.sum() == m else (0.3 if s.sum() <= m + 1 else 0.1)
                for i in np.nonzero(s)[0]:
                    score[obs.names[i]] = score.get(obs.names[i], 0) + w / max(1, s.sum())
        def key(n):
            d = rects[n]
            c = rect_verts(d['x'], d['y'], d['th'], d['w'], d['h']).mean(axis=0)
            return (-score.get(n, 0.0), np.linalg.norm(c - tc))
        return sorted(obst_names, key=key)

    def _mk_plan(self, path, vac_move, vac_final):
        if len(path) <= 1:
            return [(path[-1], vac_final)]
        return [(q, vac_move) for q in path[1:-1]] + [(path[-1], vac_final)]

    def _place_cfgs(self, model, spec, tgt, reg, n_th=48, dense=True):
        rel, w, h = spec
        RV = rect_verts(reg['x'], reg['y'], reg['th'], reg['w'], reg['h'])
        rc = RV.mean(axis=0)
        rth = reg['th']
        rcos, rsin = math.cos(rth), math.sin(rth)
        cl = np.array([w / 2, h / 2])
        c, s = math.cos(rel[2]), math.sin(rel[2])
        bc_rel = rel[:2] + np.array([c * cl[0] - s * cl[1], s * cl[0] + c * cl[1]])
        phi = rel[2]
        half = reg['w'] / 2
        hb = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        offs = [(0.0, 0.0)]
        for ox in ((-0.025, 0.0, 0.025) if dense else ()):
            for oy in (-0.025, 0.0, 0.025):
                if ox or oy:
                    offs.append((ox, oy))
        cfgs = []
        ths = [wrap(rth - phi + k * np.pi / 2) for k in range(4)]
        ths += list(np.linspace(-np.pi, np.pi, n_th, endpoint=False))
        for th in ths:
            bth = th + phi
            a = bth - rth  # block angle in region frame
            ca, sa = math.cos(a), math.sin(a)
            corners = hb @ np.array([[ca, sa], [-sa, ca]])
            ct, st = math.cos(th), math.sin(th)
            for ox, oy in offs:
                pts = corners + np.array([ox, oy])
                if np.abs(pts).max() > half - 0.004:
                    continue
                bcw = rc + np.array([rcos * ox - rsin * oy, rsin * ox + rcos * oy])
                ttp = bcw - np.array([ct * bc_rel[0] - st * bc_rel[1], st * bc_rel[0] + ct * bc_rel[1]])
                for aj in (ARM_MIN, 0.15, ARM_MAX):
                    base = ttp - np.array([ct, st]) * (aj + model.gw / 2)
                    cfgs.append(np.array([base[0], base[1], th, aj]))
        return cfgs

    def _important_circles(self, model, rects, held):
        """Bounding circles of robot (+target) at place/grasp goals and their retreats."""
        tgt = rects['target_block']
        reg = rects['target_region']
        rc = rect_verts(reg['x'], reg['y'], reg['th'], reg['w'], reg['h']).mean(axis=0)
        tc = rect_verts(tgt['x'], tgt['y'], tgt['th'], tgt['w'], tgt['h']).mean(axis=0)
        def circ(cfgs, ctr, with_blk):
            out = []
            for q in cfgs:
                u = np.array([math.cos(q[2]), math.sin(q[2])])
                tt = model.tooltip(q)[:2]
                cs = []
                for back in (0.0, 0.12, 0.25):
                    cs.append((q[0] - u[0] * back, q[1] - u[1] * back, 0.1))
                    cs.append((tt[0] - u[0] * back, tt[1] - u[1] * back, 0.045))
                    if with_blk:
                        cs.append((ctr[0] - u[0] * back, ctr[1] - u[1] * back, 0.099))
                out.append(cs)
            return np.array(out) if out else np.zeros((0, 1, 3))
        pl = []
        for spec in self._hyp_specs(model, tgt):
            pl += [q for q in self._place_cfgs(model, spec, tgt, reg, n_th=8, dense=False) if q[3] == ARM_MIN]
        P = circ(pl, rc, True)
        if 'target_block' in held:
            G = np.zeros((0, 1, 3))
        else:
            G = circ(self._grasp_cfgs(tgt, model, ajs=(ARM_MIN,), n_off=3), tc, False)
        return P, G

    @staticmethod
    def _cfg_circles(model, cfgs, ctrs=None):
        out = []
        for k, q in enumerate(cfgs):
            u = np.array([math.cos(q[2]), math.sin(q[2])])
            tt = model.tooltip(q)[:2]
            cs = []
            for back in (0.0, 0.12):
                cs.append((q[0] - u[0] * back, q[1] - u[1] * back, 0.1))
                cs.append((tt[0] - u[0] * back, tt[1] - u[1] * back, 0.045))
                if ctrs is not None:
                    c = ctrs[k]
                    cs.append((c[0] - u[0] * back, c[1] - u[1] * back, 0.099))
            out.append(cs)
        return np.array(out) if out else np.zeros((0, 1, 3))

    @staticmethod
    def _blocked_frac(C, c, r):
        if len(C) == 0:
            return 0.0
        d = np.hypot(C[..., 0] - c[0], C[..., 1] - c[1])
        return float((d < C[..., 2] + r - 0.01).any(axis=1).mean())

    def _drop_cfgs(self, chk, q0, rects, held, target_mode=False, samples=None):
        tgt = rects['target_block']
        reg = rects['target_region']
        tc = rect_verts(tgt['x'], tgt['y'], tgt['th'], tgt['w'], tgt['h']).mean(axis=0)
        rc = rect_verts(reg['x'], reg['y'], reg['th'], reg['w'], reg['h']).mean(axis=0)
        others = [rect_verts(d['x'], d['y'], d['th'], d['w'], d['h']).mean(axis=0)
                  for n, d in rects.items() if d['type'] == 'rectangle' and n not in held]
        others = np.array(others) if others else np.zeros((0, 2))
        seg = rc - tc
        L2 = max(seg @ seg, 1e-9)
        model = chk.robot
        try:
            IP, IG = self._important_circles(model, rects, held)
        except Exception:
            IP, IG = np.zeros((0, 1, 3)), np.zeros((0, 1, 3))
        hr = [0.5 * math.hypot(rects[n]['w'], rects[n]['h']) for n in held]
        lab = None
        if samples is None:
            try:
                if getattr(chk, '_lab', None) is None:
                    chk._lab = base_reach_labels(chk.obs, chk.robot.r)
                lab, lres = chk._lab
                ls = cell(lab, lres, q0)
                if ls <= 0:
                    lab = None
            except Exception:
                lab = None
        cands = []
        for k in range(700 if samples is None else len(samples)):
            if samples is None:
                q = np.array([self.rng.uniform(0.1, 2.4), self.rng.uniform(0.1, 2.4),
                              self.rng.uniform(-np.pi, np.pi), ARM_MIN])
                if lab is not None and cell(lab, lres, q) != ls:
                    continue
                if not chk.free(q, margin=0.01):
                    continue
            else:
                q = samples[k]
            cs = [P.mean(axis=0) for P in chk.held_polys(q)]
            dt = min(np.linalg.norm(c - tc) for c in cs)
            dr = min(np.linalg.norm(c - rc) for c in cs)
            dseg = min(np.linalg.norm(c - (tc + seg * np.clip((c - tc) @ seg / L2, 0, 1))) for c in cs)
            dcl = min(np.linalg.norm(others - c, axis=1).min() if len(others) else 9.0 for c in cs)
            pen = 0.0
            wd = min(q[0], q[1], 2.5 - q[0], 2.5 - q[1])
            pen += 60 * max(0.0, 0.3 - wd)
            if len(others):
                rb = np.linalg.norm(others - q[:2], axis=1).min()
                pen += 40 * max(0.0, 0.3 - rb)
            for c, rr in zip(cs, hr):
                pen += 60 * self._blocked_frac(IP, c, rr)
                cx = min(c[0], 2.5 - c[0]); cy = min(c[1], 2.5 - c[1])
                if cx < 0.4 and cy < 0.4:
                    pen += 40 * (0.4 - max(cx, cy)) / 0.4 + 20
                if self.protect is not None:
                    pen += 60 * self._blocked_frac(self.protect, c, rr)
                if not target_mode and self.path_circles is not None and self.path_blockers:
                    if self._blocked_frac(self.path_circles, c, rr) > 0:
                        pen += 40
                if not target_mode:
                    pen += 60 * self._blocked_frac(IG, c, rr)
            if target_mode:
                if dr < 0.6:
                    pen += 200
                if dcl < 0.3:
                    pen += 30
            else:
                if 'target_block' not in held:
                    pen += 100 * max(0.0, 1.0 - dt) + (50 if dt < 0.5 else 0)
                pen += 100 * max(0.0, 0.7 - dr) + (50 if dr < 0.4 else 0)
                pen += 100 * max(0.0, 0.45 - dseg) + (30 if dseg < 0.3 else 0)
                if dcl < 0.25:
                    pen += 15
            cands.append((nsteps(q0, q) + pen, q))
        if samples is not None:
            return cands
        cands.sort(key=lambda t: t[0])
        return [q for _, q in cands[:20]]

    def _partial_drop(self, chk, q0, rects, held):
        """After a failed drop plan, move the held obstruction to the best spot
        reachable within the explored RRT tree."""
        if LAST_FAIL[0] is None or not chk.free(q0):
            return None
        N, par = LAST_FAIL[0]
        LAST_FAIL[0] = None
        if len(N) < 2 or np.abs(cfg_diff(N[0], q0)).max() > 1e-6:
            return None
        idx = np.arange(len(N))
        if len(idx) > 300:
            idx = np.concatenate([[0], self.rng.choice(np.arange(1, len(N)), 299, replace=False)])
        sc = self._drop_cfgs(chk, q0, rects, held, samples=[N[i] for i in idx])
        sc = [(c - nsteps(q0, N[i]) * 0.7, i) for (c, _), i in zip(sc, idx)]
        base = sc[0][0]
        best = min(sc, key=lambda t: t[0])
        if best[0] > base - 10:
            return None
        out = []
        i = best[1]
        while i != -1:
            out.append(N[i].copy())
            i = par[i]
        return densify(out[::-1])

    # ---- action
    def _action_to(self, q0, q1, vac):
        d = cfg_diff(q0, q1)
        a = np.array([d[0], d[1], d[2], d[3], vac], dtype=np.float64)
        lo = self.action_space.low.astype(np.float64)
        hi = self.action_space.high.astype(np.float64)
        a = np.clip(a, lo, hi)
        return a.astype(np.float32)

    def get_action(self, state):
        r, rects = self._parse(state)
        q = self._q(r)
        # verify plan tracking
        if self.plan and self.last_q is not None and self.expect is not None:
            if np.abs(cfg_diff(q, self.expect)[:2]).max() > 1e-3 or abs(cfg_diff(q, self.expect)[2]) > 1e-3 \
                    or abs(q[3] - self.expect[3]) > 1e-3:
                self.plan = []
        if not self.plan:
            ok = False
            try:
                ok = self._make_plan(state)
            except Exception:
                ok = False
            if not ok:
                self.fail_count += 1
                # random jiggle
                a = self.action_space.sample()
                a[4] = 0.0 if r['vacuum'] <= 0.5 else 1.0
                a[0:4] *= 0.5
                self.last_q = q
                self.expect = None
                return a.astype(np.float32)
        tgt_q, vac = self.plan.pop(0)
        a = self._action_to(q, tgt_q, vac)
        self.last_q = q
        self.expect = tgt_q
        return a
