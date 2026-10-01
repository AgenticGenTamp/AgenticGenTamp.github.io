"""Approach for ClutteredRetrieval2D: RRT-connect planning with exact geometry,
removing obstructions when the target block cannot be grasped directly."""
import math
import time

import numpy as np
try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

WORLD = 2.5
BASE_R = 0.1
GRIP_W = 0.01   # along arm
GRIP_H = 0.07   # across arm
ARM_W = 0.006
ARM_MIN = 0.1
ARM_MAX = 0.2
MARGIN = 0.004
DEBUG = False
ITERS_PER_SEC = 600
GRASP_GAP = 0.007
MAX_DXY = 0.049
MAX_DTH = 0.19
MAX_DARM = 0.098


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rect_corners(x, y, th, w, h):
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-math.sin(th), math.cos(th)])
    p = np.array([x, y])
    return np.array([p, p + u * w, p + u * w + v * h, p + v * h])


class Obstacles:
    def __init__(self, polys, caps=None):
        # polys: list of (4,2) arrays (rectangles); caps: per-obstacle max margin
        self.n = len(polys)
        self.caps = np.full(self.n, np.inf) if caps is None else np.asarray(caps, float)
        if self.n == 0:
            return
        P = np.array(polys, dtype=float)  # (N,4,2)
        self.P = P
        e1 = P[:, 1] - P[:, 0]
        e2 = P[:, 3] - P[:, 0]
        self.w = np.linalg.norm(e1, axis=1)
        self.h = np.linalg.norm(e2, axis=1)
        self.u = e1 / self.w[:, None]
        self.v = e2 / self.h[:, None]
        self.axes = np.stack([self.u, self.v], axis=1)  # (N,2,2)
        proj = np.einsum('nkd,ncd->nkc', self.axes, P)  # (N,2 axes,4 corners)
        self.omin = proj.min(axis=2)
        self.omax = proj.max(axis=2)

    def circle_hit(self, C, r, m):
        """C: (M,2). returns (M,) bool."""
        if self.n == 0:
            return np.zeros(len(C), bool)
        d = C[:, None, :] - self.P[None, :, 0, :]  # (M,N,2)
        lu = np.einsum('mnd,nd->mn', d, self.u)
        lv = np.einsum('mnd,nd->mn', d, self.v)
        cu = np.clip(lu, 0, self.w[None])
        cv = np.clip(lv, 0, self.h[None])
        dist2 = (lu - cu) ** 2 + (lv - cv) ** 2
        mm = np.minimum(m, self.caps)[None]
        return (dist2 < (r + mm) ** 2).any(axis=1)

    def poly_hit(self, Q, m, per_obj=False):
        """Q: (M,4,2) convex polys. Broad-phase + pairwise SAT (same result as poly_hit_full)."""
        if self.n == 0 or per_obj or len(Q) < 4:
            return self.poly_hit_full(Q, m, per_obj)
        oc = getattr(self, '_oc', None)
        if oc is None:
            oc = self.P.mean(axis=1)
            self._oc = oc
            self._or = np.sqrt(((self.P - oc[:, None]) ** 2).sum(-1)).max(axis=1)
        qc = Q.mean(axis=1)
        qr = np.sqrt(((Q - qc[:, None]) ** 2).sum(-1)).max(axis=1)
        dx = qc[:, None, 0] - oc[None, :, 0]
        dy = qc[:, None, 1] - oc[None, :, 1]
        lim = qr[:, None] + self._or[None] + 2.0 * m + 1e-6
        cand = dx * dx + dy * dy < lim * lim
        im, jn = np.nonzero(cand)
        out = np.zeros(len(Q), bool)
        if len(im) == 0:
            return out
        Qk = Q[im]
        Pk = self.P[jn]
        e1 = Qk[:, 1] - Qk[:, 0]
        e2 = Qk[:, 3] - Qk[:, 0]
        a1 = e1 / np.sqrt((e1 ** 2).sum(-1))[:, None]
        a2 = e2 / np.sqrt((e2 ** 2).sum(-1))[:, None]
        A = np.concatenate([a1[:, None], a2[:, None], self.axes[jn]], axis=1)  # (K,4,2)
        pq = A[:, :, None, 0] * Qk[:, None, :, 0] + A[:, :, None, 1] * Qk[:, None, :, 1]  # (K,4,4)
        po = A[:, :, None, 0] * Pk[:, None, :, 0] + A[:, :, None, 1] * Pk[:, None, :, 1]
        mk = np.minimum(m, self.caps)[jn][:, None]
        qmin = pq.min(axis=2)
        qmax = pq.max(axis=2)
        omin = po.min(axis=2)
        omax = po.max(axis=2)
        sep = ((qmax + mk <= omin) | (omax + mk <= qmin)).any(axis=1)
        out[im[~sep]] = True
        return out

    def poly_hit_full(self, Q, m, per_obj=False):
        """Q: (M,4,2) convex polys (rectangles). returns (M,) bool."""
        if self.n == 0:
            if per_obj:
                return np.zeros((len(Q), 0), bool)
            return np.zeros(len(Q), bool)
        e1 = Q[:, 1] - Q[:, 0]
        e2 = Q[:, 3] - Q[:, 0]
        a1 = e1 / np.linalg.norm(e1, axis=1)[:, None]
        a2 = e2 / np.linalg.norm(e2, axis=1)[:, None]
        qaxes = np.stack([a1, a2], axis=1)  # (M,2,2)
        # Q's own axes: project Q and obstacles
        pq = np.einsum('mkd,mcd->mkc', qaxes, Q)  # (M,2,4)
        qmin1 = pq.min(axis=2)
        qmax1 = pq.max(axis=2)
        po = np.einsum('mkd,ncd->mnkc', qaxes, self.P)  # (M,N,2,4)
        omin1 = po.min(axis=3)
        omax1 = po.max(axis=3)
        m = np.minimum(m, self.caps)[None, :, None]
        sep1 = ((qmax1[:, None] + m <= omin1) | (omax1 + m <= qmin1[:, None])).any(axis=2)
        # obstacle axes
        pq2 = np.einsum('nkd,mcd->mnkc', self.axes, Q)  # (M,N,2,4)
        qmin2 = pq2.min(axis=3)
        qmax2 = pq2.max(axis=3)
        sep2 = ((qmax2 + m <= self.omin[None]) | (self.omax[None] + m <= qmin2)).any(axis=2)
        hit = ~(sep1 | sep2)  # (M,N)
        if per_obj:
            return hit
        return hit.any(axis=1)


def local_to_world(Q, L):
    """Q: (M,4) configs; L: (K,2) local points in robot frame (x along arm).
    returns (M,K,2)."""
    c = np.cos(Q[:, 2])[:, None]
    s = np.sin(Q[:, 2])[:, None]
    X = Q[:, 0:1] + c * L[None, :, 0] - s * L[None, :, 1]
    Y = Q[:, 1:2] + s * L[None, :, 0] + c * L[None, :, 1]
    return np.stack([X, Y], axis=2)


GRIP_LOCAL = np.array([[-GRIP_W / 2, -GRIP_H / 2], [GRIP_W / 2, -GRIP_H / 2],
                       [GRIP_W / 2, GRIP_H / 2], [-GRIP_W / 2, GRIP_H / 2]])


class Model:
    """Collision model for robot (+ optional held object) vs obstacles."""

    extra = 0.0

    def __init__(self, obstacles, held_local=None, margin=MARGIN):
        self.obs = obstacles
        self.held = held_local  # (4,2) in gripper frame (relative to (arm,0))
        self.m = margin + Model.extra if margin >= MARGIN else margin

    def body_polys(self, Q):
        M = len(Q)
        polys = []
        gl = np.repeat(GRIP_LOCAL[None], M, axis=0).copy()
        gl[:, :, 0] += Q[:, 3:4]
        polys.append(gl)
        if self.held is not None:
            hl = np.repeat(self.held[None], M, axis=0).copy()
            hl[:, :, 0] += Q[:, 3:4]
            polys.append(hl)
        al = np.zeros((M, 4, 2))
        al[:, :, 0] = np.array([BASE_R * 0.8, 1, 1, BASE_R * 0.8])[None] * np.array([1, 0, 0, 1])[None] \
            + (Q[:, 3:4] - GRIP_W / 2) * np.array([0, 1, 1, 0])[None]
        al[:, :, 1] = np.array([-ARM_W / 2, -ARM_W / 2, ARM_W / 2, ARM_W / 2])[None]
        polys.append(al)
        out = []
        for pl in polys:
            c = np.cos(Q[:, 2])[:, None]
            s = np.sin(Q[:, 2])[:, None]
            X = Q[:, 0:1] + c * pl[:, :, 0] - s * pl[:, :, 1]
            Y = Q[:, 1:2] + s * pl[:, :, 0] + c * pl[:, :, 1]
            out.append(np.stack([X, Y], axis=2))
        return out

    def hits(self, Q):
        Q = np.atleast_2d(Q)
        m = self.m
        bad = (Q[:, 0] < BASE_R + m) | (Q[:, 0] > WORLD - BASE_R - m) | \
              (Q[:, 1] < BASE_R + m) | (Q[:, 1] > WORLD - BASE_R - m)
        bad |= (Q[:, 3] < ARM_MIN - 1e-6) | (Q[:, 3] > ARM_MAX + 1e-6)
        bad |= self.obs.circle_hit(Q[:, :2], BASE_R, m)
        for P in self.body_polys(Q):
            bad |= (P < m).any(axis=(1, 2)) | (P > WORLD - m).any(axis=(1, 2))
            bad |= self.obs.poly_hit(P, m)
        return bad

    def per_obj(self, Q, m=None):
        """(M,N) bool: which obstacles each config hits."""
        Q = np.atleast_2d(Q)
        m = self.m if m is None else m
        ob = self.obs
        if ob.n == 0:
            return np.zeros((len(Q), 0), bool)
        d = Q[:, None, :2] - ob.P[None, :, 0, :]
        lu = np.einsum('mnd,nd->mn', d, ob.u)
        lv = np.einsum('mnd,nd->mn', d, ob.v)
        cu = np.clip(lu, 0, ob.w[None])
        cv = np.clip(lv, 0, ob.h[None])
        hit = ((lu - cu) ** 2 + (lv - cv) ** 2) < (BASE_R + np.minimum(m, ob.caps)[None]) ** 2
        for P in self.body_polys(Q):
            hit |= ob.poly_hit(P, m, per_obj=True)
        return hit

    def wall_bad(self, Q):
        Q = np.atleast_2d(Q)
        m = self.m
        bad = (Q[:, 0] < BASE_R + m) | (Q[:, 0] > WORLD - BASE_R - m) | \
              (Q[:, 1] < BASE_R + m) | (Q[:, 1] > WORLD - BASE_R - m)
        for P in self.body_polys(Q):
            bad |= (P < m).any(axis=(1, 2)) | (P > WORLD - m).any(axis=(1, 2))
        return bad

    def free(self, q):
        return not self.hits(np.asarray(q, float)[None])[0]

    def edge_free(self, q1, q2):
        q1 = np.asarray(q1, float)
        q2 = np.asarray(q2, float)
        d = q2 - q1
        d[2] = wrap(d[2])
        reach = 0.45 if self.held is not None else 0.25
        n = int(math.ceil(max(abs(d[0]) / 0.01, abs(d[1]) / 0.01,
                              abs(d[2]) * reach / 0.01, abs(d[3]) / 0.01, 1)))
        t = np.linspace(0, 1, n + 1)[1:]
        Q = q1[None] + t[:, None] * d[None]
        return not self.hits(Q).any()


def cdist(A, q, wth=0.3):
    d = A - q[None]
    d[:, 2] = (d[:, 2] + math.pi) % (2 * math.pi) - math.pi
    return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + (wth * d[:, 2]) ** 2 + d[:, 3] ** 2)


class Tree:
    def __init__(self, q):
        self.nodes = np.zeros((4096, 4))
        self.nodes[0] = q
        self.parent = [-1]
        self.n = 1

    def add(self, q, p):
        if self.n >= len(self.nodes):
            self.nodes = np.concatenate([self.nodes, np.zeros_like(self.nodes)])
        self.nodes[self.n] = q
        self.parent.append(p)
        self.n += 1
        return self.n - 1

    def nearest(self, q):
        d = cdist(self.nodes[:self.n], q)
        i = int(np.argmin(d))
        return i, d[i]

    def path_to(self, i):
        out = []
        while i >= 0:
            out.append(self.nodes[i].copy())
            i = self.parent[i]
        return out[::-1]


def steer(qa, qb, step):
    d = qb - qa
    d[2] = wrap(d[2])
    dist = math.sqrt(d[0] ** 2 + d[1] ** 2 + (0.3 * d[2]) ** 2 + d[3] ** 2)
    if dist <= step:
        return qb.copy(), True
    q = qa + d * (step / dist)
    q[2] = wrap(q[2])
    return q, False



GRID_RES = 0.025
DUMP_DT = 0.7
DUMP_DR = 0.6
DUMP_DSEG = 0.45
PLACE_SLACK = 0.026
TIGHT_PEN = 6.0
IMPROVE_K = 3
_GRID_CACHE = {}


def clearance_grid(obs):
    key = id(obs)
    if key in _GRID_CACHE and _GRID_CACHE[key][0] is obs:
        return _GRID_CACHE[key][1:]
    xs = np.arange(BASE_R, WORLD - BASE_R + 1e-9, GRID_RES)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    clr = np.minimum(np.minimum(pts[:, 0], WORLD - pts[:, 0]), np.minimum(pts[:, 1], WORLD - pts[:, 1]))
    if obs.n:
        d = pts[:, None, :] - obs.P[None, :, 0, :]
        lu = np.einsum('mnd,nd->mn', d, obs.u)
        lv = np.einsum('mnd,nd->mn', d, obs.v)
        du = np.maximum(np.maximum(-lu, lu - obs.w[None]), 0)
        dv = np.maximum(np.maximum(-lv, lv - obs.h[None]), 0)
        dist = np.sqrt(du ** 2 + dv ** 2)
        clr = np.minimum(clr, dist.min(axis=1))
    clr = clr.reshape(X.shape)
    if len(_GRID_CACHE) > 20:
        _GRID_CACHE.clear()
    _GRID_CACHE[key] = (obs, xs, clr)
    return xs, clr


def body_radius(model):
    r = math.hypot(ARM_MIN + GRIP_W / 2, GRIP_H / 2)
    if model.held is not None:
        hl = model.held.copy()
        hl[:, 0] += ARM_MIN
        r = max(r, float(np.linalg.norm(hl, axis=1).max()))
    return max(r, BASE_R)


def grid_path(model, qa, qb):
    """2D Dijkstra for a disk robot (arm retracted, any heading) from qa to qb.
    qa/qb must have arm == ARM_MIN. Returns list of configs or None."""
    import heapq
    xs, clr = clearance_grid(model.obs)
    R = body_radius(model) + model.m + 0.005
    free = clr >= BASE_R + model.m - 0.008
    tight = clr < R
    n = len(xs)

    def cell(p):
        i = int(round((p[0] - xs[0]) / GRID_RES))
        j = int(round((p[1] - xs[0]) / GRID_RES))
        return min(max(i, 0), n - 1), min(max(j, 0), n - 1)
    s = cell(qa)
    g = cell(qb)
    free = free.copy()
    free[s] = True
    free[g] = True
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.0), (1, -1, 1.0), (-1, 1, 1.0), (-1, -1, 1.0)]  # chebyshev cost
    found = False
    while pq:
        d0, c = heapq.heappop(pq)
        if c == g:
            found = True
            break
        if d0 > dist.get(c, 1e18):
            continue
        for di, dj, w in nbrs:
            nc = (c[0] + di, c[1] + dj)
            if nc[0] < 0 or nc[1] < 0 or nc[0] >= n or nc[1] >= n or not free[nc]:
                continue
            nd = d0 + w * (TIGHT_PEN if tight[nc] else 1.0) + 0.001 * (di != 0 and dj != 0)
            if nd < dist.get(nc, 1e18):
                dist[nc] = nd
                prev[nc] = c
                heapq.heappush(pq, (nd, nc))
    if not found:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    cells = cells[::-1]
    pts = [np.array([xs[i], xs[j]]) for i, j in cells]
    pts[0] = np.asarray(qa[:2], float)
    pts[-1] = np.asarray(qb[:2], float)
    # configs: heading interpolated from qa to qb along path length
    L = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        L.append(L[-1] + np.abs(b - a).max())
    tot = max(L[-1], 1e-9)
    dth = wrap(qb[2] - qa[2])
    cfgs = [np.array([p[0], p[1], wrap(qa[2] + dth * min(1.0, max(0.0, (l / tot - 0.25) * 2))), ARM_MIN])
            for p, l in zip(pts, L)]
    cfgs[0] = np.asarray(qa, float).copy()
    cfgs[-1] = np.asarray(qb, float).copy()
    # line-of-sight shortcut with exact model
    out = [cfgs[0]]
    i = 0
    while i < len(cfgs) - 1:
        j = len(cfgs) - 1
        while j > i + 1 and not model.edge_free(cfgs[i], cfgs[j]):
            j = max(i + 1, j - max(1, (j - i) // 4))
        if not model.edge_free(cfgs[i], cfgs[j]):
            return None
        out.append(cfgs[j])
        i = j
    return out


def retract_branch(model, tree):
    """Return index of a node in tree (retreat branch) plus retracted config."""
    for idx in range(tree.n - 1, -1, -1):
        q = tree.nodes[idx].copy()
        qr = q.copy()
        qr[3] = ARM_MIN
        if model.edge_free(q, qr):
            return idx, qr
    return None, None


_ITERS = [0]
_DEADLINE = [1e18]


def vclock():
    """Deterministic 'virtual time' based on RRT iterations performed."""
    return _ITERS[0] / ITERS_PER_SEC


def seed_retreat(model, tree):
    """Grow a straight back-off branch (opposite the arm direction, arm
    retracting) from the tree root, which usually leaves clutter."""
    q = tree.nodes[0].copy()
    back = np.array([-math.cos(q[2]), -math.sin(q[2])])
    idx = 0
    for k in range(8):
        qn = q.copy()
        qn[:2] += back * 0.05
        qn[3] = max(ARM_MIN, qn[3] - 0.02)
        if not model.edge_free(q, qn):
            break
        idx = tree.add(qn, idx)
        q = qn


def rrt_connect(model, qs, qg, rng, time_limit=1.5, step=0.3):
    qs = np.asarray(qs, float)
    qg = np.asarray(qg, float)
    if model.edge_free(qs, qg):
        return [qs, qg]
    t0 = time.time()
    ta, tb = Tree(qs), Tree(qg)
    seed_retreat(model, ta)
    seed_retreat(model, tb)
    # cheap attempt: connect the retreat branch ends directly
    for ia in range(ta.n - 1, -1, -1):
        if model.edge_free(ta.nodes[ia], tb.nodes[tb.n - 1]):
            return ta.path_to(ia) + tb.path_to(tb.n - 1)[::-1]
        if ia < ta.n - 3:
            break
    # grid-based disk planner between retracted retreat ends
    ia, qa = retract_branch(model, ta)
    ib, qb = retract_branch(model, tb)
    if ia is not None and ib is not None:
        mid = grid_path(model, qa, qb)
        if mid is not None:
            pa = ta.path_to(ia)
            pb = tb.path_to(ib)[::-1]
            return pa + mid + pb
    a_is_start = True
    lo = np.array([BASE_R, BASE_R, -math.pi, ARM_MIN])
    hi = np.array([WORLD - BASE_R, WORLD - BASE_R, math.pi, ARM_MAX])
    max_iter = int(time_limit * ITERS_PER_SEC)
    it = 0
    wall = time_limit * 3
    rem_dl = _DEADLINE[0] - time.time()
    if rem_dl < wall:
        wall = max(0.25, rem_dl)
    while it < max_iter and time.time() - t0 < wall:
        it += 1
        _ITERS[0] += 1
        qr = rng.uniform(lo, hi)
        r = rng.random()
        if r < 0.25:
            qr[2] = wrap(qg[2] + rng.normal(0, 0.25))
        elif r < 0.4:
            qr[2] = wrap(qs[2] + rng.normal(0, 0.25))
        # extend ta toward qr
        i, _ = ta.nearest(qr)
        qn, _ = steer(ta.nodes[i], qr, step)
        if model.edge_free(ta.nodes[i], qn):
            ia = ta.add(qn, i)
            # connect tb toward qn
            j, _ = tb.nearest(qn)
            cur = j
            while True:
                qc, reached = steer(tb.nodes[cur], qn, step)
                if not model.edge_free(tb.nodes[cur], qc):
                    break
                cur = tb.add(qc, cur)
                if reached:
                    pa = ta.path_to(ia)
                    pb = tb.path_to(cur)
                    path = pa + pb[::-1][1:]
                    if not a_is_start:
                        path = path[::-1]
                    return path
        ta, tb = tb, ta
        a_is_start = not a_is_start
    return None


def cost_len(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    return max(abs(d[0]) / MAX_DXY, abs(d[1]) / MAX_DXY, abs(wrap(d[2])) / MAX_DTH, abs(d[3]) / MAX_DARM)


def interp_cfg(a, b, t):
    t = min(max(t, 0.0), 1.0)
    d = np.asarray(b, float) - np.asarray(a, float)
    d[2] = wrap(d[2])
    q = np.asarray(a, float) + t * d
    q[2] = wrap(q[2])
    return q


def shortcut(model, path, rng, iters=60):
    path = [np.asarray(p, float) for p in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path) + 0))
        if j >= len(path):
            j = len(path) - 1
        if j <= i + 1:
            continue
        if model.edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    # partial shortcuts between interior points of segments
    if len(path) >= 3:
        for _ in range(iters):
            segl = [cost_len(a, b) for a, b in zip(path[:-1], path[1:])]
            cum = np.concatenate([[0.0], np.cumsum(segl)])
            tot = cum[-1]
            if tot < 1e-6:
                break
            ta, tb = sorted(rng.uniform(0, tot, 2))
            if tb - ta < 0.05 * tot:
                continue
            ia = int(np.searchsorted(cum, ta, side='right') - 1)
            ib = int(np.searchsorted(cum, tb, side='right') - 1)
            ia = min(ia, len(path) - 2)
            ib = min(ib, len(path) - 2)
            if ia == ib:
                continue
            qa = interp_cfg(path[ia], path[ia + 1], (ta - cum[ia]) / max(segl[ia], 1e-9))
            qb = interp_cfg(path[ib], path[ib + 1], (tb - cum[ib]) / max(segl[ib], 1e-9))
            if cost_len(qa, qb) >= tb - ta - 1e-6:
                continue
            if model.edge_free(qa, qb):
                path = path[:ia + 1] + [qa, qb] + path[ib + 1:]
    # greedy pass
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not model.edge_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def path_to_deltas(path):
    acts = []
    for a, b in zip(path[:-1], path[1:]):
        d = b - a
        d[2] = wrap(d[2])
        n = int(math.ceil(max(abs(d[0]) / MAX_DXY, abs(d[1]) / MAX_DXY,
                              abs(d[2]) / MAX_DTH, abs(d[3]) / MAX_DARM, 1e-9)))
        n = max(n, 1)
        for k in range(n):
            acts.append(d / n)
    return acts


def path_steps(path):
    return sum(max(1, int(math.ceil(cost_len(a, b) - 1e-9))) for a, b in zip(path[:-1], path[1:]))


def improve_path(model, q0, qg, path, rng, first_iters, scale=1.0):
    """Best-of-k: rerun RRT a few times and keep the path with the fewest steps."""
    best = path
    bs = path_steps(path)
    lb = max(1, int(math.ceil(cost_len(q0, qg))))
    if bs <= lb * 1.15 + 1:
        return best
    if scale < 0.3:
        return best
    lim = max(150, int(1.5 * first_iters)) * scale
    for _ in range(IMPROVE_K):
        p = rrt_connect(model, q0, qg, rng, time_limit=lim / ITERS_PER_SEC)
        if p is None:
            continue
        p = shortcut(model, p, rng)
        s = path_steps(p)
        if s < bs:
            best, bs = p, s
            if bs <= lb * 1.15 + 1:
                break
    return best



def resample_path(path, model):
    """Greedy max-step execution of a path (list of configs)."""
    dense = [np.asarray(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        d = np.asarray(b, float) - a
        d[2] = wrap(d[2])
        n = int(math.ceil(max(abs(d[0]) / 0.004, abs(d[1]) / 0.004, abs(d[2]) / 0.012,
                              abs(d[3]) / 0.004, 1)))
        base = dense[-1]
        for k in range(1, n + 1):
            dense.append(base + d * (k / n))
    dense = np.array(dense)  # theta unwrapped (continuous)
    lim = np.array([MAX_DXY, MAX_DXY, MAX_DTH, MAX_DARM])
    acts = []
    i = 0
    N = len(dense)
    while i < N - 1:
        j = i + 1
        while j + 1 < N and (np.abs(dense[j + 1] - dense[i]) <= lim).all():
            j += 1
        if (np.abs(dense[j] - dense[i]) > lim).any():
            # single dense step too large (should not happen) -> split
            d = dense[j] - dense[i]
            n = int(math.ceil((np.abs(d) / lim).max()))
            for k in range(n):
                acts.append(d / n)
            i = j
            continue
        jj = j
        while jj > i + 1 and not model.edge_free(dense[i], dense[jj]):
            jj = max(i + 1, (i + jj) // 2 if jj - i > 8 else jj - 1)
        acts.append(dense[jj] - dense[i])
        i = jj
    return acts


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ state
    def _parse(self, state):
        objs = {}
        for o in state.data:
            objs[o.name] = o
        self.robot = objs['robot']
        r = self.robot
        self.q = np.array([state.get(r, 'x'), state.get(r, 'y'),
                           state.get(r, 'theta'), state.get(r, 'arm_joint')], dtype=float)
        self.q[3] = min(max(self.q[3], ARM_MIN), ARM_MAX)
        self.rects = {}
        for name, o in objs.items():
            if name == 'robot':
                continue
            try:
                vals = [state.get(o, f) for f in ('x', 'y', 'theta', 'width', 'height')]
            except Exception:
                continue
            self.rects[name] = vals

    def corners(self, name):
        x, y, th, w, h = self.rects[name]
        return rect_corners(x, y, th, w, h)

    def center(self, name):
        return self.corners(name).mean(axis=0)

    def obstacle_names(self, exclude=()):
        return [n for n in self.rects if n != 'target_region' and n not in exclude]

    def make_obstacles(self, exclude=(), soft=()):
        names = self.obstacle_names(exclude)
        sset = set(soft) | self.soft_now
        caps = [0.001 if n in sset else np.inf for n in names]
        return Obstacles([self.corners(n) for n in names], caps), names

    def compute_soft_now(self):
        self.soft_now = set()
        names = self.obstacle_names(exclude=(self.held,) if self.held else ())
        obs = Obstacles([self.corners(n) for n in names])
        m = Model(obs, self.held_local if self.held else None, margin=MARGIN + 0.015)
        hit = m.per_obj(self.q[None])[0]
        self.soft_now = {names[i] for i in np.nonzero(hit)[0]}

    # ------------------------------------------------------------------ api
    def reset(self, state, info):
        self.t_start = time.time()
        _DEADLINE[0] = self.t_start + 48.0
        self.queue = []          # list of (action, expected_q)
        self.held = None         # name of held object
        self.held_local = None
        self.mode = 'idle'
        self.pending = None      # post-queue callback info
        self.dump_blacklist = set()
        self.fail_count = {}
        self.fallback_count = {}
        self.dump_ok_pending = False
        self.expected = None
        self.priority_remove = []
        self.cached_carry = None
        self.n_mismatch = 0
        self.soft_now = set()
        self.dumped = set()
        Model.extra = 0.0
        self._parse(state)

    def tscale(self):
        e = self.elapsed()
        if e < 16:
            return 1.0
        if e < 26:
            return 0.5
        if e < 36:
            return 0.25
        if e < 45:
            return 0.12
        return 0.05

    def elapsed(self):
        return time.time() - self.t_start

    def get_action(self, state):
        try:
            return self._get_action(state)
        except Exception:
            if DEBUG:
                raise
            self.queue = []
            self.pending = None
            self.cached_carry = None
            self.expected = None
            return self._act(np.zeros(4), 1.0 if self.held else 0.0)

    def _get_action(self, state):
        self._parse(state)
        # verify expectation of last action
        if self.queue and self.expected is not None:
            if np.abs(self.q[:2] - self.expected[:2]).max() > 1e-3 or \
                    abs(wrap(self.q[2] - self.expected[2])) > 1e-3:
                self.n_mismatch += 1
                Model.extra = min(0.003 * self.n_mismatch, 0.012)
                if DEBUG: print('MISMATCH q', self.q.round(4), 'expected', self.expected.round(4), 'held', self.held)
                lq = getattr(self, 'last_mm_q', None)
                if lq is not None and np.abs(lq - self.q).max() < 1e-4:
                    self.stuck = getattr(self, 'stuck', 0) + 1
                else:
                    self.stuck = 1
                self.last_mm_q = self.q.copy()
                self.queue = []
                self.pending = None
                self.cached_carry = None
        if self.pending == 'grasped' and not self.queue:
            self._record_grasp()
            self.pending = None
        elif self.pending == 'released' and not self.queue:
            if self.held is not None:
                self.dumped.add(self.held)
                if self.dump_ok_pending:
                    # scene changed: previous failures may no longer apply
                    self.fail_count = {k: max(0.0, v - 1.0) for k, v in self.fail_count.items()}
            self.held = None
            self.held_local = None
            self.pending = None
        if self.held is not None and not self._held_consistent():
            self.held = None
            self.held_local = None
            self.queue = []
        if not self.queue and getattr(self, 'stuck', 0) >= 2:
            self.stuck = 0
            self.expected = None
            return self._escape()
        if not self.queue:
            self._plan()
        if not self.queue:
            self.expected = None
            return self._act(np.zeros(4), 1.0 if self.held else 0.0)
        d, vac, exp = self.queue.pop(0)
        self.expected = exp
        return self._act(d, vac)

    def _escape(self):
        """Small move maximizing clearance (used when env keeps rejecting planned moves)."""
        names = self.obstacle_names(exclude=(self.held,) if self.held else ())
        obs = Obstacles([self.corners(n) for n in names])
        model = Model(obs, self.held_local if self.held else None, margin=0.0)
        cands = []
        for r in (0.012, 0.025):
            for k in range(16):
                a = 2 * math.pi * k / 16
                cands.append([r * math.cos(a), r * math.sin(a), 0.0, 0.0])
            cands.append([0.0, 0.0, 0.0, -r])
            cands.append([0.0, 0.0, 4 * r, 0.0])
            cands.append([0.0, 0.0, -4 * r, 0.0])
        D = np.array(cands)
        Q = self.q[None] + D
        Q[:, 3] = np.clip(Q[:, 3], ARM_MIN, ARM_MAX)
        Qm = self.q[None] + 0.5 * (Q - self.q[None])
        best = None
        for m in (0.03, 0.02, 0.012, 0.006, 0.003, 0.0):
            model.m = m
            ok = ~(model.hits(Q) | model.hits(Qm))
            if ok.any():
                best = int(np.nonzero(ok)[0][0])
                break
        vac = 1.0 if self.held else 0.0
        if best is None:
            return self._act(np.zeros(4), vac)
        d = Q[best] - self.q
        if DEBUG: print('ESCAPE', d.round(3), 'm', m)
        return self._act(d, vac)

    def _act(self, d, vac):
        a = np.array([d[0], d[1], d[2], d[3], vac], dtype=np.float32)
        lo = np.asarray(self.action_space.low, dtype=np.float32)
        hi = np.asarray(self.action_space.high, dtype=np.float32)
        return np.clip(a, lo, hi)

    # ------------------------------------------------------------------ grasp
    def local_of(self, name, q):
        C = self.corners(name)
        c, s = math.cos(q[2]), math.sin(q[2])
        d = C - q[None, :2]
        lx = c * d[:, 0] + s * d[:, 1] - q[3]
        ly = -s * d[:, 0] + c * d[:, 1]
        return np.stack([lx, ly], axis=1)

    def _record_grasp(self):
        name = self.grasp_target
        self.held = name
        self.held_local = self.local_of(name, self.q)
        self.held_q = self.q.copy()

    def _held_consistent(self):
        if self.held_local is None:
            return True
        q = self.q
        pl = self.held_local.copy()
        pl[:, 0] += q[3]
        W = local_to_world(q[None], pl)[0]
        C = self.corners(self.held)
        # compare as sets of corners (same ordering expected)
        return np.abs(W - C).max() < 5e-3

    def grasp_configs(self, name):
        """Candidate robot configs to grasp rect `name`."""
        C = self.corners(name)
        cfgs = []
        for k in range(4):
            p1 = C[k]
            p2 = C[(k + 1) % 4]
            e = p2 - p1
            L = np.linalg.norm(e)
            e = e / L
            n = np.array([e[1], -e[0]])  # outward normal for CCW polygon
            mid = (p1 + p2) / 2
            th = math.atan2(-n[1], -n[0])
            offs = [0.0]
            if L > 0.09:
                off = (L - 0.07) / 2
                offs += [off, -off]
            for arm in (ARM_MAX, 0.15, ARM_MIN):
                for o in offs:
                    p = mid + e * o + n * (arm + GRIP_W / 2 + GRASP_GAP)
                    cfgs.append(np.array([p[0], p[1], th, arm]))
        return cfgs

    def grasp_ok(self, name, q, model_other):
        """q must be collision free vs everything and gripper must not be near
        other objects (avoid multiple grasps)."""
        if not model_other.free(q):
            return False
        return True

    def grasp_candidates(self, name):
        obs_all, _ = self.make_obstacles(exclude=(self.held,) if self.held else (), soft=(name,))
        model = Model(obs_all, None)
        others, _ = self.make_obstacles(exclude=(name,))
        cands = []
        for q in self.grasp_configs(name):
            if not model.free(q):
                continue
            gp = model.body_polys(q[None])[0]
            if others.n and others.poly_hit(gp, 0.022)[0]:
                continue
            cands.append(q)
        return cands

    def retreat_ok(self, name, q):
        obs, _ = self.make_obstacles(exclude=(name,))
        hl = self.local_of(name, q)
        model = Model(obs, hl, margin=0.002)
        c, s = math.cos(q[2]), math.sin(q[2])
        ks = np.linspace(0.02, 0.2, 10)
        Q = np.repeat(q[None], len(ks), axis=0)
        Q[:, 0] -= ks * c
        Q[:, 1] -= ks * s
        return not model.hits(Q).any()

    def iter_grasp_paths(self, name, time_budget=2.0, cands=None, maxc=6):
        time_budget *= self.tscale()
        obs_all, _ = self.make_obstacles(exclude=(self.held,) if self.held else (), soft=(name,))
        model = Model(obs_all, None)
        if cands is None:
            cands = self.grasp_candidates(name)
        if not cands:
            return
        rok = {id(q): self.retreat_ok(name, q) for q in cands}
        cands = sorted(cands, key=lambda q: (not rok[id(q)], np.linalg.norm(q[:2] - self.q[:2])))
        t0 = vclock()
        for q in cands[:maxc]:
            rem = time_budget - (vclock() - t0)
            if rem <= 0:
                break
            it0 = _ITERS[0]
            path = rrt_connect(model, self.q, q, self.rng, time_limit=min(rem, 0.8))
            if path is not None:
                self._last_grasp_iters = _ITERS[0] - it0
                yield shortcut(model, path, self.rng)

    def plan_grasp(self, name, time_budget=2.0, cands=None):
        for p in self.iter_grasp_paths(name, time_budget, cands):
            return p
        return None

    def enqueue_path(self, path, vac, soft=()):
        if self.held is not None and vac > 0.5:
            obs, _ = self.make_obstacles(exclude=(self.held,))
            model = Model(obs, self.held_local)
        else:
            obs, _ = self.make_obstacles(soft=soft)
            model = Model(obs, None)
        q = self.q.copy()
        for d in resample_path(path, model):
            q = q + d
            q[2] = wrap(q[2])
            self.queue.append((d, vac, q.copy()))

    # ------------------------------------------------------------------ place
    def place_configs(self, hl=None, arm0=None, q0=None):
        R = self.corners('target_region')
        rc = R.mean(axis=0)
        rth = self.rects['target_region'][2]
        if hl is None:
            hl = self.held_local
        if arm0 is None:
            arm0 = self.q[3]
        # block local center & orientation in gripper frame
        bc = hl.mean(axis=0)
        e = hl[1] - hl[0]
        bth_local = math.atan2(e[1], e[0])
        cfgs = []
        if q0 is not None:
            # shift goal toward the carry start within the slack of the region
            ll = bc.copy()
            ll[0] += q0[3]
            c0, s0 = math.cos(q0[2]), math.sin(q0[2])
            bw = q0[:2] + np.array([c0 * ll[0] - s0 * ll[1], s0 * ll[0] + c0 * ll[1]])
            ru = np.array([math.cos(rth), math.sin(rth)])
            rv = np.array([-math.sin(rth), math.cos(rth)])
            d = bw - rc
            rcs = [rc + ru * float(np.clip(d @ ru, -PLACE_SLACK, PLACE_SLACK)) +
                   rv * float(np.clip(d @ rv, -PLACE_SLACK, PLACE_SLACK)), rc]
        else:
            rcs = [rc]
        for rc in rcs:
            for k in range(4):
                bth = rth + k * math.pi / 2
                th = wrap(bth - bth_local)
                for arm in (arm0, ARM_MAX, ARM_MIN, 0.125, 0.15, 0.175):
                    lc = np.array([bc[0] + arm, bc[1]])
                    c, s = math.cos(th), math.sin(th)
                    p = rc - np.array([c * lc[0] - s * lc[1], s * lc[0] + c * lc[1]])
                    cfgs.append(np.array([p[0], p[1], th, arm]))
        return np.array(cfgs)

    def plan_place(self, time_budget=4.0, q0=None, hl=None):
        time_budget *= self.tscale()
        q0 = self.q if q0 is None else q0
        hl = self.held_local if hl is None else hl
        obs, _ = self.make_obstacles(exclude=('target_block',))
        model = Model(obs, hl)
        P = self.place_configs(hl, q0[3], q0)
        cands = list(P[~model.hits(P)])
        if not cands:
            model = Model(obs, hl, margin=0.001)
            cands = list(P[~model.hits(P)])
        cands.sort(key=lambda q: np.linalg.norm(q[:2] - q0[:2]) + 0.1 * abs(wrap(q[2] - q0[2])))
        t0 = vclock()
        for q in cands[:8]:
            rem = time_budget - (vclock() - t0)
            if rem <= 0:
                break
            it0 = _ITERS[0]
            path = rrt_connect(model, q0, q, self.rng, time_limit=min(rem, 1.5))
            if path is not None:
                path = shortcut(model, path, self.rng)
                return improve_path(model, q0, q, path, self.rng, _ITERS[0] - it0, self.tscale())
        return None

    def escape_blockers(self, name=None, q0=None, hl=None):
        name = self.held if name is None else name
        q0 = self.q if q0 is None else q0
        hl = self.held_local if hl is None else hl
        obs, onames = self.make_obstacles(exclude=(name,))
        model = Model(obs, hl)
        M = 400
        D = np.column_stack([self.rng.uniform(-0.25, 0.25, M), self.rng.uniform(-0.25, 0.25, M),
                             self.rng.uniform(-0.6, 0.6, M), self.rng.uniform(-0.1, 0.1, M)])
        Q = q0[None] + D
        Q[:, 3] = np.clip(Q[:, 3], ARM_MIN, ARM_MAX)
        H = model.per_obj(Q)
        cnt = H.sum(axis=0)
        order = [onames[i] for i in np.argsort(-cnt) if cnt[i] > 0 and onames[i] != 'target_block'
                 and self.fail_count.get(onames[i], 0) < 2]
        return order[:4]

    def place_blockers(self, grasp_cands):
        """For target grasp candidates, return (good_cands, best_blocker_list)."""
        obs, onames = self.make_obstacles(exclude=('target_block',))
        good = []
        best = None
        for q in grasp_cands:
            hl = self.local_of('target_block', q)
            model = Model(obs, hl)
            P = self.place_configs(hl, q[3])
            if (~model.hits(P)).any():
                good.append(q)
                continue
            wb = model.wall_bad(P)
            H = model.per_obj(P)
            for k in range(len(P)):
                if wb[k]:
                    continue
                bl = [onames[i] for i in np.nonzero(H[k])[0]]
                if best is None or len(bl) < len(best):
                    best = bl
        return good, (best or [])

    # ------------------------------------------------------------------ dump
    # ------------------------------------------------------------------ connectivity
    def _conn_ctx(self, name):
        """Grid context for checking dump sites (object `name` removed)."""
        if _ndi is None:
            return None
        names = self.obstacle_names(exclude=(name,))
        polys = [self.corners(n) for n in names]
        if 'target_block' not in names and name != 'target_block' and 'target_block' in self.rects:
            polys.append(self.corners('target_block'))
        o = Obstacles(polys)
        xs, clr = clearance_grid(o)
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        r = BASE_R + 0.003
        rc = self.center('target_region')
        tb = []
        if name != 'target_block':
            for q in self.grasp_configs('target_block'):
                tb.append(q[:2])
        tb = np.array(tb) if tb else np.zeros((0, 2))
        ctx = dict(xs=xs, clr=clr, pts=pts, r=r, rc=rc, tb=tb)
        ctx['base'] = self._conn_eval(ctx, None, None)
        return ctx

    def _cell(self, ctx, p):
        xs = ctx['xs']
        n = len(xs)
        i = int(round((p[0] - xs[0]) / GRID_RES))
        j = int(round((p[1] - xs[0]) / GRID_RES))
        return min(max(i, 0), n - 1), min(max(j, 0), n - 1)

    def _label_near(self, ctx, lab, p, rad=1):
        i, j = self._cell(ctx, p)
        n = lab.shape[0]
        best = 0
        bd = 1e9
        for di in range(-rad, rad + 1):
            for dj in range(-rad, rad + 1):
                a, b = i + di, j + dj
                if 0 <= a < n and 0 <= b < n and lab[a, b] > 0:
                    d = di * di + dj * dj
                    if d < bd:
                        bd = d
                        best = lab[a, b]
        return best

    def _conn_eval(self, ctx, poly, qbase):
        clr = ctx['clr']
        if poly is not None:
            P = np.asarray(poly, float)
            o = Obstacles([P])
            d = ctx['pts'][:, None, :] - o.P[None, :, 0, :]
            lu = np.einsum('mnd,nd->mn', d, o.u)
            lv = np.einsum('mnd,nd->mn', d, o.v)
            du = np.maximum(np.maximum(-lu, lu - o.w[None]), 0)
            dv = np.maximum(np.maximum(-lv, lv - o.h[None]), 0)
            dist = np.sqrt(du ** 2 + dv ** 2)[:, 0].reshape(clr.shape)
            clr = np.minimum(clr, dist)
        lab, _ = _ndi.label(clr >= ctx['r'])
        rl = self._label_near(ctx, lab, ctx['rc'], rad=4)
        if rl == 0:
            return None
        robot_ok = True
        if qbase is not None:
            robot_ok = self._label_near(ctx, lab, qbase, rad=1) == rl
        nt = sum(1 for p in ctx['tb'] if self._label_near(ctx, lab, p, rad=1) == rl)
        return robot_ok, nt

    def _dump_ok(self, ctx, poly, qbase):
        if ctx is None or ctx['base'] is None:
            return True
        res = self._conn_eval(ctx, poly, qbase)
        if res is None:
            return False
        robot_ok, nt = res
        nt0 = ctx['base'][1]
        return robot_ok and (nt >= min(nt0, 1))

    def plan_dump(self, time_budget=3.0, q0=None, hl=None, name=None):
        time_budget *= self.tscale()
        q0 = self.q if q0 is None else q0
        hl = self.held_local if hl is None else hl
        name = self.held if name is None else name
        obs, _ = self.make_obstacles(exclude=(name,))
        model = Model(obs, hl, margin=0.01)
        tc = self.center('target_block')
        rc = self.center('target_region')
        hc_local = hl.mean(axis=0)
        M = 1000
        Q = np.column_stack([
            self.rng.uniform(BASE_R, WORLD - BASE_R, M),
            self.rng.uniform(BASE_R, WORLD - BASE_R, M),
            self.rng.uniform(-math.pi, math.pi, M),
            self.rng.uniform(ARM_MIN, ARM_MAX, M)])
        # object center in world
        lc = np.repeat(hc_local[None], M, axis=0)
        lc[:, 0] += Q[:, 3]
        c, s = np.cos(Q[:, 2]), np.sin(Q[:, 2])
        ox = Q[:, 0] + c * lc[:, 0] - s * lc[:, 1]
        oy = Q[:, 1] + s * lc[:, 0] + c * lc[:, 1]
        dt = np.hypot(ox - tc[0], oy - tc[1])
        dr = np.hypot(ox - rc[0], oy - rc[1])
        # keep away from segment between target and region
        seg = rc - tc
        L2 = max(seg @ seg, 1e-9)
        tt = np.clip(((ox - tc[0]) * seg[0] + (oy - tc[1]) * seg[1]) / L2, 0, 1)
        dseg = np.hypot(ox - (tc[0] + tt * seg[0]), oy - (tc[1] + tt * seg[1]))
        slack = np.minimum(np.minimum(dt - DUMP_DT, dr - DUMP_DR), dseg - DUMP_DSEG)
        free = ~model.hits(Q)
        chosen = None
        for relax in (0.0, 0.1, 0.2, 0.3, 0.45):
            ok = free & (slack > -relax)
            if ok.sum() >= 5 or (relax >= 0.45 and ok.any()):
                chosen = ok
                if DEBUG: print('   dump relax', relax, 'n', ok.sum(), 'free', free.sum())
                break
        if chosen is None:
            return None
        Q = Q[chosen]
        dt = dt[chosen]
        cost = np.linalg.norm(Q[:, :2] - q0[None, :2], axis=1) - 0.2 * np.minimum(dt, 1.0)
        order = np.argsort(cost)
        try:
            ctx = self._conn_ctx(name)
            if ctx is not None and ctx['base'] is not None:
                good_i, bad_i = [], []
                for i in order[:40]:
                    pl = hl.copy()
                    pl[:, 0] += Q[i, 3]
                    W = local_to_world(Q[i][None], pl)[0]
                    (good_i if self._dump_ok(ctx, W, Q[i, :2]) else bad_i).append(i)
                    if len(good_i) >= 8:
                        break
                seen = set(good_i) | set(bad_i)
                order = np.array(good_i + bad_i + [i for i in order if i not in seen], dtype=int)
        except Exception:
            if DEBUG:
                raise
        model2 = Model(obs, hl)
        t0 = vclock()
        for i in order[:8]:
            rem = time_budget - (vclock() - t0)
            if rem <= 0:
                break
            it0 = _ITERS[0]
            path = rrt_connect(model2, q0, Q[i], self.rng, time_limit=min(rem, 0.8))
            if path is not None:
                path = shortcut(model2, path, self.rng)
                return improve_path(model2, q0, Q[i], path, self.rng, _ITERS[0] - it0, self.tscale())
        return None

    # ------------------------------------------------------------------ main planner
    def _plan(self):
        self.compute_soft_now()
        if self.held is not None:
            carry = self.cached_carry
            self.cached_carry = None
            if self.held == 'target_block':
                path = carry if carry is not None else self.plan_place()
                if path is not None:
                    self.enqueue_path(path, 1.0)
                    return
                self.priority_remove = self.escape_blockers()
                if DEBUG: print('escape blockers', self.priority_remove)
                self._release()
                return
            path = carry if carry is not None else self.plan_dump()
            self.dump_ok_pending = path is not None
            if path is not None:
                self.enqueue_path(path, 1.0)
            else:
                self.fail_count[self.held] = self.fail_count.get(self.held, 0) + 1
                self.priority_remove = self.escape_blockers()
                if DEBUG: print('dump failed; escape blockers', self.priority_remove)
            self._release()
            return
        order = [n for n in self.priority_remove if n in self.rects]
        prio = bool(order)
        good = []
        self.priority_remove = []
        if not order:
            cands = self.grasp_candidates('target_block')
            good, pblock = self.place_blockers(cands)
            esc = []
            if good:
                ntry = 0
                for path in self.iter_grasp_paths('target_block', 3.0, good, maxc=6):
                    qg = path[-1]
                    hl = self.local_of('target_block', qg)
                    carry = self.plan_place(2.0, qg, hl)
                    if carry is not None:
                        self._go_grasp(path, 'target_block', carry)
                        return
                    for n in self.escape_blockers('target_block', qg, hl):
                        if n not in esc:
                            esc.append(n)
                    ntry += 1
                    if ntry >= 2:
                        break
                if DEBUG: print('target carry failed; escape blockers', esc)
            order = list(esc)
            if not good and pblock:
                order += sorted(pblock, key=lambda n: np.linalg.norm(self.center(n) - self.q[:2]))
            for n in self.removal_order():
                if n not in order:
                    order.append(n)
        # blockers of ungraspable blockers go first
        try:
            exp = []
            for k, n in enumerate(order):
                if k < 3 and n not in exp and not self.grasp_candidates(n):
                    for b in self.removal_order(n, best_only=True):
                        if b != 'target_block' and b not in exp and b in self.rects:
                            exp.append(b)
                if n not in exp:
                    exp.append(n)
            order = exp
        except Exception:
            if DEBUG:
                raise
        order = [n for n in order if self.fail_count.get(n, 0) < 2 and n not in self.dumped] + \
            [n for n in order if self.fail_count.get(n, 0) < 2 and n in self.dumped] + \
            [n for n in order if self.fail_count.get(n, 0) >= 2]
        if DEBUG: print('  order', order[:8], 'good', len(good) if not prio else 'prio')
        t0 = vclock()
        for k, name in enumerate(order[:6]):
            if vclock() - t0 > 6.0 * self.tscale():
                break
            gb = 2.5 if k == 0 else (1.5 if k == 1 else 1.0)
            for path in self.iter_grasp_paths(name, gb, maxc=3):
                qg = path[-1]
                hl = self.local_of(name, qg)
                carry = self.plan_dump(1.0, qg, hl, name)
                if carry is not None:
                    self._go_grasp(path, name, carry)
                    return
                break
            self.fail_count[name] = self.fail_count.get(name, 0) + 0.5
        # fallback: try grasping target anyway (limited repeats to avoid loops)
        fb = self.fallback_count
        names = ['target_block'] + [n for n in order if n != 'target_block']
        names = sorted(names, key=lambda n: fb.get(n, 0))
        for name in names:
            if fb.get(name, 0) >= 3 and name != names[0]:
                continue
            path = self.plan_grasp(name, time_budget=1.5 if name == 'target_block' else 0.5)
            if path is not None:
                fb[name] = fb.get(name, 0) + 1
                self._go_grasp(path, name)
                return
        # last resort: wander to a random nearby free config
        model = Model(self.make_obstacles()[0], None)
        for _ in range(50):
            q = self.q + np.array([*self.rng.normal(0, 0.3, 2), self.rng.normal(0, 1.0), 0.0])
            q[3] = ARM_MIN
            q[2] = wrap(q[2])
            if model.free(q):
                path = rrt_connect(model, self.q, q, self.rng, time_limit=0.3)
                if path is not None:
                    self.enqueue_path(shortcut(model, path, self.rng), 0.0)
                    return

    def _go_grasp(self, path, name, carry=None):
        self.cached_carry = carry
        try:
            obs_all, _ = self.make_obstacles(exclude=(self.held,) if self.held else (), soft=(name,))
            model = Model(obs_all, None)
            path = improve_path(model, self.q, path[-1], path, self.rng,
                                getattr(self, '_last_grasp_iters', 200), self.tscale())
        except Exception:
            if DEBUG:
                raise
        self.enqueue_path(path, 0.0, soft=(name,))
        if self.queue:
            d, _, e = self.queue[-1]
            self.queue[-1] = (d, 1.0, e)
        else:
            self.queue.append((np.zeros(4), 1.0, None))
        self.grasp_target = name
        self.pending = 'grasped'

    def _release(self):
        if self.queue and self.queue[-1][1] > 0.5 and self.queue[-1][2] is not None:
            d, _, e = self.queue[-1]
            self.queue[-1] = (d, 0.0, e)
        else:
            self.queue.append((np.zeros(4), 0.0, None))
        self.pending = 'released'

    def removal_order(self, target='target_block', best_only=False):
        """Obstructions ordered by how much they block grasping the target."""
        names = [n for n in self.obstacle_names(exclude=(target,)) if n != 'target_block']
        if not names:
            return []
        obs, onames = self.make_obstacles(exclude=(target,))
        count = {n: 0.0 for n in onames}
        cfgs = self.grasp_configs(target)
        best_blockers = None
        best = 1e9
        for q in cfgs:
            Q = q[None]
            m = Model(obs, None)
            hit = np.zeros(obs.n, bool)
            # per-object check
            d = Q[:, None, :2] - obs.P[None, :, 0, :]
            lu = np.einsum('mnd,nd->mn', d, obs.u)
            lv = np.einsum('mnd,nd->mn', d, obs.v)
            cu = np.clip(lu, 0, obs.w[None])
            cv = np.clip(lv, 0, obs.h[None])
            hit |= (((lu - cu) ** 2 + (lv - cv) ** 2) < (BASE_R + 0.03) ** 2)[0]
            gp = m.body_polys(Q)[0]
            hit |= obs.poly_hit(gp, 0.025, per_obj=True)[0]
            # swept approach corridor: points behind the robot
            th = q[2]
            back = np.array([-math.cos(th), -math.sin(th)])
            for k in (0.1, 0.2):
                Qb = Q.copy()
                Qb[0, :2] += back * k
                d = Qb[:, None, :2] - obs.P[None, :, 0, :]
                lu = np.einsum('mnd,nd->mn', d, obs.u)
                lv = np.einsum('mnd,nd->mn', d, obs.v)
                cu = np.clip(lu, 0, obs.w[None])
                cv = np.clip(lv, 0, obs.h[None])
                hit |= (((lu - cu) ** 2 + (lv - cv) ** 2) < (BASE_R + 0.01) ** 2)[0]
            x, y = q[0], q[1]
            if x < BASE_R or x > WORLD - BASE_R or y < BASE_R or y > WORLD - BASE_R:
                continue
            nb = int(hit.sum())
            for i in np.nonzero(hit)[0]:
                count[onames[i]] += 1.0 / (1 + nb)
            if nb < best:
                best = nb
                best_blockers = [onames[i] for i in np.nonzero(hit)[0]]
        if best_only:
            return best_blockers or []
        tc = self.center(target)
        order = []
        if best_blockers:
            bb = sorted(best_blockers, key=lambda n: np.linalg.norm(self.center(n) - self.q[:2]))
            order += bb
        rest = sorted([n for n in names if n not in order],
                      key=lambda n: -count[n] + 0.3 * np.linalg.norm(self.center(n) - tc))
        order += rest
        return order
