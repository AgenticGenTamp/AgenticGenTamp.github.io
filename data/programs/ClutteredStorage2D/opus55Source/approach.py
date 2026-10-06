"""Approach for ClutteredStorage2D (variable number of blocks).

Strategy: stack blocks horizontally in "columns" of the shelf slot (one column
per 0.3175 of slot width), filling from the top of each column downwards.
Blocks are always grasped from the middle of one of their long sides so that,
with the arm pointing straight up, the held block is level.  Motion between
regions is planned with a BiRRT in (x, y, theta) using an exact re-implementation
of the env's collision geometry.
"""

import math
import time

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
WORLD_W, WORLD_H = 5.0, 3.0
SHELF_Y = 2.625
COL_W = 0.3175
GRIP_W = 0.02  # depth along arm
GRIP_H = 0.14  # width across arm
ARM_MIN, ARM_MAX = 0.2, 0.8
BASE_R = 0.2
GAP = 0.015  # gap between gripper face and block surface when grasping
LAYER_GAP = 0.012
TOP_MARGIN = 0.008
BASE_Y_PLACE = 2.34

MAX_DX = 0.05
MAX_DTH = np.pi / 16
MAX_DARM = 0.1
SDX, SDTH, SDARM = 0.0499, 0.1960, 0.0998


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def rect_corners(x, y, th, w, h):
    c, s = math.cos(th), math.sin(th)
    return [
        (x, y),
        (x + w * c, y + w * s),
        (x + w * c - h * s, y + w * s + h * c),
        (x - h * s, y + h * c),
    ]


def rect_from_center(cx, cy, w, h, th):
    c, s = math.cos(th), math.sin(th)
    x = cx - (w / 2 * c - h / 2 * s)
    y = cy - (w / 2 * s + h / 2 * c)
    return rect_corners(x, y, th, w, h)


class Poly:
    __slots__ = ("pts", "axes", "minx", "maxx", "miny", "maxy", "name")

    def __init__(self, pts, name=None):
        self.pts = pts
        self.name = name
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.minx, self.maxx, self.miny, self.maxy = min(xs), max(xs), min(ys), max(ys)
        axes = []
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            L = math.hypot(dx, dy)
            if L > 1e-12:
                axes.append((-dy / L, dx / L))
        self.axes = axes


def poly_sep(a, b):
    """Lower bound on distance between two convex polys (negative if overlap)."""
    best = -1e9
    for axes in (a.axes, b.axes):
        for ax, ay in axes:
            amin = amax = None
            for (x, y) in a.pts:
                p = x * ax + y * ay
                if amin is None or p < amin:
                    amin = p
                if amax is None or p > amax:
                    amax = p
            bmin = bmax = None
            for (x, y) in b.pts:
                p = x * ax + y * ay
                if bmin is None or p < bmin:
                    bmin = p
                if bmax is None or p > bmax:
                    bmax = p
            g = max(bmin - amax, amin - bmax)
            if g > best:
                best = g
    return best


def circle_poly_dist(cx, cy, r, poly):
    pts = poly.pts
    n = len(pts)
    inside = True
    best = 1e9
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-15 else max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / L2))
        px, py = x1 + t * dx, y1 + t * dy
        d = math.hypot(cx - px, cy - py)
        if d < best:
            best = d
        if dx * (cy - y1) - dy * (cx - x1) < 0:
            inside = False
    if inside:
        return -1.0
    return best - r


def aabb_far(a, b, m):
    return (a.minx > b.maxx + m or b.minx > a.maxx + m or a.miny > b.maxy + m
            or b.miny > a.maxy + m)


# ----------------------------------------------------------------------------
# SE2 helpers (x, y, theta)
# ----------------------------------------------------------------------------
def se2_mul(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def se2_inv(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), wrap(-a[2]))


def tip_pose(q):
    x, y, th, arm = q
    return (x + (arm + GRIP_W / 2) * math.cos(th), y + (arm + GRIP_W / 2) * math.sin(th), th)


def base_from_tip(tip, arm):
    tx, ty, th = tip
    return (tx - (arm + GRIP_W / 2) * math.cos(th), ty - (arm + GRIP_W / 2) * math.sin(th), th, arm)


# ----------------------------------------------------------------------------
# World model
# ----------------------------------------------------------------------------
class World:
    def __init__(self, state):
        self.robot = None
        self.blocks = {}
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            t = obj.type.name
            if t == "crv_robot":
                self.robot = obj
            elif t == "shelf":
                self.shelf = obj
            elif t == "target_block":
                self.blocks[name] = obj
        sh = self.shelf
        self.x1 = state.get(sh, "x1")
        self.w1 = state.get(sh, "width1")
        self.sy = state.get(sh, "y1")
        self.sh = state.get(sh, "height1")
        self.ncols = max(1, int(round(self.w1 / COL_W)))
        self.colw = self.w1 / self.ncols
        top = self.sy + self.sh
        self.static_rects = [(-1.0, -1.0, 0.0, 1.0, 5.0), (WORLD_W, -1.0, 0.0, 1.0, 5.0),
                             (-1.0, -1.0, 0.0, 7.0, 1.0), (-1.0, WORLD_H, 0.0, 7.0, 1.0)]
        if self.x1 > 1e-6:
            self.static_rects.append((0.0, self.sy, 0.0, self.x1, top - self.sy))
        if self.x1 + self.w1 < WORLD_W - 1e-6:
            self.static_rects.append((self.x1 + self.w1, self.sy, 0.0, WORLD_W - self.x1 - self.w1, top - self.sy))
        static = []
        static.append(Poly(rect_corners(-1.0, -1.0, 0, 1.0, 5.0), "wall"))
        static.append(Poly(rect_corners(WORLD_W, -1.0, 0, 1.0, 5.0), "wall"))
        static.append(Poly(rect_corners(-1.0, -1.0, 0, 7.0, 1.0), "wall"))
        static.append(Poly(rect_corners(-1.0, WORLD_H, 0, 7.0, 1.0), "wall"))
        if self.x1 > 1e-6:
            static.append(Poly(rect_corners(0.0, self.sy, 0, self.x1, top - self.sy), "shelf"))
        if self.x1 + self.w1 < WORLD_W - 1e-6:
            static.append(Poly(rect_corners(self.x1 + self.w1, self.sy, 0,
                                            WORLD_W - self.x1 - self.w1, top - self.sy), "shelf"))
        self.static = static
        self.update(state)

    def update(self, state):
        self.bpose = {}
        self.bsize = {}
        for n, o in self.blocks.items():
            self.bpose[n] = (state.get(o, "x"), state.get(o, "y"), state.get(o, "theta"))
            self.bsize[n] = (state.get(o, "width"), state.get(o, "height"))
        r = self.robot
        self.q = (state.get(r, "x"), state.get(r, "y"), state.get(r, "theta"), state.get(r, "arm_joint"))
        self.vac = state.get(r, "vacuum")

    def block_poly(self, n, pose=None):
        p = self.bpose[n] if pose is None else pose
        w, h = self.bsize[n]
        return Poly(rect_corners(p[0], p[1], p[2], w, h), n)

    def block_center(self, n, pose=None):
        p = self.bpose[n] if pose is None else pose
        w, h = self.bsize[n]
        c, s = math.cos(p[2]), math.sin(p[2])
        return (p[0] + w / 2 * c - h / 2 * s, p[1] + w / 2 * s + h / 2 * c)

    def inside_slot(self, n, pose=None, tol=0.0):
        poly = self.block_poly(n, pose)
        for x, y in poly.pts:
            if not (self.x1 - tol <= x <= self.x1 + self.w1 + tol and self.sy - tol <= y <= self.sy + self.sh + tol):
                return False
        return True

    def col_center(self, i):
        return self.x1 + self.colw * (i + 0.5)

    def layer_bottom(self, k, h=0.04):
        return self.sy + self.sh - TOP_MARGIN - (k + 1) * h - k * LAYER_GAP

    def max_layers(self, h=0.04):
        k = 0
        while self.layer_bottom(k, h) >= self.sy + 0.005:
            k += 1
        return k


def _rect_arrays(rects):
    """rects: list of (x, y, th, w, h) corner-poses -> origin, axes, dims, verts arrays."""
    r = np.asarray(rects, dtype=float).reshape(-1, 5)
    c, s = np.cos(r[:, 2]), np.sin(r[:, 2])
    u = np.stack([c, s], -1)
    v = np.stack([-s, c], -1)
    o = r[:, :2]
    w = r[:, 3:4]
    h = r[:, 4:5]
    verts = np.stack([o, o + w * u, o + w * u + h * v, o + h * v], 1)
    axes = np.stack([u, v], 1)
    return o, axes, r[:, 3:5], verts


def _sat_pairs(PV, PA, OV, OA):
    """Pairwise SAT separation for P pairs: PV (P,4,2), PA (P,2,2), OV, OA same -> (P,)."""
    axes = np.concatenate([PA, OA], 1)  # (P,4,2)
    pp = PV[:, None, :, 0] * axes[:, :, None, 0] + PV[:, None, :, 1] * axes[:, :, None, 1]
    po = OV[:, None, :, 0] * axes[:, :, None, 0] + OV[:, None, :, 1] * axes[:, :, None, 1]
    gap = np.maximum(po.min(-1) - pp.max(-1), pp.min(-1) - po.max(-1))
    return gap.max(-1)


def _rects_collide(PV, PA, OV, OA, Omin, Omax, m):
    """Return (K,) bool: whether body rect k collides (sep < m) with any obstacle."""
    Pmin = PV.min(1)
    Pmax = PV.max(1)
    ov = ((Pmin[:, None, :] <= Omax[None] + m) & (Pmax[:, None, :] >= Omin[None] - m)).all(-1)
    ki, ni = np.nonzero(ov)
    out = np.zeros(PV.shape[0], dtype=bool)
    if len(ki) == 0:
        return out
    sep = _sat_pairs(PV[ki], PA[ki], OV[ni], OA[ni])
    hit = sep < m
    out[ki[hit]] = True
    return out


class Checker:
    """Vectorized collision checker for robot configs against fixed obstacles."""

    def __init__(self, world, exclude=(), held=None, margin=0.006):
        self.world = world
        self.held = held
        self.margin = margin
        rects = []
        for p in world.static_rects:
            rects.append(p)
        nstatic = len(rects)
        for n in world.blocks:
            if n in exclude or (held is not None and n == held[0]):
                continue
            x, y, th = world.bpose[n]
            w, h = world.bsize[n]
            rects.append((x, y, th, w, h))
        self.O, self.OA, self.OD, self.OV = _rect_arrays(rects)
        self.Omin = self.OV.min(1)
        self.Omax = self.OV.max(1)
        self.nstatic = nstatic
        self.has_blocks = len(rects) > nstatic
        if held is not None:
            self.hw, self.hh = world.bsize[held[0]]

    def _check(self, Q, m):
        """Return boolean array (K,) of collisions."""
        x, y, th, arm = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
        d = Q[:, None, :2] - self.O[None]  # (K,N,2)
        loc = np.einsum("knd,nad->kna", d, self.OA)
        cl = np.clip(loc, 0.0, self.OD[None])
        dist = np.sqrt(((loc - cl) ** 2).sum(-1))
        coll = (dist - BASE_R < m).any(1)
        c, s = np.cos(th), np.sin(th)
        U = np.stack([c, s], -1)
        V = np.stack([-s, c], -1)
        PA = np.stack([U, V], 1)
        A01 = np.array([0.0, 1.0, 1.0, 0.0])
        B01 = np.array([0.0, 0.0, 1.0, 1.0])
        U4 = U[:, None, :]
        V4 = V[:, None, :]
        base = Q[:, None, :2]
        g0 = base + (arm - GRIP_W / 2)[:, None, None] * U4 - (GRIP_H / 2) * V4
        GV = g0 + (GRIP_W * A01)[None, :, None] * U4 + (GRIP_H * B01)[None, :, None] * V4
        coll |= _rects_collide(GV, PA, self.OV, self.OA, self.Omin, self.Omax, m)
        a0 = base - (GRIP_W / 4) * V4
        AV = a0 + (arm[:, None] * A01[None, :])[:, :, None] * U4 + ((GRIP_W / 2) * B01)[None, :, None] * V4
        coll |= _rects_collide(AV, PA, self.OV, self.OA, self.Omin, self.Omax, m)
        if self.held is not None:
            rx, ry, rth = self.held[1]
            tip = base + (arm + GRIP_W / 2)[:, None, None] * U4
            b0 = tip + rx * U4 + ry * V4
            c2, s2 = math.cos(rth), math.sin(rth)
            U2 = c2 * U + s2 * V
            V2 = -s2 * U + c2 * V
            BV = b0 + (self.hw * A01)[None, :, None] * U2[:, None, :] + (self.hh * B01)[None, :, None] * V2[:, None, :]
            coll |= _rects_collide(BV, np.stack([U2, V2], 1), self.OV, self.OA, self.Omin, self.Omax, m)
            if self.has_blocks:
                s0_ = base + (arm + GRIP_W)[:, None, None] * U4 - (GRIP_H / 2) * V4
                SV = s0_ + (GRIP_W * A01)[None, :, None] * U4 + (GRIP_H * B01)[None, :, None] * V4
                k = self.nstatic
                coll |= _rects_collide(SV, PA, self.OV[k:], self.OA[k:], self.Omin[k:], self.Omax[k:], 0.003)
        return coll

    def collides(self, q, margin=None):
        m = self.margin if margin is None else margin
        return bool(self._check(np.asarray([q], dtype=float), m)[0])

    def path_ok(self, qs, margin=None):
        if len(qs) == 0:
            return True
        m = self.margin if margin is None else margin
        Q = np.asarray(qs, dtype=float)
        # check in chunks, early exit
        for i in range(0, len(Q), 64):
            if self._check(Q[i:i + 64], m).any():
                return False
        return True


# ----------------------------------------------------------------------------
# Path utilities
# ----------------------------------------------------------------------------
def densify(qa, qb):
    dx, dy = qb[0] - qa[0], qb[1] - qa[1]
    dth = wrap(qb[2] - qa[2])
    da = qb[3] - qa[3]
    n = int(math.ceil(max(abs(dx) / SDX, abs(dy) / SDX, abs(dth) / SDTH, abs(da) / SDARM, 1e-9)))
    n = max(n, 1)
    out = []
    for i in range(1, n + 1):
        t = i / n
        out.append((qa[0] + t * dx, qa[1] + t * dy, wrap(qa[2] + t * dth), qa[3] + t * da))
    return out


def densify_path(path):
    out = []
    for a, b in zip(path[:-1], path[1:]):
        out.extend(densify(a, b))
    return out


def tip_rotate_path(q, th_target):
    """Rotate about the tool tip (keep arm) from q to theta=th_target."""
    tip = tip_pose(q)
    dth = wrap(th_target - q[2])
    n = max(1, int(math.ceil(abs(dth) / 0.02)))
    pts = [q]
    for i in range(1, n + 1):
        th = wrap(q[2] + dth * i / n)
        pts.append(base_from_tip((tip[0], tip[1], th), q[3]))
    return densify_path(pts)


def qdist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) + 0.25 * abs(wrap(a[2] - b[2]))


def birrt(checker, start, goal, rng, arm, max_iters=3000, time_limit=6.0, margin=None):
    """BiRRT over (x, y, theta) with fixed arm; returns list of q (sparse)."""
    t0 = time.time()
    if checker.collides(goal, margin) or checker.collides(start, 0.0):
        return None
    direct = densify(start, goal)
    if checker.path_ok(direct, margin):
        return [start, goal]
    xlo, xhi, ylo, yhi = 0.2, WORLD_W - 0.2, 0.2, SHELF_Y - 0.2
    ta = [(start, -1)]
    tb = [(goal, -1)]
    step = 0.3

    def nearest(tree, q):
        best, bi = 1e9, 0
        for i, (p, _) in enumerate(tree):
            d = qdist(p, q)
            if d < best:
                best, bi = d, i
        return bi

    def steer(qa, qb):
        d = qdist(qa, qb)
        if d <= step:
            return qb
        t = step / d
        return (qa[0] + t * (qb[0] - qa[0]), qa[1] + t * (qb[1] - qa[1]),
                wrap(qa[2] + t * wrap(qb[2] - qa[2])), arm)

    def extend(tree, q):
        i = nearest(tree, q)
        qn = steer(tree[i][0], q)
        if checker.path_ok(densify(tree[i][0], qn), margin):
            tree.append((qn, i))
            return len(tree) - 1
        return None

    def connect(tree, q):
        last = None
        while True:
            i = nearest(tree, q)
            qn = steer(tree[i][0], q)
            if not checker.path_ok(densify(tree[i][0], qn), margin):
                return last
            tree.append((qn, i))
            last = len(tree) - 1
            if qdist(qn, q) < 1e-6:
                return last

    def trace(tree, i):
        out = []
        while i != -1:
            out.append(tree[i][0])
            i = tree[i][1]
        return out

    swapped = False
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            return None
        if rng.random() < 0.1:
            qr = tb[0][0]
        else:
            qr = (rng.uniform(xlo, xhi), rng.uniform(ylo, yhi), rng.uniform(-np.pi, np.pi), arm)
        ia = extend(ta, qr)
        if ia is not None:
            qa = ta[ia][0]
            ib = connect(tb, qa)
            if ib is not None and qdist(tb[ib][0], qa) < 1e-6:
                pa = trace(ta, ia)[::-1]
                pb = trace(tb, ib)
                path = pa + pb[1:]
                if swapped:
                    path = path[::-1]
                return shortcut(checker, path, rng, margin)
        ta, tb = tb, ta
        swapped = not swapped
    return None


def smooth(checker, pts, rng, margin=None, iters=80):
    """Shortcut a dense path (list of q, each step feasible); returns dense path."""
    if len(pts) <= 2:
        return list(pts)
    path = list(pts)
    # Greedy pass from the start.
    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = n - 1
        step = max(1, (n - 1 - i) // 8)
        found = i + 1
        while j > i + 1:
            if checker.path_ok(densify(path[i], path[j]), margin):
                found = j
                break
            j -= step
            step = max(1, step // 2) if j - i < 8 * step else step
        out.append(path[found])
        i = found
    path = shortcut(checker, out, rng, margin, iters=iters)
    return [path[0]] + densify_path(path)


def shortcut(checker, path, rng, margin, iters=60):
    path = list(path)
    for _ in range(iters):
        if len(path) <= 2:
            break
        i = int(rng.integers(0, len(path) - 1))
        j = int(rng.integers(0, len(path) - 1))
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        if checker.path_ok(densify(path[i], path[j]), margin):
            path = path[: i + 1] + path[j:]
    return path


# ----------------------------------------------------------------------------
# Approach
# ----------------------------------------------------------------------------
class GeneratedApproach:
    K_PICK = 14
    ENABLE_STAGING = True
    STAGE_MIN_GAIN = 2.5
    K_TRANS = 5
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.rng = np.random.default_rng(0)

    def reset(self, state, info):
        self.world = World(state)
        self.plan = []  # list of (q, vac)
        self.idx = 0
        self.last_q = None
        self.stuck = 0
        self.failed = {}
        self.t_start = time.time()

    # ------------------------------------------------------------------
    def get_action(self, state):
        w = self.world
        w.update(state)
        q = w.q
        if self.last_q is not None and self.idx < len(self.plan):
            moved = max(abs(q[0] - self.last_q[0]), abs(q[1] - self.last_q[1]),
                        abs(wrap(q[2] - self.last_q[2])), abs(q[3] - self.last_q[3]))
            if moved < 1e-7 and self._last_motion:
                self.stuck += 1
                if self.stuck >= 2:
                    self.plan, self.idx = [], 0
                    self.stuck = 0
            else:
                self.stuck = 0
        # Advance through reached waypoints
        while self.idx < len(self.plan):
            tq, tv = self.plan[self.idx]
            if self._close(q, tq) and (w.vac > 0.5) == (tv > 0.5):
                self.idx += 1
            else:
                break
        if self.idx >= len(self.plan):
            self.plan, self.idx = [], 0
            try:
                self.plan = self._make_plan() or []
            except Exception:  # pragma: no cover - safety
                self.plan = []
            if not self.plan:
                self.plan = self._random_wiggle()
        tq, tv = self.plan[self.idx]
        act = np.zeros(5, dtype=np.float32)
        act[0] = np.clip(tq[0] - q[0], -MAX_DX * 0.999, MAX_DX * 0.999)
        act[1] = np.clip(tq[1] - q[1], -MAX_DX * 0.999, MAX_DX * 0.999)
        act[2] = np.clip(wrap(tq[2] - q[2]), -MAX_DTH * 0.999, MAX_DTH * 0.999)
        act[3] = np.clip(tq[3] - q[3], -MAX_DARM * 0.999, MAX_DARM * 0.999)
        act[4] = 1.0 if tv > 0.5 else 0.0
        self._last_motion = bool(np.max(np.abs(act[:4])) > 1e-6)
        self.last_q = q
        return np.clip(act, self.action_space.low, self.action_space.high).astype(np.float32)

    @staticmethod
    def _close(a, b, tol=2e-4):
        return (abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol and abs(wrap(a[2] - b[2])) < tol
                and abs(a[3] - b[3]) < tol)

    def _k_pick(self):
        return 1 if time.time() - self.t_start > 25.0 else self.K_PICK

    def _k_trans(self):
        return 1 if time.time() - self.t_start > 25.0 else self.K_TRANS

    def _random_wiggle(self):
        w = self.world
        q = w.q
        vac = 1.0 if w.vac > 0.5 else 0.0
        ck = Checker(w, held=self._held(), margin=0.002)
        for _ in range(50):
            d = self.rng.uniform(-1, 1, size=4) * np.array([0.04, 0.04, 0.15, 0.0])
            nq = (q[0] + d[0], q[1] + d[1], wrap(q[2] + d[2]), q[3])
            if not ck.collides(nq):
                return [(nq, vac)]
        return [(q, 0.0)]

    # ------------------------------------------------------------------
    def _held(self):
        """Return (name, rel) of a block currently held (vacuum on + in suction zone)."""
        w = self.world
        if w.vac <= 0.5:
            return None
        x, y, th, arm = w.q
        c, s = math.cos(th), math.sin(th)
        sd = arm + GRIP_W + GRIP_W / 2
        zone = Poly(rect_from_center(x + sd * c, y + sd * s, GRIP_W, GRIP_H, th))
        tip = tip_pose(w.q)
        for n in w.blocks:
            bp = w.block_poly(n)
            if poly_sep(zone, bp) < 0:
                return (n, se2_mul(se2_inv(tip), w.bpose[n]))
        return None

    # ------------------------------------------------------------------
    def _column_status(self):
        """Return per-column lists of in-slot blocks sorted top-down, plus outside blocks."""
        w = self.world
        cols = [[] for _ in range(w.ncols)]
        outside = []
        for n in w.blocks:
            cx, cy = w.block_center(n)
            if w.x1 <= cx <= w.x1 + w.w1 and cy >= w.sy - 0.01:
                i = int((cx - w.x1) // w.colw)
                i = min(max(i, 0), w.ncols - 1)
                cols[i].append((cy, n))
            else:
                outside.append(n)
        for c in cols:
            c.sort(reverse=True)
        return [[n for _, n in c] for c in cols], outside

    def _is_placed(self, n, col, layer):
        w = self.world
        x, y, th = w.bpose[n]
        bw, bh = w.bsize[n]
        cx, cy = w.block_center(n)
        if abs(math.sin(th)) > 0.03:
            return False
        if abs(cx - w.col_center(col)) > 0.03:
            return False
        tb = w.layer_bottom(layer, bh)
        if abs((cy - bh / 2) - tb) > 0.012:
            return False
        return w.inside_slot(n)

    def _target_pose(self, n, col, layer, flip):
        w = self.world
        bw, bh = w.bsize[n]
        cx = w.col_center(col)
        yb = w.layer_bottom(layer, bh)
        if not flip:
            return (cx - bw / 2, yb, 0.0)
        return (cx + bw / 2, yb + bh, np.pi)

    def _choose_task(self):
        """Return list of (block, col, layer) candidate tasks in priority order."""
        w = self.world
        cols, outside = self._column_status()
        tasks = []
        # First fix any misplaced in-slot blocks (top-most first).
        for i, c in enumerate(cols):
            for k, n in enumerate(c):
                if not self._is_placed(n, i, k):
                    tasks.append((n, i, k))
                    break
        if tasks:
            return tasks
        counts = [len(c) for c in cols]
        rq = w.q
        cand = []
        for n in outside:
            bc = w.block_center(n)
            order = sorted(range(w.ncols), key=lambda i: (counts[i], abs(w.col_center(i) - bc[0])))
            for i in order[:2]:
                if counts[i] >= w.max_layers():
                    continue
                cost = math.hypot(bc[0] - rq[0], bc[1] - rq[1]) + 3.0 * counts[i] + 2.0 * self.failed.get(n, 0)
                cand.append((cost, n, i, counts[i]))
        cand.sort()
        return [(n, i, k) for _, n, i, k in cand]

    # ------------------------------------------------------------------
    def _make_plan(self):
        w = self.world
        remaining = self.t_start + 45.0 - time.time()
        held = self._held()
        if held is not None:
            name = held[0]
            cols, outside = self._column_status()
            # Determine target column/layer for the held block.
            tgt = None
            for i, c in enumerate(cols):
                if name in c:
                    tgt = (i, c.index(name))
            if tgt is None:
                counts = [len(c) for c in cols]
                bc = w.block_center(name)
                i = min(range(w.ncols), key=lambda i: (counts[i], abs(w.col_center(i) - bc[0])))
                tgt = (i, counts[i])
            plan = self._plan_transport(w.q, held, tgt[0], tgt[1], time_limit=min(3.0, max(0.3, remaining / 4)))
            if plan:
                return plan
            # Could not transport: release and back off
            return [(w.q, 0.0)]
        if w.vac > 0.5:
            return [(w.q, 0.0)]
        tasks = self._choose_task()
        if not tasks:
            return None
        budget = min(12.0, max(0.5, remaining / 3))
        t0 = time.time()
        stage = self._plan_staging(tasks)
        if stage:
            return stage
        for (n, col, layer) in tasks[:8]:
            left = budget - (time.time() - t0)
            if left <= 0:
                break
            plan = self._plan_pick(n, lambda qg, held, tl: self._plan_transport(qg, held, col, layer, time_limit=tl),
                                   time_cap=min(left, 4.0))
            if plan:
                return plan
            self.failed[n] = self.failed.get(n, 0) + 1
        # Fallback: relocate some block to free space (to unclutter).
        cols, outside = self._column_status()
        rq = w.q
        outside.sort(key=lambda n: math.hypot(w.block_center(n)[0] - rq[0], w.block_center(n)[1] - rq[1]))
        t0 = time.time()
        for n in outside[:6]:
            left = budget - (time.time() - t0)
            if left <= 0:
                break
            plan = self._plan_pick(n, self._plan_relocate, time_cap=min(left, 3.0))
            if plan:
                return plan
        return None

    STAGE_DX = 0.36
    STAGE_Y = 2.52

    def _stage_poses(self, n):
        w = self.world
        bw, bh = w.bsize[n]
        out = []
        for cx in (w.x1 - self.STAGE_DX - bw / 2, w.x1 + w.w1 + self.STAGE_DX + bw / 2):
            if cx - bw / 2 < 0.05 or cx + bw / 2 > WORLD_W - 0.05:
                continue
            out.append([(cx - bw / 2, self.STAGE_Y, 0.0), (cx + bw / 2, self.STAGE_Y + bh, np.pi)])
        return out

    def _plan_staging(self, tasks):
        """First-trip optimisation: carry an outside block to a spot beside the shelf
        while in-shelf blocks still have to be lifted."""
        if not self.ENABLE_STAGING:
            return None
        w = self.world
        cols, outside = self._column_status()
        in_slot = set(n for c in cols for n in c)
        if not tasks or tasks[0][0] not in in_slot or not outside:
            return None
        rq = w.q
        sx = w.x1 + w.w1 / 2
        d_shelf = math.hypot(rq[0] - sx, rq[1] - BASE_Y_PLACE)
        if d_shelf < 1.0:
            return None
        # Already staged something near the shelf?
        for n in outside:
            bc = w.block_center(n)
            if bc[1] > 2.3:
                return None
        cand = []
        for n in outside:
            bc = w.block_center(n)
            to_shelf = math.hypot(bc[0] - sx, bc[1] - BASE_Y_PLACE)
            detour = math.hypot(bc[0] - rq[0], bc[1] - rq[1]) + to_shelf - d_shelf
            gain = 2.0 * to_shelf - detour - 0.4 * w.w1
            cand.append((-gain, n))
        cand.sort()
        t0 = time.time()
        for neg_gain, n in cand[:2]:
            if -neg_gain < self.STAGE_MIN_GAIN:
                break
            poses = self._stage_poses(n)
            if not poses:
                return None
            poses.sort(key=lambda ps: abs(ps[0][0] - rq[0]))
            for tps in poses:
                if time.time() - t0 > 4.0:
                    return None
                plan = self._plan_pick(n, lambda qg, held, tl, tps=tps: self._plan_transport(
                    qg, held, None, None, time_limit=tl, tposes=tps), time_cap=2.0)
                if plan:
                    return plan
        return None

    def _grasp_candidates(self, n, near=None):
        w = self.world
        bx, by, bth = w.bpose[n]
        bw, bh = w.bsize[n]
        cx, cy = w.block_center(n)
        u = (math.cos(bth), math.sin(bth))
        nn = (-math.sin(bth), math.cos(bth))
        out = []
        for side in (1.0, -1.0):
            nx, ny = side * nn[0], side * nn[1]
            th = math.atan2(-ny, -nx)
            for off in (0.0, 0.03, -0.03, 0.06, -0.06, 0.09, -0.09):
                for arm in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8):
                    d = bh / 2 + GAP + GRIP_W / 2 + arm
                    x = cx + off * u[0] + d * nx
                    y = cy + off * u[1] + d * ny
                    out.append(((x, y, th, arm), abs(off), side))
            if near is not None:
                # Analytic candidates from base positions near the robot.
                for (bx_, by_) in near:
                    o = (bx_ - cx) * u[0] + (by_ - cy) * u[1]
                    d = (bx_ - cx) * nx + (by_ - cy) * ny
                    arm = d - (bh / 2 + GAP + GRIP_W / 2)
                    if abs(o) <= bw / 2 - 0.035 and ARM_MIN <= arm <= ARM_MAX:
                        out.append(((bx_, by_, th, arm), abs(o), side))
        return out

    def _plan_pick(self, n, transport_fn, time_cap=4.0):
        w = self.world
        q0 = w.q
        ck_free = Checker(w, margin=0.006)
        ck_grasp = Checker(w, margin=0.004)
        t0 = time.time()
        # Step 0: retract arm at current pose.
        pre = []
        if q0[3] > ARM_MIN + 1e-4:
            qr = (q0[0], q0[1], q0[2], ARM_MIN)
            pre = [(p, 0.0) for p in densify(q0, qr)]
            qstart = qr
        else:
            qstart = q0
        near = [(qstart[0], qstart[1])]
        for _ in range(40):
            r = self.rng.uniform(0.0, 0.3)
            a = self.rng.uniform(-np.pi, np.pi)
            near.append((qstart[0] + r * math.cos(a), qstart[1] + r * math.sin(a)))
        cands = self._grasp_candidates(n, near)
        scored = []
        for qg, off, side in cands:
            if not (0.0 < qg[0] < WORLD_W and 0.0 < qg[1] < WORLD_H):
                continue
            cost = math.hypot(qg[0] - qstart[0], qg[1] - qstart[1]) + 2.0 * off + 0.3 * (qg[3] - ARM_MIN)
            scored.append((cost, qg))
        scored.sort(key=lambda t: t[0])
        found = []
        for cost, qg in scored:
            left = time_cap - (time.time() - t0)
            if left <= 0:
                break
            if ck_grasp.collides(qg):
                continue
            if not self._suction_only(qg, n):
                continue
            qa = (qg[0], qg[1], qg[2], ARM_MIN)
            approach = densify(qa, qg)
            if not ck_grasp.path_ok(approach):
                continue
            if ck_free.collides(qa):
                continue
            rel = se2_mul(se2_inv(tip_pose(qg)), w.bpose[n])
            held = (n, rel)
            trans = transport_fn(qg, held, min(left, 1.5))
            if trans is None:
                continue
            left = time_cap - (time.time() - t0)
            if self._close(qstart, qa, 1e-3):
                nav = [qstart, qa]
            else:
                nav = birrt(ck_free, qstart, qa, self.rng, ARM_MIN, time_limit=max(0.2, min(left, 1.5)))
                if nav is None:
                    continue
            pts = [q0] + [p for p, _ in pre] + densify_path(nav) + approach
            pts = smooth(ck_grasp, pts, self.rng)
            plan = [(p, 0.0) for p in pts[1:]]
            if plan:
                plan[-1] = (plan[-1][0], 1.0)  # vacuum on during last approach step
            else:
                plan = [(qg, 1.0)]
            plan += trans
            found.append(plan)
            if len(found) >= self._k_pick():
                break
        if not found:
            return None
        return min(found, key=len)

    def _plan_relocate(self, qg, held, time_limit):
        """Carry held block to free space (low band of the room) and drop it."""
        w = self.world
        ck = Checker(w, held=held, margin=0.006)
        seg = densify(qg, (qg[0], qg[1], qg[2], ARM_MIN))
        if not ck.path_ok(seg, 0.003):
            return None
        qs = seg[-1]
        t0 = time.time()
        for _ in range(30):
            if time.time() - t0 > time_limit:
                return None
            if self.rng.random() < 0.7:
                goal = (self.rng.uniform(0.3, WORLD_W - 0.3), self.rng.uniform(0.25, 0.55),
                        self.rng.uniform(-np.pi, np.pi), ARM_MIN)
            else:
                goal = (self.rng.uniform(0.3, WORLD_W - 0.3), self.rng.uniform(0.25, 2.3),
                        self.rng.uniform(-np.pi, np.pi), ARM_MIN)
            if ck.collides(goal, 0.03):
                continue
            nav = birrt(ck, qs, goal, self.rng, ARM_MIN,
                        time_limit=max(0.1, time_limit - (time.time() - t0)), margin=0.005)
            if nav is None:
                continue
            pts = smooth(ck, [qg] + seg + densify_path(nav), self.rng, margin=0.004)
            plan = [(p, 1.0) for p in pts[1:]]
            if plan:
                plan[-1] = (plan[-1][0], 0.0)
            else:
                plan = [(goal, 0.0)]
            return plan
        return None

    def _suction_only(self, q, n):
        w = self.world
        x, y, th, arm = q
        c, s = math.cos(th), math.sin(th)
        sd = arm + GRIP_W + GRIP_W / 2
        zone = Poly(rect_from_center(x + sd * c, y + sd * s, GRIP_W, GRIP_H, th))
        hit = False
        for m in w.blocks:
            sep = poly_sep(zone, w.block_poly(m))
            if m == n:
                if sep < -1e-4:
                    hit = True
            elif sep < 0.004:
                return False
        return hit

    def _place_configs(self, held, col, layer, tposes=None):
        """Return list of (q_pre, q_place) for placing held block."""
        w = self.world
        name, rel = held
        out = []
        rel_inv = se2_inv(rel)
        if tposes is None:
            tposes = [self._target_pose(name, col, layer, flip) for flip in (False, True)]
        for tp in tposes:
            tip = se2_mul(tp, rel_inv)
            th = tip[2]
            if abs(wrap(th - np.pi / 2)) > 0.6:
                continue
            # Choose base so that it sits at BASE_Y_PLACE along the arm direction.
            s = math.sin(th)
            if s < 0.5:
                continue
            arm_hi = (tip[1] - BASE_Y_PLACE) / s - GRIP_W / 2
            top_place = max(p[1] for p in w.block_poly(name, tp).pts)
            need = top_place - (w.sy - 0.02)
            arms = [a for a in (ARM_MAX, 0.7, 0.6, 0.5, 0.4, 0.3) if a > arm_hi + 1e-6][:3] + [min(arm_hi, ARM_MAX)]
            for arm_place in arms:
                if arm_place < ARM_MIN:
                    continue
                qp = base_from_tip(tip, arm_place)
                arm_pre = max(ARM_MIN, arm_place - need / s)
                qpre = (qp[0], qp[1], qp[2], arm_pre)
                out.append(((abs(wrap(th - np.pi / 2)), -arm_place), qpre, qp))
        out.sort(key=lambda t: t[0])
        return [(a, b) for _, a, b in out]

    def _plan_transport(self, qg, held, col, layer, time_limit=4.0, tposes=None):
        """Plan from grasp config qg (vacuum on, holding) to placing & retreat."""
        w = self.world
        name = held[0]
        ck = Checker(w, held=held, margin=0.005)
        ck_tight = Checker(w, held=held, margin=0.003)
        ck_free = Checker(w, exclude=(), margin=0.003)
        configs = self._place_configs(held, col, layer, tposes)
        if not configs:
            return None
        found = []
        for qpre, qp in configs:
            if ck_tight.collides(qp) or ck_tight.collides(qpre):
                continue
            insert = densify(qpre, qp)
            if not ck_tight.path_ok(insert):
                continue
            # Post-place retreat: release, retract arm (block stays).
            # Build a checker with the block at its placed pose.
            placed_pose = se2_mul(tip_pose(qp), held[1])
            saved = w.bpose[name]
            w.bpose[name] = placed_pose
            ck_after = Checker(w, margin=0.002)
            w.bpose[name] = saved
            qback = (qp[0], qp[1], qp[2], ARM_MIN)
            retreat = densify(qp, qback)
            if not ck_after.path_ok(retreat):
                continue
            path_to_pre = None
            # Option A: local - rotate about tip to place theta, then straight.
            if qdist(qg, qpre) < 1.0 or abs(qg[3] - qpre[3]) > 0.15:
                segA = tip_rotate_path(qg, qp[2])
                if segA and ck_tight.path_ok(segA):
                    qa_end = segA[-1] if segA else qg
                    segB = densify(qa_end, qp)
                    if ck_tight.path_ok(segB):
                        path_to_pre = segA + segB
                        insert = []
            if path_to_pre is None:
                # Option B: retract (possibly after leveling), RRT, insert.
                options = [densify(qg, (qg[0], qg[1], qg[2], ARM_MIN))]
                rot = tip_rotate_path(qg, qp[2])
                if rot:
                    options.append(rot + densify(rot[-1], (rot[-1][0], rot[-1][1], rot[-1][2], ARM_MIN)))
                for seg in options:
                    if not ck.path_ok(seg, 0.003):
                        continue
                    qs = seg[-1]
                    qpre_nav = qpre
                    tail = []
                    if qpre[3] > ARM_MIN + 1e-6:
                        qpre_nav = (qpre[0], qpre[1], qpre[2], ARM_MIN)
                        tail = densify(qpre_nav, qpre)
                        if not ck_tight.path_ok(tail):
                            continue
                    nav = birrt(ck, qs, qpre_nav, self.rng, ARM_MIN, time_limit=time_limit, margin=0.005)
                    if nav is None:
                        continue
                    path_to_pre = seg + densify_path(nav) + tail
                    break
            if path_to_pre is None:
                continue
            pts = smooth(ck_tight, [qg] + path_to_pre + insert, self.rng)
            plan = [(p, 1.0) for p in pts[1:]]
            if plan:
                plan[-1] = (plan[-1][0], 0.0)  # release on final insertion step
            else:
                plan = [(qp, 0.0)]
            found.append(plan)
            if len(found) >= self._k_trans():
                break
        if not found:
            return None
        return min(found, key=len)
