"""Rovers approach: grid planning over task regions + opportunistic operator selection."""
import itertools
import math
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

RES = 0.05
CELLS_PER_STEP = 4  # 0.2 / 0.05
NI, NJ = 41, 91  # x: 0.25..2.25 (per side), y: -2.25..2.25
SEL = {"sample": -5 / 6, "calibrate": -0.5, "image": -1 / 6, "noop": 0.0,
       "send": 0.5, "drop": 5 / 6}
CAM_OFF = (-0.087, -0.0125)
OBJ_RANGE = 1.97
OBJ_RANGE_EXEC = 1.99
LANDER_RANGE = 3.95
LANDER_RANGE_EXEC = 3.975
SAMPLE_RANGE = 0.24
SAMPLE_RANGE_EXEC = 0.245
HOME_RANGE = 0.15
IMG_PEN = 1.0
PILLAR_INFL_X = 0.21
PILLAR_INFL_Y = 0.20
MOUND_INFL = 0.20
LANDER_CLEAR = 0.78
ROVER_BLOCK_R = 0.22


def _box_dist(px, py, ox, oy, hx, hy):
    dx = np.maximum(np.abs(px - ox) - hx, 0.0)
    dy = np.maximum(np.abs(py - oy) - hy, 0.0)
    return np.hypot(dx, dy)


def _seg_box_hit(x0, y0, x1, y1, bx, by, hx, hy):
    """Vectorized over segment starts (x0,y0 arrays); slab test vs one AABB."""
    dx = x1 - x0
    dy = y1 - y0
    tmin = np.zeros_like(x0, dtype=float)
    tmax = np.ones_like(x0, dtype=float)
    for d, p0, lo, hi in ((dx, x0, bx - hx, bx + hx), (dy, y0, by - hy, by + hy)):
        d = np.asarray(d, dtype=float)
        p0 = np.asarray(p0, dtype=float)
        small = np.abs(d) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - p0) / d
            t2 = (hi - p0) / d
        ta = np.where(small, -np.inf, np.minimum(t1, t2))
        tb = np.where(small, np.inf, np.maximum(t1, t2))
        outside = small & ((p0 < lo) | (p0 > hi))
        tmin = np.maximum(tmin, ta)
        tmax = np.minimum(tmax, tb)
        tmax = np.where(outside, -1.0, tmax)
    return tmin <= tmax


def _seg_circle_hit(x0, y0, x1, y1, cx, cy, r):
    dx = x1 - x0
    dy = y1 - y0
    L2 = dx * dx + dy * dy + 1e-12
    t = np.clip(((cx - x0) * dx + (cy - y0) * dy) / L2, 0.0, 1.0)
    px = x0 + t * dx
    py = y0 + t * dy
    return np.hypot(px - cx, py - cy) < r


def _pillar_free(px, py, ox, oy, hx, hy, slack):
    return (np.abs(px - ox) >= hx + PILLAR_INFL_X - slack) | \
        (np.abs(py - oy) >= hy + PILLAR_INFL_Y - slack)


class Side:
    """Grid for one rover's half of the arena."""

    def __init__(self, sign, pillars, mounds, lander, theta):
        self.sign = sign
        self.theta = theta
        ii, jj = np.meshgrid(np.arange(NI), np.arange(NJ), indexing="ij")
        self.X = sign * (0.25 + RES * ii)
        self.Y = -2.25 + RES * jj
        free = np.ones((NI, NJ), dtype=bool)
        for (ox, oy, hx, hy) in pillars:
            free &= _pillar_free(self.X, self.Y, ox, oy, hx, hy, 0.0)
        for (ox, oy, hx, hy) in mounds:
            free &= _box_dist(self.X, self.Y, ox, oy, hx, hy) >= MOUND_INFL
        free &= np.hypot(self.X - lander[0], self.Y - lander[1]) >= LANDER_CLEAR
        self.free = free
        c, s = math.cos(theta), math.sin(theta)
        self.cdx = c * CAM_OFF[0] - s * CAM_OFF[1]
        self.cdy = s * CAM_OFF[0] + c * CAM_OFF[1]
        self.CX = self.X + self.cdx
        self.CY = self.Y + self.cdy
        # graph
        idx = -np.ones((NI, NJ), dtype=np.int64)
        fi, fj = np.nonzero(free)
        idx[fi, fj] = np.arange(len(fi))
        self.idx = idx
        self.fi, self.fj = fi, fj
        self.N = len(fi)
        rows, cols = [], []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = fi + di, fj + dj
                ok = (ni >= 0) & (ni < NI) & (nj >= 0) & (nj < NJ)
                ni2, nj2 = ni[ok], nj[ok]
                src = idx[fi[ok], fj[ok]]
                dst = idx[ni2, nj2]
                good = dst >= 0
                rows.append(src[good])
                cols.append(dst[good])
        self.rows = np.concatenate(rows)
        self.cols = np.concatenate(cols)
        self.base = csr_matrix((np.ones(len(self.rows)), (self.rows, self.cols)),
                               shape=(self.N, self.N))
        self._sp_cache = {}

    def cell_of(self, x, y):
        i = int(round((self.sign * x - 0.25) / RES))
        j = int(round((y + 2.25) / RES))
        return min(max(i, 0), NI - 1), min(max(j, 0), NJ - 1)

    def pos(self, n):
        return float(self.X[self.fi[n], self.fj[n]]), float(self.Y[self.fi[n], self.fj[n]])

    def node_of(self, x, y):
        i, j = self.cell_of(x, y)
        n = self.idx[i, j]
        if n >= 0:
            return int(n)
        # nearest free node
        d = (self.X[self.fi, self.fj] - x) ** 2 + (self.Y[self.fi, self.fj] - y) ** 2
        return int(np.argmin(d))

    def field_from(self, init):
        """init: array (N,) of initial costs (inf = not a source). Returns geodesic field."""
        src = np.nonzero(np.isfinite(init))[0]
        if len(src) == 0:
            return np.full(self.N, np.inf)
        N = self.N
        rows = np.concatenate([self.rows, np.full(len(src), N)])
        cols = np.concatenate([self.cols, src])
        data = np.concatenate([np.ones(len(self.rows)), init[src] + 1.0])
        g = csr_matrix((data, (rows, cols)), shape=(N + 1, N + 1))
        d = dijkstra(g, directed=True, indices=N)
        return d[:N] - 1.0

    def dist_from_node(self, n):
        if n not in self._sp_cache:
            self._sp_cache[n] = dijkstra(self.base, directed=True, indices=n)
            if len(self._sp_cache) > 64:
                self._sp_cache.pop(next(iter(self._sp_cache)))
        return self._sp_cache[n]


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.T = {}
        for name in ["rover", "lander", "objective", "sample", "obstacle"]:
            try:
                self.T[name] = observation_space.get_type(name)
            except Exception:
                self.T[name] = name

    # ------------------------------------------------------------------ parse
    def _objs(self, state, t):
        return sorted(state.get_objects(self.T[t]), key=lambda o: o.name)

    def _parse_static(self, state):
        g = state.get
        rovers = self._objs(state, "rover")
        self.rover_objs = {o.name: o for o in rovers}
        self.rnames = sorted(self.rover_objs)
        lo = self._objs(state, "lander")[0]
        self.lander = (g(lo, "x"), g(lo, "y"))
        self.pillars, self.mounds = [], []
        for o in self._objs(state, "obstacle"):
            b = (g(o, "x"), g(o, "y"), g(o, "half_x"), g(o, "half_y"))
            if g(o, "half_z") > 0.1:
                self.pillars.append(b)
            else:
                self.mounds.append(b)
        self.objectives = self._objs(state, "objective")
        self.obj_xy = [(g(o, "x"), g(o, "y")) for o in self.objectives]
        self.samples = self._objs(state, "sample")
        self.samp_xy = [(g(o, "x"), g(o, "y")) for o in self.samples]
        self.samp_soil = [g(o, "is_soil") > 0.5 for o in self.samples]
        self.home = {}
        for rn in self.rnames:
            o = self.rover_objs[rn]
            self.home[rn] = (g(o, "x"), g(o, "y"), g(o, "theta"))

    # -------------------------------------------------------------- visibility
    def _los_clear(self, cx, cy, tx, ty, margin, blockers=()):
        cx = np.asarray(cx, dtype=float)
        cy = np.asarray(cy, dtype=float)
        ok = np.ones(cx.shape, dtype=bool)
        for (ox, oy, hx, hy) in self.pillars:
            ok &= ~_seg_box_hit(cx, cy, tx, ty, ox, oy, hx + margin, hy + margin)
        for (bx, by, br) in blockers:
            ok &= ~_seg_circle_hit(cx, cy, tx, ty, bx, by, br)
        return ok

    # ---------------------------------------------------------------- planning
    def reset(self, state, info):
        t0 = time.time()
        self._parse_static(state)
        self.sides = {}
        for rn in self.rnames:
            hx, hy, hth = self.home[rn]
            sign = 1.0 if hx > 0 else -1.0
            self.sides[rn] = Side(sign, self.pillars, self.mounds, self.lander, hth)
        self.other = {self.rnames[0]: self.rnames[1], self.rnames[1]: self.rnames[0]}
        # task regions per rover
        self.regions = {}
        self.ovis, self.odist = {}, {}
        for rn in self.rnames:
            S = self.sides[rn]
            cx = S.CX[S.fi, S.fj]
            cy = S.CY[S.fi, S.fj]
            px = S.X[S.fi, S.fj]
            py = S.Y[S.fi, S.fj]
            reg = {}
            self.ovis.setdefault(rn, {})
            self.odist.setdefault(rn, {})
            for k, (ox, oy) in enumerate(self.obj_xy):
                dd = np.hypot(cx - ox, cy - oy)
                m = dd <= OBJ_RANGE
                m &= self._los_clear(cx, cy, ox, oy, 0.03)
                reg[("O", k)] = m
                # loose visibility (could steal the image)
                ml = (dd <= OBJ_RANGE_EXEC + 0.03) & self._los_clear(cx, cy, ox, oy, -0.03)
                self.ovis[rn][k] = ml
                self.odist[rn][k] = dd
            for k, (sx, sy) in enumerate(self.samp_xy):
                dk = np.hypot(px - sx, py - sy)
                m = dk <= SAMPLE_RANGE
                for j, (qx, qy) in enumerate(self.samp_xy):
                    if j != k:
                        m &= np.hypot(px - qx, py - qy) > dk + 0.02
                if m.any():
                    reg[("S" if not self.samp_soil[k] else "T", k)] = m
            oh = self.home[self.other[rn]]
            lx, ly = self.lander
            m = np.hypot(cx - lx, cy - ly) <= LANDER_RANGE
            m &= self._los_clear(cx, cy, lx, ly, 0.03, [(oh[0], oh[1], ROVER_BLOCK_R + 0.05)])
            reg["send"] = m
            hxy = self.home[rn]
            dl = math.hypot(hxy[0] - self.lander[0], hxy[1] - self.lander[1])
            dlo = math.hypot(oh[0] - self.lander[0], oh[1] - self.lander[1])
            hr = 0.01 if dl < dlo else HOME_RANGE
            reg["home"] = np.hypot(px - hxy[0], py - hxy[1]) <= hr
            self.regions[rn] = reg
        self.plan = self._make_plan(t0)
        self.blacklist = {rn: set() for rn in self.rnames}
        self.stuck = {rn: 0 for rn in self.rnames}
        self.targets = {rn: None for rn in self.rnames}
        self.last_pos = {}
        self._bw_cache = {}
        self.forced = {}
        self._pred = {}
        self.bad_points = []
        self.cmd = {}
        self.step_count = 0

    def _img_mask(self, rn, k, imaged):
        m = self.regions[rn][("O", k)].copy()
        dk = self.odist[rn][k]
        for j in range(len(self.obj_xy)):
            if j == k or j in imaged:
                continue
            m &= ~(self.ovis[rn][j] & (self.odist[rn][j] <= dk + 0.03))
        return m

    def _rover_costs(self, rn, deadline):
        """Best cost & order for every feasible task set of rover rn."""
        S = self.sides[rn]
        reg = self.regions[rn]
        start = S.node_of(self.home[rn][0], self.home[rn][1])
        init = np.full(S.N, np.inf)
        init[start] = 0.0
        F0 = S.field_from(init)
        dhome = S.field_from(np.where(reg["home"], 0.0, np.inf))
        send = reg["send"]
        send_home = np.where(send, dhome, np.inf)
        objs = [t for t in reg if t[0] == "O" and reg[t].any()]
        stones = [t for t in reg if t[0] == "S"]
        soils = [t for t in reg if t[0] == "T"]
        best = {frozenset(): (0.0, ())}
        # DFS over sequences
        stack = [((), F0)]
        while stack:
            seq, F = stack.pop()
            if seq:
                c = np.min(F + send_home)
                n_img = sum(1 for t in seq if t[0] == "O")
                cost = c / CELLS_PER_STEP + IMG_PEN * n_img + 1.0
                key = frozenset(seq)
                if key not in best or cost < best[key][0]:
                    best[key] = (cost, seq)
            if time.time() > deadline:
                continue
            has_s = any(t[0] == "S" for t in seq)
            has_t = any(t[0] == "T" for t in seq)
            for t in objs + ([] if has_s else stones) + ([] if has_t else soils):
                if t in seq:
                    continue
                if t[0] == "O":
                    m = self._img_mask(rn, t[1], set(u[1] for u in seq if u[0] == "O"))
                else:
                    m = reg[t]
                init = np.where(m, F, np.inf)
                if not np.isfinite(init).any():
                    continue
                F2 = S.field_from(init)
                stack.append((seq + (t,), F2))
        return best

    def _make_plan(self, t0):
        r0, r1 = self.rnames
        b0 = self._rover_costs(r0, t0 + 12.0)
        b1 = self._rover_costs(r1, t0 + 24.0)
        nobj = len(self.objectives)
        stones = [(k) for k in range(len(self.samples)) if not self.samp_soil[k]]
        soils = [(k) for k in range(len(self.samples)) if self.samp_soil[k]]
        best = None
        for assign in itertools.product((0, 1), repeat=nobj):
            for s in stones:
                for t in soils:
                    sets = [set(), set()]
                    for k, a in enumerate(assign):
                        sets[a].add(("O", k))
                    for kk, typ in ((s, "S"), (t, "T")):
                        if (typ, kk) in self.regions[r0]:
                            sets[0].add((typ, kk))
                        elif (typ, kk) in self.regions[r1]:
                            sets[1].add((typ, kk))
                        else:
                            sets[0].add(None)
                    if None in sets[0]:
                        continue
                    k0, k1 = frozenset(sets[0]), frozenset(sets[1])
                    if k0 not in b0 or k1 not in b1:
                        continue
                    c0, c1 = b0[k0][0], b1[k1][0]
                    score = (max(c0, c1), c0 + c1)
                    if best is None or score < best[0]:
                        best = (score, b0[k0][1], b1[k1][1])
        if best is None:
            # fallback: everything greedy
            return {r0: (), r1: ()}
        self.plan_score = best[0]
        return {r0: list(best[1]), r1: list(best[2])}

    # ---------------------------------------------------------------- execution
    def _task_done(self, state, rn, t):
        g = state.get
        ri = self.rnames.index(rn)
        if t[0] == "O":
            o = self.objectives[t[1]]
            return g(o, "received_image") > 0.5 or g(o, "have_image_%s" % rn) > 0.5
        if t[0] in ("S", "T"):
            soil = t[0] == "T"
            for k, o in enumerate(self.samples):
                if self.samp_soil[k] == soil and (g(o, "analyzed_%s" % rn) > 0.5):
                    return True
            return False
        return False

    def _unsent(self, state, rn):
        g = state.get
        for o in self.objectives:
            if g(o, "have_image_%s" % rn) > 0.5 and g(o, "received_image") < 0.5:
                return True
        for k, o in enumerate(self.samples):
            if g(o, "analyzed_%s" % rn) > 0.5 and g(o, "received_analysis") < 0.5 \
                    and not self._type_received(state, self.samp_soil[k]):
                return True
        return False

    def _type_received(self, state, soil):
        g = state.get
        return any(g(o, "received_analysis") > 0.5 for k, o in enumerate(self.samples)
                   if self.samp_soil[k] == soil)

    def _current_task(self, state, rn):
        for t in self.plan[rn]:
            if t[0] == "O" and state.get(self.objectives[t[1]], "received_image") > 0.5:
                continue
            if t[0] in ("S", "T") and self._type_received(state, t[0] == "T"):
                continue
            if not self._task_done(state, rn, t):
                return t
        if self._unsent(state, rn):
            return "send"
        return "home"

    def _pick_target(self, rn, task, cur_node, state=None):
        S = self.sides[rn]
        m = self.regions[rn][task].copy() if task in self.regions[rn] else None
        if m is not None and task[0] == "O" and state is not None:
            imaged = set(k for k, o in enumerate(self.objectives)
                         if state.get(o, "have_image_%s" % rn) > 0.5)
            m2 = self._img_mask(rn, task[1], imaged)
            if m2.any():
                m = m2
        if m is None:
            return None
        for n in self.blacklist[rn]:
            if task != "home":
                m[n] = False
        if not m.any():
            m = self.regions[rn][task]
        d = S.dist_from_node(cur_node)
        if task == "home":
            hx, hy, _ = self.home[rn]
            hn = S.node_of(hx, hy)
            return hn
        # choose cell minimizing dist from here + (estimated) dist to next goal (home)
        dh = S.dist_from_node(S.node_of(self.home[rn][0], self.home[rn][1]))
        score = np.where(m, d + 0.3 * dh, np.inf)
        n = int(np.argmin(score))
        if not np.isfinite(score[n]):
            return None
        return n

    def _next_pos(self, rn, x, y, target):
        """Return next commanded position (x,y) toward target node."""
        S = self.sides[rn]
        cur = S.node_of(x, y)
        tx, ty = S.pos(target)
        if abs(tx - x) <= 0.2 + 1e-6 and abs(ty - y) <= 0.2 + 1e-6 and cur == target:
            return tx, ty
        d = S.dist_from_node(target)
        # follow gradient path from cur up to 4 cells Chebyshev, choose furthest reachable
        path = [cur]
        n = cur
        for _ in range(CELLS_PER_STEP * 3):
            if n == target:
                break
            i, j = S.fi[n], S.fj[n]
            bestn, bestd = None, d[n]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < NI and 0 <= nj < NJ:
                        m = S.idx[ni, nj]
                        if m >= 0 and d[m] < bestd - 1e-9:
                            # prefer straight-ish toward target
                            bestd, bestn = d[m], m
            if bestn is None:
                break
            n = int(bestn)
            path.append(n)
        best = path[0]
        if len(path) > 1:
            p1 = S.pos(path[1])
            if not any(abs(bx - p1[0]) < 0.03 and abs(by - p1[1]) < 0.03
                       for (bx, by) in self.bad_points):
                best = path[1]
        for n in path[2:]:
            px, py = S.pos(n)
            if abs(px - x) <= 0.2 + 1e-6 and abs(py - y) <= 0.2 + 1e-6:
                if self._seg_free(S, x, y, px, py):
                    best = n
            else:
                break
        return S.pos(best)

    def _seg_free(self, S, x0, y0, x1, y1):
        if not self._point_free(x1, y1, S.sign):
            return False
        for t in (0.25, 0.5, 0.75):
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if not self._point_free(x, y, S.sign, 0.05):
                return False
        return True

    def _point_free(self, x, y, sign, slack=0.01):
        if sign * x < 0.235 or abs(x) > 2.26 or abs(y) > 2.26:
            return False
        for (ox, oy, hx, hy) in self.pillars:
            if not _pillar_free(x, y, ox, oy, hx, hy, slack):
                return False
        for (bx, by) in self.bad_points:
            if abs(bx - x) < 0.03 and abs(by - y) < 0.03:
                return False
        for (ox, oy, hx, hy) in self.mounds:
            if _box_dist(x, y, ox, oy, hx, hy) < MOUND_INFL - 0.01:
                return False
        if math.hypot(x - self.lander[0], y - self.lander[1]) < LANDER_CLEAR - 0.02:
            return False
        return True

    def _cam(self, rn, x, y):
        S = self.sides[rn]
        return x + S.cdx, y + S.cdy

    def _visible_objs(self, rn, x, y, other_xy):
        cx, cy = self._cam(rn, x, y)
        res = []
        for k, (ox, oy) in enumerate(self.obj_xy):
            d = math.hypot(cx - ox, cy - oy)
            if d <= OBJ_RANGE_EXEC:
                if self._los_clear(np.array([cx]), np.array([cy]), ox, oy, 0.0,
                                   [(other_xy[0], other_xy[1], ROVER_BLOCK_R)])[0]:
                    res.append((d, k))
        res.sort()
        return res

    def _lander_visible(self, rn, x, y, other_xy):
        cx, cy = self._cam(rn, x, y)
        lx, ly = self.lander
        if math.hypot(cx - lx, cy - ly) > LANDER_RANGE_EXEC:
            return False
        return bool(self._los_clear(np.array([cx]), np.array([cy]), lx, ly, 0.03,
                                    [(other_xy[0], other_xy[1], ROVER_BLOCK_R)])[0])

    def _choose_op(self, state, rn, task, nx, ny, other_xy):
        g = state.get
        ro = self.rover_objs[rn]
        store_full = g(ro, "store_full") > 0.5
        calibrated = g(ro, "calibrated") > 0.5
        # needed sample types
        need_types = set()
        for t in self.plan[rn]:
            if t[0] in ("S", "T") and not self._task_done(state, rn, t) \
                    and not self._type_received(state, t[0] == "T"):
                need_types.add(t[0] == "T")
        if not store_full and need_types:
            best = None
            for k, (sx, sy) in enumerate(self.samp_xy):
                d = math.hypot(nx - sx, ny - sy)
                if d <= SAMPLE_RANGE_EXEC and (best is None or d < best[0]):
                    best = (d, k)
            if best is not None and self.samp_soil[best[1]] in need_types:
                return "sample"
        my_imgs = set()
        for t in self.plan[rn]:
            if t[0] == "O":
                o = self.objectives[t[1]]
                if g(o, "have_image_%s" % rn) < 0.5 and g(o, "received_image") < 0.5:
                    my_imgs.add(t[1])
        vis = self._visible_objs(rn, nx, ny, other_xy) if (my_imgs or calibrated) else []
        if calibrated and my_imgs:
            needed = [(d, k) for (d, k) in vis
                      if g(self.objectives[k], "have_image_%s" % rn) < 0.5]
            if needed and needed[0][1] in my_imgs:
                return "image"
            if needed and any(k in my_imgs for (_, k) in needed) and self.stuck[rn] >= 2:
                return "image"
        if store_full and need_types:
            return "drop"
        if not calibrated and my_imgs and vis:
            return "calibrate"
        if self._unsent(state, rn) and self._lander_visible(rn, nx, ny, other_xy):
            return "send"
        if store_full:
            return "drop"
        return "noop"

    def _remaining(self, state, rn):
        rem = []
        for t in self.plan[rn]:
            if t[0] == "O" and state.get(self.objectives[t[1]], "received_image") > 0.5:
                continue
            if t[0] in ("S", "T") and self._type_received(state, t[0] == "T"):
                continue
            if not self._task_done(state, rn, t):
                rem.append(t)
        if rem or self._unsent(state, rn):
            rem.append("send")
        rem.append("home")
        return tuple(rem)

    def _mask(self, rn, task, imaged):
        if task[0] == "O":
            m2 = self._img_mask(rn, task[1], imaged)
            if m2.any():
                return m2
        return self.regions[rn][task]

    def _backward(self, rn, rem, imaged):
        """G[k] = cost-to-go field after completing rem[k] standing at a cell."""
        key = (rn, rem, imaged)
        if key in self._bw_cache:
            return self._bw_cache[key]
        S = self.sides[rn]
        ims = []
        cur = set(imaged)
        for t in rem:
            ims.append(frozenset(cur))
            if t != "send" and t != "home" and t[0] == "O":
                cur.add(t[1])
        G = [None] * len(rem)
        G[-1] = np.zeros(S.N)
        for k in range(len(rem) - 2, -1, -1):
            nxt = rem[k + 1]
            m = self._mask(rn, nxt, ims[k + 1]) if nxt not in ("send", "home") \
                else self.regions[rn][nxt]
            init = np.where(m, G[k + 1], np.inf)
            G[k] = S.field_from(init)
        if len(self._bw_cache) > 200:
            self._bw_cache.clear()
        self._bw_cache[key] = (G, ims)
        return G, ims

    def _target(self, state, rn, rem, cur, other_xy):
        S = self.sides[rn]
        imaged = frozenset(k for k, o in enumerate(self.objectives)
                           if state.get(o, "have_image_%s" % rn) > 0.5)
        G, ims = self._backward(rn, rem, imaged)
        task = rem[0]
        if task == "home":
            m = self.regions[rn]["home"]
        elif task == "send":
            m = self.regions[rn]["send"].copy()
            # exclude cells whose lander LOS is blocked by the other rover now
            cx = S.CX[S.fi, S.fj]
            cy = S.CY[S.fi, S.fj]
            lx, ly = self.lander
            traj = self._pred.get(self.other[rn]) or [other_xy]
            dcur = S.dist_from_node(cur)
            karr = np.maximum(np.ceil(dcur / CELLS_PER_STEP), 1)
            blk = np.zeros(S.N, dtype=bool)
            for kk in range(len(traj)):
                if kk == len(traj) - 1:
                    sel = karr >= kk
                elif kk == 0:
                    sel = karr <= 0
                else:
                    sel = karr == kk
                if not sel.any():
                    continue
                px_, py_ = traj[kk]
                blk[sel] = _seg_circle_hit(cx[sel], cy[sel], lx, ly, px_, py_,
                                           ROVER_BLOCK_R + 0.03)
            m2 = m & ~blk
            if m2.any():
                m = m2
        else:
            m = self._mask(rn, task, ims[0]).copy()
        bl = [n for n in self.blacklist[rn] if n[0] == task]
        if bl:
            m = m.copy()
            for (_, n) in bl:
                m[n] = False
            if not m.any():
                m = self.regions[rn][task] if task in self.regions[rn] else m
        d = S.dist_from_node(cur)
        score = np.where(m, d + G[0], np.inf)
        n = int(np.argmin(score))
        if not np.isfinite(score[n]):
            return None
        return n

    def _other_at(self, rn, k):
        tr = self._pred.get(self.other[rn])
        if not tr:
            o = self.rover_objs[self.other[rn]]
            return None
        return tr[min(k, len(tr) - 1)]

    def _img_ok(self, state, rn, k, px, py, other_xy):
        vis = self._visible_objs(rn, px, py, other_xy)
        needed = [(d, j) for (d, j) in vis
                  if state.get(self.objectives[j], "have_image_%s" % rn) < 0.5]
        return bool(needed) and needed[0][1] == k

    def _lookahead(self, state, rn, rem, cur, x, y, other_now):
        """Try to perform rem[0]'s op while already moving toward rem[1:]."""
        task = rem[0]
        tgt2 = self._target(state, rn, rem[1:], cur, other_now)
        if tgt2 is None:
            return None
        p1 = self._next_pos(rn, x, y, tgt2)
        if abs(p1[0] - x) < 1e-6 and abs(p1[1] - y) < 1e-6:
            return None
        o1 = self._other_at(rn, 1) or other_now
        if task == "send":
            if self._unsent(state, rn) and self._lander_visible(rn, p1[0], p1[1], o1):
                ro = self.rover_objs[rn]
                if state.get(ro, "store_full") > 0.5 and not self._unsent(state, rn):
                    return None
                return p1, "send"
            return None
        k = task[1]
        calibrated = state.get(self.rover_objs[rn], "calibrated") > 0.5
        if calibrated:
            if self._img_ok(state, rn, k, p1[0], p1[1], o1):
                return p1, "image"
            return None
        # need calibrate now (at p1) and image at p2
        if not self._visible_objs(rn, p1[0], p1[1], o1):
            return None
        S = self.sides[rn]
        c1 = S.node_of(p1[0], p1[1])
        p2 = self._next_pos(rn, p1[0], p1[1], tgt2)
        o2 = self._other_at(rn, 2) or other_now
        if self._img_ok(state, rn, k, p2[0], p2[1], o2):
            return p1, "calibrate"
        return None

    def get_action(self, state):
        g = state.get
        self.step_count += 1
        act = np.zeros(8, dtype=np.float32)
        poses = {rn: (g(self.rover_objs[rn], "x"), g(self.rover_objs[rn], "y"),
                      g(self.rover_objs[rn], "theta")) for rn in self.rnames}
        nexts = {}
        tasks = {}
        self._pred = {}
        for rn in self.rnames:
            x, y, _ = poses[rn]
            tr = [(x, y)]
            t = self.targets.get(rn)
            if t is not None and t[1] is not None:
                px, py = x, y
                for _ in range(25):
                    q = self._next_pos(rn, px, py, t[1])
                    if abs(q[0] - px) < 1e-6 and abs(q[1] - py) < 1e-6:
                        break
                    px, py = q
                    tr.append(q)
            self._pred[rn] = tr
        for rn in self.rnames:
            x, y, th = poses[rn]
            S = self.sides[rn]
            rem = self._remaining(state, rn)
            task = rem[0]
            tasks[rn] = task
            cur = S.node_of(x, y)
            ox, oy, _ = poses[self.other[rn]]
            c = self.cmd.get(rn)
            if c is not None and abs(c[0] - x) < 1e-4 and abs(c[1] - y) < 1e-4 and \
                    (abs(c[2] - x) > 1e-3 or abs(c[3] - y) > 1e-3):
                self.bad_points.append((c[2], c[3]))
            # stuck detection: at target (or not moving) with task unchanged
            lp = self.last_pos.get(rn)
            prev = self.targets[rn]
            if prev is not None and prev[0] == task and task != "home" and (
                    cur == prev[1] or (lp is not None and abs(lp[0] - x) < 1e-4
                                       and abs(lp[1] - y) < 1e-4)):
                self.stuck[rn] += 1
                if self.stuck[rn] > 2:
                    self.blacklist[rn].add((task, prev[1]))
                    self.stuck[rn] = 0
            else:
                self.stuck[rn] = 0
            tgt = self._target(state, rn, rem, cur, (ox, oy))
            self.targets[rn] = (task, tgt) if tgt is not None else None
            if tgt is None:
                nexts[rn] = (x, y)
            else:
                nexts[rn] = self._next_pos(rn, x, y, tgt)
            self.forced[rn] = None
            if len(rem) > 1 and (task == "send" or task[0] == "O"):
                alt = self._lookahead(state, rn, rem, cur, x, y, (ox, oy))
                if alt is not None:
                    nexts[rn], self.forced[rn] = alt
            self.last_pos[rn] = (x, y)
        for ri, rn in enumerate(self.rnames):
            x, y, th = poses[rn]
            nx, ny = nexts[rn]
            other = self.other[rn]
            op = self.forced.get(rn) or self._choose_op(state, rn, tasks[rn], nx, ny, nexts[other])
            hth = self.home[rn][2]
            dth = math.atan2(math.sin(hth - th), math.cos(hth - th))
            act[4 * ri + 0] = np.clip(nx - x, -0.2, 0.2)
            act[4 * ri + 1] = np.clip(ny - y, -0.2, 0.2)
            act[4 * ri + 2] = np.clip(dth, -0.4, 0.4)
            act[4 * ri + 3] = SEL[op]
            self.cmd[rn] = (x, y, x + float(act[4 * ri]), y + float(act[4 * ri + 1]))
        return act
