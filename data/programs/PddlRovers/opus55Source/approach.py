"""Rovers approach: grid-based subset DP planner over regions.

Key facts used:
  * Motion is holonomic: per-step deltas up to 0.2 in x and y (independent),
    heading up to 0.4. Collision checked only at the final pose.
  * Rovers live in separate halves (divider at x=0).
  * Visibility ignores occlusion when the target is behind the rover (camera
    ray hits the rover's own body), so facing away from a target makes it
    visible iff it is within range.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.ndimage import minimum_filter
import refine3d as R3

RES = 0.025
K = 8  # cells per max step (0.2 / RES)
WIN = 2 * K + 1
R3.KX = K
INF = 10**6
R_ROVER = 0.179  # base cylinder (low obstacles: walls, mounds)
R_TALL = 0.234  # incl. kinect bar (tall obstacles: pillars, lander)
MARGIN = 0.012
CAM_OFF = 0.087
CAM_Z = 0.2992
R_OBJ_BASE = 2.04  # base-to-objective horizontal range when facing away
R_COM_BASE = 4.03
SAMPLE_R = 0.248
HOME_R = 0.248
CONE_AWAY = 1.3
NB = 80
BIN_W = 2 * math.pi / NB
BIN_C = -math.pi + BIN_W * (np.arange(NB) + 0.5)
REL_MIN = 1.8
REL_MIN_OPT = 1.55
RNG_MARGIN = 0.004
USE_RAY = True
STEP_BINS = 5  # floor(0.4 / BIN_W)


def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    vv = vx * vx + vy * vy + 1e-12
    sp = np.clip(((px - ax) * vx + (py - ay) * vy) / vv, 0.0, 1.0)
    return np.hypot(px - ax - sp * vx, py - ay - sp * vy)


def vis_cone(X, Y, tx, ty, tz, rng, occl, bc=None, opt=False):
    """Visibility per heading bin, shape X.shape + (NB,).

    Visible iff camera within range and either the target is behind the
    camera (ray hits own body first) or the ray is clear of occluders.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if bc is None:
        bc = BIN_C
    nb = len(bc)
    out = np.zeros(X.shape + (nb,), dtype=bool)
    Xf, Yf = X.reshape(-1), Y.reshape(-1)
    dz = tz - CAM_Z
    cand = np.hypot(Xf - tx, Yf - ty) <= rng + 0.1
    if not cand.any():
        return out
    ci = np.nonzero(cand)[0]
    Xc, Yc = Xf[ci], Yf[ci]
    c, s_ = np.cos(bc), np.sin(bc)
    cx = Xc[:, None] - CAM_OFF * c + 0.0125 * s_
    cy = Yc[:, None] - CAM_OFF * s_ - 0.0125 * c
    vx, vy = tx - cx, ty - cy
    d3 = np.sqrt(vx ** 2 + vy ** 2 + dz ** 2)
    if opt:
        res = d3 <= rng + 0.01
        rmin = REL_MIN_OPT
    else:
        res = d3 <= rng - RNG_MARGIN
        rmin = REL_MIN
    # rows that may need occlusion tests
    amb_rows = np.zeros(len(ci), dtype=bool)
    occ_rows = []
    for (ox, oy, rr) in occl:
        dd = _seg_dist(ox, oy, Xc, Yc, tx, ty)
        m = dd <= rr + 0.1
        if m.any():
            occ_rows.append((ox, oy, rr, m))
            amb_rows |= m
    cross_rows = (Xc * tx) < 0
    amb_rows |= cross_rows
    if amb_rows.any():
        ai = np.nonzero(amb_rows)[0]
        cxa, cya, vxa, vya = cx[ai], cy[ai], vx[ai], vy[ai]
        rel = np.arctan2(vya, vxa) - bc
        rel = np.abs((rel + np.pi) % (2 * np.pi) - np.pi)
        behind = rel >= rmin
        clear = np.ones(cxa.shape, dtype=bool)
        vv = vxa ** 2 + vya ** 2 + 1e-12
        for (ox, oy, rr, m) in occ_rows:
            sub = np.nonzero(m[ai])[0]
            if len(sub) == 0:
                continue
            dx, dy = ox - cxa[sub], oy - cya[sub]
            sp = np.clip((dx * vxa[sub] + dy * vya[sub]) / vv[sub], 0.0, 1.0)
            dist = np.hypot(dx - sp * vxa[sub], dy - sp * vya[sub])
            clear[sub] &= dist > rr
        sub = np.nonzero(cross_rows[ai])[0]
        if len(sub):
            xs = -0.05 if tx < 0 else 0.05
            vxs = vxa[sub]
            with np.errstate(divide="ignore", invalid="ignore"):
                sx = np.clip((xs - cxa[sub]) / np.where(vxs == 0, 1e-9, vxs), 0.0, 1.0)
            zc = CAM_Z + sx * dz
            clear[sub] &= zc > (0.09 if opt else 0.11)
        res[ai] &= behind | clear
    out.reshape(-1, nb)[ci] = res
    return out


def dilate(m, n):
    if n <= 0:
        return m.copy()
    if n >= NB // 2:
        return np.full(NB, bool(m.any()))
    out = m.copy()
    for j in range(1, n + 1):
        out |= np.roll(m, j) | np.roll(m, -j)
    return out
CONE_HOME = 0.34
HOME_TOL3 = 0.39
MAX_CANDS = 12
CAND_TIME = 15.0

import os
DEBUG = bool(os.environ.get('RDEBUG'))
SAMPLE, CALIBRATE, IMAGE, NOOP, SEND, DROP = range(6)


def op_value(op: int) -> float:
    return (op + 0.5) / 3.0 - 1.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def minf(arr: np.ndarray) -> np.ndarray:
    return minimum_filter(arr, size=WIN, mode="constant", cval=INF)


class ENode:
    """A propagated layer: min time to have finished some op prefix at cell."""

    def __init__(self, arr, sources, root=False):
        self.arr = arr
        self.sources = sources  # list of (D array, opinfo, prev ENode)
        self.root = root


class RoverGrid:
    def __init__(self, idx, home, state_info):
        self.idx = idx
        self.home = home
        hx, hy = home[0], home[1]
        if hx > 0:
            xlo, xhi = 0.05 + R_ROVER + MARGIN, 2.45 - R_ROVER - MARGIN
        else:
            xlo, xhi = -2.45 + R_ROVER + MARGIN, -0.05 - R_ROVER - MARGIN
        ylo, yhi = -2.45 + R_ROVER + MARGIN, 2.45 - R_ROVER - MARGIN
        i0 = math.ceil((xlo - hx) / RES - 1e-9)
        i1 = math.floor((xhi - hx) / RES + 1e-9)
        j0 = math.ceil((ylo - hy) / RES - 1e-9)
        j1 = math.floor((yhi - hy) / RES + 1e-9)
        self.xs = hx + RES * np.arange(i0, i1 + 1)
        self.ys = hy + RES * np.arange(j0, j1 + 1)
        self.i0, self.j0 = i0, j0
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.shape = self.X.shape
        free = np.ones(self.shape, dtype=bool)
        X, Y = self.X, self.Y
        for (cx, cy, hxx, hyy, rr) in state_info["boxes"]:
            dx = np.maximum(np.abs(X - cx) - hxx, 0.0)
            dy = np.maximum(np.abs(Y - cy) - hyy, 0.0)
            free &= np.hypot(dx, dy) > rr + MARGIN
        for (cx, cy, rad) in state_info["circles"]:
            free &= np.hypot(X - cx, Y - cy) > R_TALL + rad + MARGIN
        self.free = free

    def cell_of(self, x, y):
        i = int(round((x - self.home[0]) / RES)) - self.i0
        j = int(round((y - self.home[1]) / RES)) - self.j0
        i = min(max(i, 0), self.shape[0] - 1)
        j = min(max(j, 0), self.shape[1] - 1)
        return (i, j)

    def pos_of(self, c):
        return float(self.xs[c[0]]), float(self.ys[c[1]])

    def dist(self, x, y):
        return np.hypot(self.X - x, self.Y - y)

    def prop(self, D):
        E = D.copy()
        free = self.free
        while True:
            N = np.where(free, np.minimum(E, minf(E) + 1), INF)
            if np.array_equal(N, E):
                return E
            E = N

    def stage(self, E, mask):
        return np.where(mask & self.free, minf(E) + 1, INF)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------------------------
    def _parse(self, state):
        objs = {o.name: o for o in state}
        g = lambda o, f: float(state.get(o, f))
        rovers = []
        for r in range(2):
            o = objs[f"rover{r}"]
            rovers.append({f: g(o, f) for f in
                           ["x", "y", "theta", "store_full", "calibrated", "at_home"]})
        lander = objs["lander"]
        lander_xy = (g(lander, "x"), g(lander, "y"))
        objectives = []
        i = 0
        while f"objective{i}" in objs:
            o = objs[f"objective{i}"]
            objectives.append({f: g(o, f) for f in
                               ["x", "y", "z", "have_image_rover0",
                                "have_image_rover1", "received_image"]})
            i += 1
        samples = []
        i = 0
        while f"sample{i}" in objs:
            o = objs[f"sample{i}"]
            samples.append({f: g(o, f) for f in
                            ["x", "y", "z", "is_soil", "analyzed_rover0",
                             "analyzed_rover1", "received_analysis"]})
            i += 1
        obstacles = []
        i = 0
        while f"obstacle{i}" in objs:
            o = objs[f"obstacle{i}"]
            obstacles.append({f: g(o, f) for f in
                              ["x", "y", "z", "half_x", "half_y", "half_z"]})
            i += 1
        return rovers, lander_xy, objectives, samples, obstacles

    def reset(self, state, info):
        rovers, lander_xy, objectives, samples, obstacles = self._parse(state)
        self.homes = []
        defaults = [(1.0, -1.75, math.pi), (-1.0, -1.75, 0.0)]
        for r in range(2):
            rv = rovers[r]
            if rv["at_home"] > 0.5:
                self.homes.append((rv["x"], rv["y"], rv["theta"]))
            else:
                self.homes.append(defaults[r])
        self.lander_xy = lander_xy
        self.lander_z = float(state.get(state.get_object_from_name("lander"), "z"))
        boxes = []
        circles = []
        for ob in obstacles:
            if ob["half_x"] > 0.15:  # mound
                boxes.append((ob["x"], ob["y"], ob["half_x"], ob["half_y"], R_ROVER))
            else:  # pillar, unknown yaw -> bounding circle
                circles.append((ob["x"], ob["y"],
                                math.hypot(ob["half_x"], ob["half_y"])))
        # lander AABB (Husky)
        boxes.append((lander_xy[0] + 0.0, lander_xy[1] + 0.0, 0.505, 0.346, R_TALL))
        info_ = {"boxes": boxes, "circles": circles}
        self.pillar_occl = [(cx, cy, rad + 0.015) for (cx, cy, rad) in circles]
        self.grids = [RoverGrid(r, self.homes[r], info_) for r in range(2)]
        self.objectives = objectives
        self.samples = samples
        self.assign = None
        self.plans = [None, None]
        self.hbins = [None, None]
        self.th0 = [0.0, 0.0]
        self.bad = [[], []]
        self._vcache = {}
        self.extra_occl = [[], []]
        self.t = [0, 0]
        self._full_plan(state)

    # ------------------------------------------------------------------
    def _masks(self, r):
        G = self.grids[r]
        objd = []
        objr = []
        objr_opt = []
        self._objr_opt = objr_opt
        th0 = self.homes[r][2]
        for i, o in enumerate(self.objectives):
            d = G.dist(o["x"], o["y"])
            objd.append(d)
            objr.append(self._vis(r, th0, ("o", i), o["x"], o["y"], o["z"], 2.0,
                                  self._occl_obj(o)).any(-1))
            objr_opt.append(self._vis(r, th0, ("o", i), o["x"], o["y"], o["z"], 2.0,
                                      self._occl_obj_opt(o), opt=True).any(-1))
        samd = [G.dist(s["x"], s["y"]) for s in self.samples]
        lx, ly = self.lander_xy
        self._comm_cone = self._vis(r, th0, "L", lx, ly, self.lander_z, 4.0,
                                    self._comm_occl(r))
        comm = self._comm_cone.any(-1)
        hx, hy, _ = self.homes[r]
        home = G.dist(hx, hy) <= HOME_R
        return objd, objr, samd, comm, home

    def _img_mask(self, r, o, held, objd, objr):
        m = objr[o].copy()
        for o2 in range(len(self.objectives)):
            if o2 == o or o2 in held:
                continue
            m &= ~(self._objr_opt[o2] & (objd[o2] < objd[o] + 0.12))
        return m

    def _sample_mask(self, s, samd):
        m = samd[s] <= SAMPLE_R
        for s2 in range(len(self.samples)):
            if s2 != s:
                m &= samd[s2] > samd[s] + 0.03
        return m

    def _dp(self, r, rv, obj_tasks, sample_tasks, held0, need_send0,
            only_full=False):
        """Subset DP. Returns dict key->(cost, final ENode, final cell).

        obj_tasks: list of objective indices; sample_tasks: list of sample idx.
        Subsets allowed: any objectives, <=1 rock, <=1 soil sample.
        """
        G = self.grids[r]
        objd, objr, samd, comm, home = self._masks(r)
        cal_mask = np.zeros(G.shape, dtype=bool)
        for m in objr:
            cal_mask |= m
        start = G.cell_of(rv["x"], rv["y"])
        D0 = np.full(G.shape, INF, dtype=np.int64)
        D0[start] = 0
        root = ENode(G.prop(D0), [], root=True)
        cal0 = 1 if rv["calibrated"] > 0.5 else 0
        full0 = 1 if rv["store_full"] > 0.5 else 0
        tasks = [("o", o) for o in obj_tasks] + [("s", s) for s in sample_tasks]
        nT = len(tasks)
        soil = [self.samples[s]["is_soil"] > 0.5 for s in range(len(self.samples))]
        allmask = np.ones(G.shape, dtype=bool)

        def valid(mask_bits):
            nr = ns = 0
            for k in range(nT):
                if mask_bits >> k & 1 and tasks[k][0] == "s":
                    if soil[tasks[k][1]]:
                        ns += 1
                    else:
                        nr += 1
            return nr <= 1 and ns <= 1

        smasks = {}
        for k in range(nT):
            if tasks[k][0] == "s":
                smasks[k] = self._sample_mask(tasks[k][1], samd) & G.free
        imask_cache = {}

        def imask_for(k, bits):
            held = set(held0)
            for k2 in range(nT):
                if bits >> k2 & 1 and tasks[k2][0] == "o":
                    held.add(tasks[k2][1])
            key = (k, frozenset(held))
            if key not in imask_cache:
                imask_cache[key] = self._img_mask(r, tasks[k][1], held, objd, objr)
            return imask_cache[key]

        nodes = {(0, cal0, full0): root}
        full_bits = (1 << nT) - 1
        order = sorted(range(1 << nT), key=lambda b: bin(b).count("1"))
        flag_order = [(0, 1), (1, 1), (0, 0), (1, 0)]
        for bits in order:
            if not valid(bits):
                continue
            for (cf, ff) in flag_order:
                key = (bits, cf, ff)
                if key == (0, cal0, full0):
                    continue
                sources = []
                # calibrate: (bits,0,ff) -> (bits,1,ff)
                if cf == 1:
                    prev = nodes.get((bits, 0, ff))
                    if prev is not None:
                        sources.append((G.stage(prev.arr, cal_mask),
                                        (CALIBRATE, None), prev))
                # drop: (bits,cf,1) -> (bits,cf,0)
                if ff == 0:
                    prev = nodes.get((bits, cf, 1))
                    if prev is not None:
                        sources.append((G.stage(prev.arr, allmask), (DROP, None), prev))
                for k in range(nT):
                    if not bits >> k & 1:
                        continue
                    pb = bits & ~(1 << k)
                    kind, idx = tasks[k]
                    if kind == "o":
                        if cf != 0:
                            continue
                        prev = nodes.get((pb, 1, ff))
                        if prev is None:
                            continue
                        im = imask_for(k, pb)
                        if not im.any():
                            continue
                        sources.append((G.stage(prev.arr, im), (IMAGE, idx), prev))
                    else:
                        if ff != 1:
                            continue
                        prev = nodes.get((pb, cf, 0))
                        if prev is None or not smasks[k].any():
                            continue
                        sources.append((G.stage(prev.arr, smasks[k]), (SAMPLE, idx), prev))
                if not sources:
                    continue
                Dmin = sources[0][0]
                for s_ in sources[1:]:
                    Dmin = np.minimum(Dmin, s_[0])
                if Dmin.min() >= INF:
                    continue
                nodes[key] = ENode(G.prop(Dmin), sources)

        results = {}
        home_ok = home & G.free
        dist_home = G.prop(np.where(home_ok, 0, INF).astype(np.int64))
        lx, ly = self.lander_xy
        bc = self.homes[r][2] + R3.BW3 * np.arange(R3.NB3)
        hdiff = np.abs((bc - self.homes[r][2] + np.pi) % (2 * np.pi) - np.pi)
        rot_b = np.ceil(np.maximum(0.0, hdiff - CONE_HOME - BIN_W / 2) / 0.4 - 1e-9)
        krot = np.where(self._comm_cone, rot_b[None, None, :], 99).min(-1)
        tail = np.maximum(dist_home, krot.astype(np.int64))
        for (bits, cf, ff), node in nodes.items():
            if ff != 0:
                continue
            if only_full and bits != full_bits:
                continue
            need_send = need_send0 or bits != 0
            if need_send:
                Dsnd = G.stage(node.arr, comm)
                vals = np.where(Dsnd < INF, Dsnd + tail, INF)
                c = np.unravel_index(int(np.argmin(vals)), vals.shape)
                cost = int(vals[c])
                fin = ("send", Dsnd, node, int(Dsnd[c]), dist_home)
            else:
                vals = np.where(home_ok, node.arr, INF)
                c = np.unravel_index(int(np.argmin(vals)), vals.shape)
                cost = int(vals[c])
                fin = ("plain", node)
            if cost >= INF:
                continue
            objs = frozenset(tasks[k][1] for k in range(nT)
                             if bits >> k & 1 and tasks[k][0] == "o")
            sams = frozenset(tasks[k][1] for k in range(nT)
                             if bits >> k & 1 and tasks[k][0] == "s")
            if (objs, sams) not in results or cost < results[(objs, sams)][0]:
                results[(objs, sams)] = (cost, fin, c)
        return results

    def _make_plan(self, r, fin, c, cost):
        G = self.grids[r]
        if fin[0] == "plain":
            return self._backtrack(G, fin[1], c, cost)
        _, Dsnd, node, vs, dist_home = fin
        fake = ENode(np.full(G.shape, INF, dtype=np.int64),
                     [(Dsnd, (SEND, None), node)])
        plan = self._backtrack(G, fake, c, vs)
        # tail: descend dist_home, then wait (rotating) until cost reached
        cur = c
        while dist_home[cur] > 0:
            d = dist_home[cur]
            best = None
            for di in range(-K, K + 1):
                for dj in range(-K, K + 1):
                    a, b = cur[0] + di, cur[1] + dj
                    if 0 <= a < G.shape[0] and 0 <= b < G.shape[1]:
                        if dist_home[a, b] == d - 1:
                            score = max(abs(di), abs(dj)) * 10 + abs(di) + abs(dj)
                            if best is None or score < best[0]:
                                best = (score, (a, b))
            cur = best[1]
            plan.append((G.pos_of(cur), None))
        while len(plan) < cost:
            plan.append((G.pos_of(cur), None))
        return plan

    def _backtrack(self, G, fin, c, v):
        steps = []  # reversed list of (time, cell, opinfo)
        node = fin
        while True:
            if node.root and v == 0:
                break
            arr = node.arr
            i, j = c
            prevc = None
            if v > 0:
                best = None
                for di in range(-K, K + 1):
                    for dj in range(-K, K + 1):
                        a, b = i + di, j + dj
                        if 0 <= a < G.shape[0] and 0 <= b < G.shape[1]:
                            if arr[a, b] == v - 1:
                                score = max(abs(di), abs(dj)) * 10 + abs(di) + abs(dj)
                                if best is None or score < best[0]:
                                    best = (score, (a, b))
                if best is not None:
                    prevc = best[1]
            if prevc is not None:
                steps.append((v, c, None))
                c = prevc
                v -= 1
                continue
            done = False
            for (D, op, prev) in node.sources:
                if D[c] == v:
                    pa = prev.arr
                    best = None
                    for di in range(-K, K + 1):
                        for dj in range(-K, K + 1):
                            a, b = i + di, j + dj
                            if 0 <= a < G.shape[0] and 0 <= b < G.shape[1]:
                                if pa[a, b] == v - 1:
                                    score = max(abs(di), abs(dj)) * 10 + abs(di) + abs(dj)
                                    if best is None or score < best[0]:
                                        best = (score, (a, b))
                    if best is None:
                        continue
                    steps.append((v, c, op))
                    c = best[1]
                    v -= 1
                    node = prev
                    done = True
                    break
            if not done:
                raise RuntimeError("backtrack failed")
        steps.reverse()
        return [(G.pos_of(c), op) for (_, c, op) in steps]

    # ------------------------------------------------------------------
    def _status(self, state):
        rovers, _, objectives, samples, _ = self._parse(state)
        self.objectives = objectives
        self.samples = samples
        return rovers, objectives, samples

    def _rover_needs(self, r, rovers, objectives, samples):
        held0 = {i for i, o in enumerate(objectives)
                 if o[f"have_image_rover{r}"] > 0.5}
        need_send0 = any(o[f"have_image_rover{r}"] > 0.5 and o["received_image"] < 0.5
                         for o in objectives)
        need_send0 |= any(s[f"analyzed_rover{r}"] > 0.5 and s["received_analysis"] < 0.5
                          for s in samples)
        return held0, need_send0

    def _full_plan(self, state):
        rovers, objectives, samples = self._status(state)
        n = len(objectives)
        res = []
        for r in range(2):
            held0, need_send0 = self._rover_needs(r, rovers, objectives, samples)
            obj_tasks = [i for i, o in enumerate(objectives)
                         if o["received_image"] < 0.5 and i not in held0]
            G = self.grids[r]
            sample_tasks = []
            for i, s in enumerate(samples):
                if (G.dist(s["x"], s["y"]) <= SAMPLE_R)[G.free].any():
                    sample_tasks.append(i)
            res.append(self._dp(r, rovers[r], obj_tasks, sample_tasks,
                                held0, need_send0))
        # objectives already covered (received, or held by a rover)
        covered = set()
        for i, o in enumerate(objectives):
            if o["received_image"] > 0.5 or o["have_image_rover0"] > 0.5 \
                    or o["have_image_rover1"] > 0.5:
                covered.add(i)
        have_rock = any((s["analyzed_rover0"] > 0.5 or s["analyzed_rover1"] > 0.5)
                        and s["is_soil"] < 0.5 for s in samples)
        have_soil = any((s["analyzed_rover0"] > 0.5 or s["analyzed_rover1"] > 0.5)
                        and s["is_soil"] > 0.5 for s in samples)
        best = None
        allobj = set(range(n))
        for k0, (c0, f0, e0) in res[0].items():
            for k1, (c1, f1, e1) in res[1].items():
                if (k0[0] | k1[0] | covered) != allobj:
                    continue
                ss = k0[1] | k1[1]
                rock = have_rock or any(samples[s]["is_soil"] < 0.5 for s in ss)
                soil = have_soil or any(samples[s]["is_soil"] > 0.5 for s in ss)
                if not (rock and soil):
                    continue
                key = (max(c0, c1), c0 + c1)
                if best is None or key < best[0]:
                    best = (key, (k0, k1))
        cands = []
        for k0, (c0, f0, e0) in res[0].items():
            for k1, (c1, f1, e1) in res[1].items():
                if (k0[0] | k1[0] | covered) != allobj:
                    continue
                ss = k0[1] | k1[1]
                rock = have_rock or any(samples[s]["is_soil"] < 0.5 for s in ss)
                soil = have_soil or any(samples[s]["is_soil"] > 0.5 for s in ss)
                if rock and soil:
                    cands.append(((max(c0, c1), c0 + c1), k0, k1))
        if best is None:
            self.assign = None
            self.plans = [[], []]
            self.hbins = [None, None]
            return
        cands.sort(key=lambda x: x[0])
        self.cost2d = cands[0][0][0] if cands else -1
        cache = {}
        t0 = time.time()

        def ref(r, k):
            if (r, k) not in cache:
                cost, fin, c = res[r][k]
                plan = self._make_plan(r, fin, c, cost)
                cache[(r, k)] = self._refine_compute(r, rovers[r], plan)
            return cache[(r, k)]
        best3 = None
        for i, (key2, k0, k1) in enumerate(cands):
            if best3 is not None:
                if key2[0] > best3[0][0] or (key2[0] == best3[0][0] and key2[1] >= best3[0][1]):
                    break
                if i >= MAX_CANDS or time.time() - t0 > CAND_TIME:
                    break
            o0, o1 = ref(0, k0), ref(1, k1)
            key3 = (max(o0[0], o1[0]), o0[0] + o1[0])
            if best3 is None or key3 < best3[0]:
                best3 = (key3, k0, k1, o0, o1)
        self.assign = [best3[1], best3[2]]
        self._apply_refined(0, best3[3])
        self._apply_refined(1, best3[4])
        self._fix_comm(rovers)
        self.planned_cost = max(len(self.plans[0]), len(self.plans[1]))

    def _replan_rover(self, r, state):
        self.n_replans = getattr(self, "n_replans", 0) + 1
        rovers, objectives, samples = self._status(state)
        held0, need_send0 = self._rover_needs(r, rovers, objectives, samples)
        objs, sams = self.assign[r]
        obj_tasks = [i for i in objs if i not in held0
                     and objectives[i]["received_image"] < 0.5]
        sample_tasks = []
        for s in sams:
            typ = samples[s]["is_soil"] > 0.5
            done = any(s2[f"analyzed_rover{r}"] > 0.5 and (s2["is_soil"] > 0.5) == typ
                       for s2 in samples)
            if not done:
                sample_tasks.append(s)
        res = self._dp(r, rovers[r], obj_tasks, sample_tasks, held0, need_send0,
                       only_full=True)
        key = (frozenset(obj_tasks), frozenset(sample_tasks))
        if key not in res:
            self.plans[r] = []
            self.hbins[r] = None
            self.t[r] = 0
            return
        cost, fin, c = res[key]
        self.plans[r] = self._make_plan(r, fin, c, cost)
        self.t[r] = 0
        self._refine(r, rovers[r])
        self._fix_comm(rovers, which=(r,))

    def _refine(self, r, rv):
        """Replace plans[r] by an exact heading-aware plan for its op sequence."""
        out = self._refine_compute(r, rv, self.plans[r])
        self._apply_refined(r, out)
        return out[0]

    def _apply_refined(self, r, out):
        cost, plan, hb, th0 = out
        self.plans[r] = plan
        self.hbins[r] = hb
        self.th0[r] = th0
        self.t[r] = 0

    def _refine_compute(self, r, rv, plan):
        G = self.grids[r]
        th0 = rv["theta"]
        bc = th0 + R3.BW3 * np.arange(R3.NB3)
        ops = [op for (_, op) in plan if op is not None]
        objectives = self.objectives
        held = {i for i, o in enumerate(objectives) if o[f"have_image_rover{r}"] > 0.5}
        vis = {}
        viso = {}

        def VO(i):
            if i not in viso:
                o = objectives[i]
                viso[i] = self._vis(r, th0, ("o", i), o["x"], o["y"], o["z"], 2.0,
                                    self._occl_obj_opt(o), opt=True)
            return viso[i]

        def V(i):
            if i not in vis:
                o = objectives[i]
                vis[i] = self._vis(r, th0, ("o", i), o["x"], o["y"], o["z"], 2.0,
                                   self._occl_obj(o))
            return vis[i]
        shape3 = G.shape + (R3.NB3,)
        samd = None
        masks = []
        for (kind, idx) in ops:
            if kind == CALIBRATE:
                m = np.zeros(shape3, dtype=bool)
                for i in range(len(objectives)):
                    m |= V(i)
            elif kind == IMAGE:
                m = V(idx).copy()
                d0 = G.dist(objectives[idx]["x"], objectives[idx]["y"])
                for o2 in range(len(objectives)):
                    if o2 == idx or o2 in held:
                        continue
                    d2 = G.dist(objectives[o2]["x"], objectives[o2]["y"])
                    m &= ~(VO(o2) & (d2 < d0 + 0.12)[:, :, None])
                held.add(idx)
            elif kind == SAMPLE:
                if samd is None:
                    samd = [G.dist(s["x"], s["y"]) for s in self.samples]
                m = np.broadcast_to(self._sample_mask(idx, samd)[:, :, None], shape3)
            elif kind == SEND:
                lx, ly = self.lander_xy
                m = self._vis(r, th0, "L", lx, ly, self.lander_z, 4.0,
                              self._comm_occl(r))
            else:
                m = np.ones(shape3, dtype=bool)
            for (bx, by, bth, bk) in self.bad[r]:
                if bk == kind:
                    ad = np.abs((bc - bth + np.pi) % (2 * np.pi) - np.pi) <= 0.1
                    m = m & ~((G.dist(bx, by) <= 0.03)[:, :, None] & ad[None, None, :])
            masks.append(m)
        hx, hy, hth = self.homes[r]
        hd = np.abs((bc - hth + np.pi) % (2 * np.pi) - np.pi) <= HOME_TOL3
        home3 = (G.dist(hx, hy) <= HOME_R)[:, :, None] & hd[None, None, :]
        try:
            res = R3.solve(G.free, G.cell_of(rv["x"], rv["y"]), masks, home3)
        except Exception:
            res = None
        if res is None:
            return (len(plan) + 3, plan, None, th0)
        cost, steps = res
        newplan = [(G.pos_of(c), None if k is None else ops[k])
                   for (c, b, k) in steps]
        return (cost, newplan, [b for (c, b, k) in steps], th0)

    # ------------------------------------------------------------------
    def _cone(self, centers, half):
        """Boolean bin mask of headings within `half` of any center."""
        m = np.zeros(NB, dtype=bool)
        for c in centers:
            d = np.abs((BIN_C - c + np.pi) % (2 * np.pi) - np.pi)
            m |= d <= half
        return m

    def _comm_occl(self, r):
        ho = self.homes[1 - r]
        return list(self.pillar_occl) + [(ho[0], ho[1], 0.27)] + list(self.extra_occl[r])

    def _pos_at(self, r, d):
        """Planned xy of rover r, d steps from now (None if unknown)."""
        plan = self.plans[r]
        if not plan:
            return None
        k = min(self.t[r] + d, len(plan) - 1)
        return plan[k][0]

    def _fix_comm(self, rovers, which=(0, 1)):
        """Re-refine rovers whose planned send is blocked by the other rover."""
        lx, ly = self.lander_xy
        for _ in range(3):
            changed = False
            for r in which:
                if self.hbins[r] is None or self.t[r] != 0:
                    continue
                o = 1 - r
                for k, ((px, py), op) in enumerate(self.plans[r]):
                    if op is None or op[0] != SEND:
                        continue
                    qs = [self._pos_at(o, k - 1) if k > 0 else None, self._pos_at(o, k)]
                    qs = [q for q in qs if q is not None] or [(rovers[o]["x"], rovers[o]["y"])]
                    th = self.th0[r] + R3.BW3 * self.hbins[r][k]
                    cx = px - CAM_OFF * math.cos(th) + 0.0125 * math.sin(th)
                    cy = py - CAM_OFF * math.sin(th) - 0.0125 * math.cos(th)
                    vx, vy = lx - cx, ly - cy
                    rel = abs(wrap(math.atan2(vy, vx) - th))
                    if rel >= REL_MIN:
                        continue
                    vv = vx * vx + vy * vy + 1e-12
                    hitq = None
                    for q in qs:
                        sp = min(max(((q[0] - cx) * vx + (q[1] - cy) * vy) / vv, 0.0), 1.0)
                        dd = math.hypot(q[0] - cx - sp * vx, q[1] - cy - sp * vy)
                        if dd <= 0.27:
                            hitq = q
                    if hitq is not None:
                        q = hitq
                        if DEBUG:
                            print(f"FIXCOMM rover{r} send@{k} blocked by rover{o} at {q}")
                        self.extra_occl[r].append((q[0], q[1], 0.27))
                        self._refine(r, rovers[r])
                        changed = True
                        break
            if not changed:
                break

    def _vis(self, r, th0, key, tx, ty, tz, rng, occl, opt=False):
        k = (r, th0, key, opt, tuple(occl))
        v = self._vcache.get(k)
        if v is None:
            G = self.grids[r]
            bc = th0 + R3.BW3 * np.arange(R3.NB3)
            v = vis_cone(G.X, G.Y, tx, ty, tz, rng, occl, bc=bc, opt=opt)
            if len(self._vcache) > 64:
                self._vcache.clear()
            self._vcache[k] = v
        return v

    def _occl_obj_opt(self, target):
        occ = [(cx, cy, 0.04) for (cx, cy, _) in self.pillar_occl]
        for o in self.objectives:
            if o is target or (o["x"] == target["x"] and o["y"] == target["y"]):
                continue
            occ.append((o["x"], o["y"], 0.025))
        return occ

    def _occl_obj(self, target):
        occ = list(self.pillar_occl)
        for o in self.objectives:
            if o is target or (o["x"] == target["x"] and o["y"] == target["y"]):
                continue
            occ.append((o["x"], o["y"], 0.065))
        return occ

    def _op_cone(self, r, px, py, op):
        kind, idx = op
        if kind == IMAGE:
            o = self.objectives[idx]
            return vis_cone(px, py, o["x"], o["y"], o["z"], 2.0, self._occl_obj(o))
        if kind == CALIBRATE:
            m = np.zeros(NB, dtype=bool)
            for o in self.objectives:
                if math.hypot(px - o["x"], py - o["y"]) <= R_OBJ_BASE + 0.1:
                    m |= vis_cone(px, py, o["x"], o["y"], o["z"], 2.0, self._occl_obj(o))
            return m
        if kind == SEND:
            lx, ly = self.lander_xy
            return vis_cone(px, py, lx, ly, self.lander_z, 4.0, self._comm_occl(r))
        return None

    def _desired_heading(self, r, rv):
        plan = self.plans[r]
        t = self.t[r]
        cons = []
        for k in range(t, len(plan)):
            (px, py), op = plan[k]
            if op is None:
                continue
            m = self._op_cone(r, px, py, op)
            if m is not None:
                cons.append((k, m))
        kend = max(len(plan) - 1, t)
        cons.append((kend, self._cone([self.homes[r][2]], CONE_HOME)))
        # backward pass
        F = [None] * len(cons)
        F[-1] = cons[-1][1]
        for i in range(len(cons) - 2, -1, -1):
            dk = cons[i + 1][0] - cons[i][0]
            F[i] = cons[i][1] & dilate(F[i + 1], STEP_BINS * dk)
        target = F[0] if F[0].any() else cons[0][1]
        if not target.any():
            return rv["theta"]
        th = rv["theta"]
        d = (BIN_C - th + np.pi) % (2 * np.pi) - np.pi
        depth = np.zeros(NB, dtype=int)
        cur = target.copy()
        for k in range(1, 4):
            cur = cur & np.roll(cur, 1) & np.roll(cur, -1)
            depth += cur
        want = min(3, int(depth.max()))
        cand = np.where(target & (depth >= want))[0]
        cur_bin = int((th + np.pi) / BIN_W) % NB
        if target[cur_bin] and depth[cur_bin] >= want:
            return th
        best = cand[np.argmin(np.abs(d[cand]))]
        return th + d[best]

    def _expect(self, r, op, prev, rv, objectives, samples):
        """Check whether the previous op had its effect."""
        kind, idx = op
        if kind == CALIBRATE:
            return rv["calibrated"] > 0.5
        if kind == IMAGE:
            return objectives[idx][f"have_image_rover{r}"] > 0.5
        if kind == SAMPLE:
            return any(s[f"analyzed_rover{r}"] > 0.5 and
                       (s["is_soil"] > 0.5) == (samples[idx]["is_soil"] > 0.5)
                       for s in samples)
        if kind == DROP:
            return rv["store_full"] < 0.5
        if kind == SEND:
            ok = all(o["received_image"] > 0.5 for o in objectives
                     if o[f"have_image_rover{r}"] > 0.5)
            ok &= all(s["received_analysis"] > 0.5 for s in samples
                      if s[f"analyzed_rover{r}"] > 0.5)
            return ok
        return True

    def get_action(self, state):
        rovers, objectives, samples = self._status(state)
        action = np.zeros(8, dtype=np.float32)
        for r in range(2):
            rv = rovers[r]
            plan = self.plans[r]
            t = self.t[r]
            # verify previous step
            if plan is not None and t > 0 and self.assign is not None:
                (px, py), op = plan[t - 1]
                bad = math.hypot(rv["x"] - px, rv["y"] - py) > 0.02
                if op is not None and not self._expect(r, op, None, rv,
                                                       objectives, samples):
                    bad = True
                    self.bad[r].append((rv["x"], rv["y"], rv["theta"], op[0]))
                if bad:
                    if DEBUG:
                        print(f"REPLAN rover{r} t={t} op={op} planned=({px:.2f},{py:.2f}) actual=({rv['x']:.2f},{rv['y']:.2f},{rv['theta']:.2f})")
                    self._replan_rover(r, state)
                    plan = self.plans[r]
                    t = self.t[r]
            if self.assign is None:
                self._full_plan(state)
                plan = self.plans[r]
                t = self.t[r]
            if self.hbins[r] is not None:
                if t < len(plan):
                    th_des = self.th0[r] + R3.BW3 * self.hbins[r][t]
                else:
                    th_des = rv["theta"]
            else:
                th_des = self._desired_heading(r, rv)
            dth = float(np.clip(wrap(th_des - rv["theta"]), -0.4, 0.4))
            if plan and t < len(plan):
                (px, py), op = plan[t]
                dx = float(np.clip(px - rv["x"], -0.2, 0.2))
                dy = float(np.clip(py - rv["y"], -0.2, 0.2))
                opk = NOOP if op is None else op[0]
                self.t[r] = t + 1
            else:
                hx, hy, _ = self.homes[r]
                far = math.hypot(hx - rv["x"], hy - rv["y"]) > HOME_R + 0.01
                dx = float(np.clip(hx - rv["x"], -0.2, 0.2)) if far else 0.0
                dy = float(np.clip(hy - rv["y"], -0.2, 0.2)) if far else 0.0
                opk = NOOP
            action[4 * r:4 * r + 4] = [dx, dy, dth, op_value(opk)]
        return action
