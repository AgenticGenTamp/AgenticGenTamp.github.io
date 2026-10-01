"""Motion2D approach: grid A* in (x, y) with Chebyshev step metric.

Primary planner: robot keeps a fixed orientation (initial theta, or an axis
aligned one) and plans on a nonuniform grid that includes the "critical"
coordinates (obstacle edge +/- robot extents) so exact-fit passages are found.
Fallback planner: layered A* over (x, y, theta-layer) where layer switches are
in-place rotations checked at every pi/16 intermediate pose.
"""
import heapq
import math

import numpy as np
from scipy import ndimage

STEP = 0.05
DTH = math.pi / 16  # max rotation per step (0.19635)
WORLD = (0.0, 2.5, 0.0, 2.5)
MARGINS = (0.004, 0.0005, -0.0001)
AXES = (0.0, math.pi / 2, math.pi, -math.pi / 2)


def _feat(state, obj, f, default=0.0):
    try:
        return float(state.get(obj, f))
    except Exception:
        return default


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.res = 0.01
        self.margin = 0.004
        self.ptheta = 0.0
        self.arm = 0.1
        self.base_r = 0.1
        self.grip_w = 0.01
        self.grip_h = 0.07
        self.obstacles = []
        self.target = None
        self.robot = None
        self._clear_exec()

    def _clear_exec(self):
        self.segments = None
        self.seg = 0
        self.path = None
        self.layered = False
        self.idx = 0
        self.last_move = (0.0, 0.0)
        self.last_pos = None
        self.last_full = None
        self.stuck = 0
        self.idle = 0
        self.t = 0
        self.last_plan_t = 0
        self.bad = set()
        self.n_stuck_replans = 0

    # ------------------------------------------------------------------ parsing
    def _parse(self, state):
        robot = None
        target = None
        rects = []
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            tname = getattr(obj.type, "name", str(obj.type))
            if tname == "crv_robot":
                robot = obj
            elif tname == "target_region":
                target = obj
            elif tname == "rectangle":
                rects.append(obj)
        self.robot = robot
        self.base_r = _feat(state, robot, "base_radius", 0.1)
        self.grip_h = _feat(state, robot, "gripper_height", 0.07)
        self.grip_w = _feat(state, robot, "gripper_width", 0.01)
        obs = []
        for o in rects:
            x, y = _feat(state, o, "x"), _feat(state, o, "y")
            th = _feat(state, o, "theta")
            w, h = _feat(state, o, "width"), _feat(state, o, "height")
            c, s = math.cos(th), math.sin(th)
            if w < 0:
                x, y, w = x + w * c, y + w * s, -w
            if h < 0:
                x, y, h = x - h * s, y + h * c, -h
            obs.append((x, y, th, w, h))
        self.obstacles = obs
        if target is not None:
            self.target = tuple(_feat(state, target, f) for f in ("x", "y", "theta", "width", "height"))
        else:
            self.target = None

    # ---------------------------------------------------------------- geometry
    def _rect_dist(self, px, py):
        """Min distance from points (arrays) to any obstacle rect (0 inside)."""
        d = np.full(np.shape(px), np.inf)
        for (x, y, th, w, h) in self.obstacles:
            c, s = math.cos(th), math.sin(th)
            dx, dy = px - x, py - y
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            ex = np.maximum(np.maximum(-lx, lx - w), 0.0)
            ey = np.maximum(np.maximum(-ly, ly - h), 0.0)
            d = np.minimum(d, np.hypot(ex, ey))
        return d

    def _grip_offsets(self, n=11, theta=None):
        th = self.ptheta if theta is None else theta
        c, s = math.cos(th), math.sin(th)
        arm = self.arm
        out = []
        for a in (arm - self.grip_w / 2, arm, arm + self.grip_w / 2):
            for b in np.linspace(-self.grip_h / 2, self.grip_h / 2, n):
                out.append((a * c - b * s, a * s + b * c))
        return out

    def _free(self, px, py, d=None):
        """Robot (base + gripper at self.ptheta) collision-free at points."""
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        shape = px.shape
        px = px.ravel()
        py = py.ravel()
        r = self.base_r + self.margin
        x0, x1, y0, y1 = WORLD
        ok = (px - r >= x0) & (px + r <= x1) & (py - r >= y0) & (py + r <= y1)
        if d is None:
            d = self._rect_dist(px, py)
        else:
            d = np.asarray(d).ravel()
        ok &= d > r
        reach = self.arm + math.hypot(self.grip_w / 2, self.grip_h / 2) + abs(self.margin) + 0.01
        near = ok & ((d < reach) | (px - reach < x0) | (px + reach > x1)
                     | (py - reach < y0) | (py + reach > y1))
        idx = np.nonzero(near)[0]
        if idx.size:
            ok[idx] = self._grip_clear(px[idx], py[idx])
        return ok.reshape(shape)

    def _grip_clear(self, px, py):
        """Exact SAT test: gripper rect (at self.ptheta, inflated by margin)
        does not overlap any obstacle and stays inside the world."""
        th = self.ptheta
        c, s = math.cos(th), math.sin(th)
        m = self.margin
        a1 = self.grip_w / 2 + max(m, 0.0)
        a2 = self.grip_h / 2 + max(m, 0.0)
        cx = px + self.arm * c
        cy = py + self.arm * s
        x0, x1, y0, y1 = WORLD
        ex = a1 * abs(c) + a2 * abs(s)
        ey = a1 * abs(s) + a2 * abs(c)
        ok = (cx - ex - m >= x0) & (cx + ex + m <= x1) & (cy - ey - m >= y0) & (cy + ey + m <= y1)
        ua = (c, s)
        va = (-s, c)
        for (ox, oy, oth, w, h) in self.obstacles:
            oc, os_ = math.cos(oth), math.sin(oth)
            ub = (oc, os_)
            vb = (-os_, oc)
            b1, b2 = w / 2, h / 2
            bcx = ox + b1 * oc - b2 * os_
            bcy = oy + b1 * os_ + b2 * oc
            tx = bcx - cx
            ty = bcy - cy
            sep = np.zeros(px.shape, dtype=bool)
            for L in (ua, va, ub, vb):
                ra = a1 * abs(ua[0] * L[0] + ua[1] * L[1]) + a2 * abs(va[0] * L[0] + va[1] * L[1])
                rb = b1 * abs(ub[0] * L[0] + ub[1] * L[1]) + b2 * abs(vb[0] * L[0] + vb[1] * L[1])
                sep |= np.abs(tx * L[0] + ty * L[1]) > ra + rb
            ok &= sep
        return ok

    def _pose_ok(self, x, y, th, dth, margin=0.002):
        """Rotation from th by dth at (x, y) collision-free (pi/16 samples)."""
        saved = (self.ptheta, self.margin)
        ok = True
        try:
            self.margin = margin
            n = max(1, int(math.ceil(abs(dth) / DTH - 1e-9)))
            for k in range(0, n + 1):
                self.ptheta = th + dth * k / float(n)
                if not bool(self._free(np.array([x]), np.array([y]))[0]):
                    ok = False
                    break
        finally:
            self.ptheta, self.margin = saved
        return ok

    def _seg_free(self, a, b):
        n = max(2, int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 0.004)) + 1)
        t = np.linspace(0, 1, n)
        return bool(np.all(self._free(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))))

    def _in_target(self, px, py, shrink=0.0):
        x, y, th, w, h = self.target
        c, s = math.cos(th), math.sin(th)
        dx, dy = px - x, py - y
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        return (lx >= shrink) & (lx <= w - shrink) & (ly >= shrink) & (ly <= h - shrink)

    # ---------------------------------------------------------------- planning
    def _axis_coords(self, lo, hi, crit):
        res = self.res
        n = int(round((hi - lo) / res))
        base = [lo + (k + 0.5) * res for k in range(n)]
        crit = sorted(c for c in crit if lo < c < hi)
        carr = np.array(crit) if crit else np.array([1e9])
        out = [b for b in base if np.min(np.abs(carr - b)) >= 1e-3]
        out.extend(crit)
        return np.array(sorted(set(round(v, 7) for v in out)))

    def _grid(self, thetas):
        x0, x1, y0, y1 = WORLD
        e = self.margin + 2e-5
        r = self.base_r + e
        cx = [x0 + r, x1 - r]
        cy = [y0 + r, y1 - r]
        ext = []
        for th in thetas:
            go = self._grip_offsets(2, th)
            ext.append((max(o[0] for o in go) + e, min(o[0] for o in go) - e,
                        max(o[1] for o in go) + e, min(o[1] for o in go) - e))
        for (gxp, gxm, gyp, gym) in ext:
            cx += [x0 - gxm, x1 - gxp]
            cy += [y0 - gym, y1 - gyp]
        for (x, y, th, w, h) in self.obstacles:
            if abs(math.sin(th)) > 1e-6:
                continue
            xa, xb = min(x, x + w), max(x, x + w)
            ya, yb = min(y, y + h), max(y, y + h)
            cx += [xa - r, xb + r]
            cy += [ya - r, yb + r]
            for (gxp, gxm, gyp, gym) in ext:
                cx += [xa - gxp, xb - gxm]
                cy += [ya - gyp, yb - gym]
        tx, ty, tth, tw, th_ = self.target
        cx += [tx + 0.5 * tw]
        cy += [ty + 0.5 * th_]
        xs = self._axis_coords(x0, x1, cx)
        ys = self._axis_coords(y0, y1, cy)
        return xs, ys

    def _heur(self, xs, ys):
        tx, ty, tth, tw, th_ = self.target
        c, s = math.cos(tth), math.sin(tth)
        cs = [(tx, ty), (tx + tw * c, ty + tw * s), (tx - th_ * s, ty + th_ * c),
              (tx + tw * c - th_ * s, ty + tw * s + th_ * c)]
        bx0 = min(p[0] for p in cs); bx1 = max(p[0] for p in cs)
        by0 = min(p[1] for p in cs); by1 = max(p[1] for p in cs)
        hx = np.maximum(np.maximum(bx0 + 0.003 - xs, 0.0), xs - bx1 + 0.003).tolist()
        hy = np.maximum(np.maximum(by0 + 0.003 - ys, 0.0), ys - by1 + 0.003).tolist()
        return hx, hy

    def _astar(self, xs, ys, frees, goals, rots, start, k0):
        """A* over (layer, i, j). frees/goals: list of bool arrays per layer.
        rots: dict k -> list of (k2, mask, cost). Returns list of (k, x, y)."""
        nx, ny = len(xs), len(ys)
        GX, GY = np.meshgrid(xs, ys, indexing="ij")
        f0 = frees[k0]
        si = int(np.argmin(np.abs(xs - start[0])))
        sj = int(np.argmin(np.abs(ys - start[1])))
        if not f0[si, sj]:
            d = np.maximum(np.abs(GX - start[0]), np.abs(GY - start[1]))
            d = np.where(f0, d, np.inf)
            k = int(np.argmin(d))
            if not np.isfinite(d.flat[k]) or d.flat[k] > 0.03:
                return None
            si, sj = (int(v) for v in np.unravel_index(k, d.shape))
        hx, hy = self._heur(xs, ys)
        xl = xs.tolist(); yl = ys.tolist()
        freel = [f.tolist() for f in frees]
        goall = [g.tolist() for g in goals]
        rotl = {k: [(k2, m.tolist(), c) for (k2, m, c) in v] for k, v in rots.items()}
        s0 = (k0, si, sj)
        g = {s0: 0.0}
        parent = {s0: None}
        pq = [(max(hx[si], hy[sj]), 0.0, k0, si, sj)]
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        closed = set()
        found = None
        while pq:
            f, gc, k, i, j = heapq.heappop(pq)
            st = (k, i, j)
            if st in closed:
                continue
            closed.add(st)
            if goall[k][i][j]:
                found = st
                break
            fr = freel[k]
            for di, dj in nbrs:
                a, b = i + di, j + dj
                if a < 0 or b < 0 or a >= nx or b >= ny or not fr[a][b]:
                    continue
                if di and dj and not (fr[a][j] and fr[i][b]):
                    continue
                ddx = abs(xl[a] - xl[i]); ddy = abs(yl[b] - yl[j])
                ng = gc + max(ddx, ddy) + 1e-3 * (ddx + ddy)
                ns = (k, a, b)
                if ng < g.get(ns, 1e18):
                    g[ns] = ng
                    parent[ns] = st
                    heapq.heappush(pq, (ng + max(hx[a], hy[b]), ng, k, a, b))
            for (k2, m, cost) in rotl.get(k, ()):
                if m[i][j]:
                    ng = gc + cost
                    ns = (k2, i, j)
                    if ng < g.get(ns, 1e18):
                        g[ns] = ng
                        parent[ns] = st
                        heapq.heappush(pq, (ng + max(hx[i], hy[j]), ng, k2, i, j))
        if found is None:
            return None
        out = []
        c = found
        while c is not None:
            out.append((c[0], xl[c[1]], yl[c[2]]))
            c = parent[c]
        out.reverse()
        return out

    def _goal_mask(self, xs, ys, free):
        GX, GY = np.meshgrid(xs, ys, indexing="ij")
        goal = self._in_target(GX, GY, 0.003) & free
        if not goal.any():
            goal = self._in_target(GX, GY, 0.0) & free
        return goal

    def _plan_single_m(self, start, theta):
        self.ptheta = theta
        xs, ys = self._grid([theta])
        GX, GY = np.meshgrid(xs, ys, indexing="ij")
        free = self._free(GX, GY)
        goal = self._goal_mask(xs, ys, free)
        if not goal.any():
            return None
        # cheap connectivity pre-check (8-connected superset of A* moves)
        lab, _ = ndimage.label(free, structure=np.ones((3, 3), dtype=bool))
        si = int(np.argmin(np.abs(xs - start[0])))
        sj = int(np.argmin(np.abs(ys - start[1])))
        w = 4
        sl = lab[max(0, si - w):si + w + 1, max(0, sj - w):sj + w + 1]
        slabs = set(np.unique(sl[sl > 0]).tolist())
        glabs = set(np.unique(lab[goal]).tolist())
        if not (slabs & glabs):
            return None
        res = self._astar(xs, ys, [free], [goal], {}, start, 0)
        if res is None:
            return None
        pts = [(float(start[0]), float(start[1]))] + [(x, y) for (_, x, y) in res]
        return [(theta, pts)]

    def _plan_layered_m(self, start, th0):
        thetas = [th0] + [a for a in AXES if abs(_wrap(a - th0)) > 1e-3]
        xs, ys = self._grid(thetas)
        GX, GY = np.meshgrid(xs, ys, indexing="ij")
        D = self._rect_dist(GX, GY)
        frees, goals = [], []
        for th in thetas:
            self.ptheta = th
            f = self._free(GX, GY, D)
            frees.append(f)
            goals.append(self._goal_mask(xs, ys, f))
        if not any(g.any() for g in goals):
            return None
        # rotation edges between angularly adjacent layers
        order = sorted(range(len(thetas)), key=lambda k: thetas[k] % (2 * math.pi))
        rots = {}
        L = len(order)
        for idx in range(L):
            ka, kb = order[idx], order[(idx + 1) % L]
            if ka == kb:
                continue
            ta = thetas[ka]
            dth = (thetas[kb] - ta) % (2 * math.pi)
            if dth < 1e-6:
                continue
            nsteps = max(1, int(math.ceil(dth / DTH - 1e-9)))
            mask = frees[ka] & frees[kb]
            cand = np.nonzero(mask.ravel())[0]
            if cand.size:
                px, py, dd = GX.ravel()[cand], GY.ravel()[cand], D.ravel()[cand]
                okc = np.ones(cand.size, dtype=bool)
                for s in range(1, nsteps):
                    self.ptheta = ta + dth * s / nsteps
                    okc &= self._free(px, py, dd)
                m2 = np.zeros(mask.size, dtype=bool)
                m2[cand[okc]] = True
                mask = m2.reshape(mask.shape)
            cost = nsteps * STEP
            rots.setdefault(ka, []).append((kb, mask, cost))
            rots.setdefault(kb, []).append((ka, mask, cost))
        res = self._astar(xs, ys, frees, goals, rots, start, 0)
        if res is None:
            return None
        segs = []
        cur_k = None
        for (k, x, y) in res:
            if k != cur_k:
                segs.append([thetas[k], []])
                cur_k = k
            segs[-1][1].append((x, y))
        segs[0][1].insert(0, (float(start[0]), float(start[1])))
        return [(th, pts) for th, pts in segs]

    def _make_plan(self, start, th, allow_axes=True):
        """Per margin tier: single-theta plans (current theta, then axes),
        then the layered planner. Skips (margin, theta) combos that got stuck."""
        cands = [th]
        if allow_axes:
            cands += [a for a in AXES if abs(_wrap(a - th)) > 1e-3]
        for attempt in range(2):
            for m in MARGINS:
                self.margin = m
                for c in cands:
                    if (round(m, 5), round(_wrap(c), 3)) in self.bad:
                        continue
                    p = self._plan_single_m(start, c)
                    if p is not None:
                        self._set_plan(p, layered=False)
                        return True
                if (round(m, 5), "L") in self.bad:
                    continue
                p = self._plan_layered_m(start, th)
                if p is not None:
                    self._set_plan(p, layered=True)
                    return True
            if not self.bad:
                break
            self.bad = set()
        self.segments = None
        self.path = None
        return False

    def _mark_bad(self):
        if self.segments is None:
            return
        if self.layered:
            self.bad.add((round(self.margin, 5), "L"))
        else:
            self.bad.add((round(self.margin, 5), round(_wrap(self.ptheta), 3)))

    def _set_plan(self, segs, layered):
        self.segments = segs
        self.layered = layered
        self.seg = 0
        self.ptheta = segs[0][0]
        self.path = segs[0][1]
        self.idx = 0

    # ------------------------------------------------------------------ acting
    def reset(self, state, info):
        self._clear_exec()
        self._parse(state)
        rx = _feat(state, self.robot, "x")
        ry = _feat(state, self.robot, "y")
        th = _feat(state, self.robot, "theta")
        self.arm = max(_feat(state, self.robot, "arm_joint", self.base_r), self.base_r)
        self._make_plan((rx, ry), th)

    def _follow(self, rx, ry):
        pts = self.path
        best_i, best_d = self.idx, 1e9
        for k in range(self.idx, min(len(pts), self.idx + 60)):
            d = max(abs(pts[k][0] - rx), abs(pts[k][1] - ry))
            if d < best_d:
                best_d, best_i = d, k
        self.idx = best_i
        cheb = lambda p: max(abs(p[0] - rx), abs(p[1] - ry))
        last_in = None
        for k in range(self.idx, len(pts)):
            if cheb(pts[k]) <= STEP + 1e-9:
                last_in = k
            else:
                break
        if last_in is None:
            last_in = self.idx
        tgt = None
        if last_in + 1 < len(pts) and cheb(pts[last_in]) <= STEP:
            a0, a1 = pts[last_in], pts[last_in + 1]
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                p = (a0[0] + mid * (a1[0] - a0[0]), a0[1] + mid * (a1[1] - a0[1]))
                if cheb(p) <= STEP:
                    lo = mid
                else:
                    hi = mid
            p = (a0[0] + lo * (a1[0] - a0[0]), a0[1] + lo * (a1[1] - a0[1]))
            if self._seg_free((rx, ry), p):
                tgt = p
        if tgt is None:
            for k in range(last_in, self.idx - 1, -1):
                if self._seg_free((rx, ry), pts[k]):
                    tgt = pts[k]
                    break
        if tgt is None:
            tgt = pts[min(self.idx + 1, len(pts) - 1)]
        dx = float(np.clip(tgt[0] - rx, -STEP, STEP))
        dy = float(np.clip(tgt[1] - ry, -STEP, STEP))
        return dx, dy

    def get_action(self, state):
        if self.robot is None:
            self._parse(state)
        rx = _feat(state, self.robot, "x")
        ry = _feat(state, self.robot, "y")
        th = _feat(state, self.robot, "theta")
        darm = -0.1
        moved_cmd = max(abs(self.last_move[0]), abs(self.last_move[1])) > 1e-7
        if moved_cmd and self.last_pos is not None and abs(rx - self.last_pos[0]) < 1e-6 \
                and abs(ry - self.last_pos[1]) < 1e-6:
            self.stuck += 1
        else:
            self.stuck = 0
        cur = (rx, ry, th)
        if self.last_full is not None and max(abs(cur[i] - self.last_full[i]) for i in range(3)) < 1e-6:
            self.idle += 1
        else:
            self.idle = 0
        self.last_full = cur
        self.last_pos = (rx, ry)
        self.t += 1

        replan = False
        if self.idle >= 4 and self.t - self.last_plan_t >= 10:
            replan = True
        elif self.segments is None and self.t - self.last_plan_t >= 25:
            replan = True
        elif self.stuck >= 2 and self.t - self.last_plan_t >= 3:
            replan = True
        if replan:
            self.n_stuck_replans += 1
            if self.n_stuck_replans >= 2 and self.margin < 0.004:
                self._mark_bad()
            self._make_plan((rx, ry), th)
            self.last_plan_t = self.t
            self.stuck = 0
            self.idle = 0

        if self.segments is None:
            tx, ty, _, tw, thh = self.target
            gx, gy = tx + tw / 2, ty + thh / 2
            self.last_move = (0.0, 0.0)
            return np.array([np.clip(gx - rx, -STEP, STEP), np.clip(gy - ry, -STEP, STEP),
                             0.0, darm, 0], dtype=np.float32)

        # layered plan: advance segment when its end point is reached
        if self.layered and self.seg + 1 < len(self.segments):
            end = self.path[-1]
            if max(abs(end[0] - rx), abs(end[1] - ry)) < 1e-4:
                self.seg += 1
                self.ptheta, self.path = self.segments[self.seg]
                self.idx = 0

        err = _wrap(self.ptheta - th)
        dth = float(np.clip(err, -DTH, DTH))
        if self.layered and self.seg > 0 and abs(err) > 1e-4:
            # rotate in place at the switch point
            self.last_move = (0.0, 0.0)
            return np.array([0.0, 0.0, dth, darm, 0.0], dtype=np.float32)

        dx, dy = self._follow(rx, ry)
        if abs(err) > 1e-4:
            # not aligned yet: only move if whole remaining rotation stays free
            if not (self._pose_ok(rx + dx, ry + dy, th, dth)
                    and self._pose_ok(rx + dx, ry + dy, th, err)):
                if self._pose_ok(rx, ry, th, dth):
                    dx = dy = 0.0
                elif self._pose_ok(rx + dx, ry + dy, th, 0.0):
                    dth = 0.0
                else:
                    dx = dy = 0.0
        if self.stuck == 1:
            dx *= 0.5
            dy *= 0.5
        self.last_move = (dx, dy)
        return np.array([dx, dy, dth, darm, 0.0], dtype=np.float32)
