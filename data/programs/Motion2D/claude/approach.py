"""Motion2D navigation: lattice A* (Chebyshev cost) + exact-geometry executor.

The robot is a disc of radius `base_radius` plus a small vacuum gripper box
carried at the tip of the (fully retracted) arm.  The gripper sticks out past
the disc, so it is modelled explicitly, both when planning (assuming the
heading the executor drives to) and when executing (at the true heading).
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------- geometry
def _pt_rect_dist(px, py, x0, y0, x1, y1):
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def _pt_seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < EPS:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / L2
    t = min(1.0, max(0.0, t))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _ccw(ax, ay, bx, by, cx, cy):
    return (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)


def _seg_seg_intersect(ax, ay, bx, by, cx, cy, dx_, dy_):
    d1 = _ccw(cx, cy, dx_, dy_, ax, ay)
    d2 = _ccw(cx, cy, dx_, dy_, bx, by)
    d3 = _ccw(ax, ay, bx, by, cx, cy)
    d4 = _ccw(ax, ay, bx, by, dx_, dy_)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _seg_rect_dist(ax, ay, bx, by, x0, y0, x1, y1):
    if x0 <= ax <= x1 and y0 <= ay <= y1:
        return 0.0
    if x0 <= bx <= x1 and y0 <= by <= y1:
        return 0.0
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    for i in range(4):
        cx, cy = corners[i]
        dx_, dy_ = corners[(i + 1) % 4]
        if _seg_seg_intersect(ax, ay, bx, by, cx, cy, dx_, dy_):
            return 0.0
    best = min(_pt_rect_dist(ax, ay, x0, y0, x1, y1),
               _pt_rect_dist(bx, by, x0, y0, x1, y1))
    for cx, cy in corners:
        d = _pt_seg_dist(cx, cy, ax, ay, bx, by)
        if d < best:
            best = d
    return best


class _Rect:
    """Rectangle given by an origin corner, a rotation and (width, height)."""

    __slots__ = ("cx", "cy", "ct", "st", "x0", "y0", "x1", "y1",
                 "axis_aligned", "bbox")

    def __init__(self, x, y, theta, w, h):
        self.cx = x
        self.cy = y
        self.ct = math.cos(theta)
        self.st = math.sin(theta)
        self.x0 = min(0.0, w)
        self.x1 = max(0.0, w)
        self.y0 = min(0.0, h)
        self.y1 = max(0.0, h)
        self.axis_aligned = abs(self.st) < 1e-9 and self.ct > 0
        self.bbox = self.world_bbox()

    def to_local(self, px, py):
        dx = px - self.cx
        dy = py - self.cy
        return (dx * self.ct + dy * self.st, -dx * self.st + dy * self.ct)

    def pt_dist(self, px, py):
        lx, ly = self.to_local(px, py)
        return _pt_rect_dist(lx, ly, self.x0, self.y0, self.x1, self.y1)

    def seg_dist(self, ax, ay, bx, by):
        lax, lay = self.to_local(ax, ay)
        lbx, lby = self.to_local(bx, by)
        return _seg_rect_dist(lax, lay, lbx, lby, self.x0, self.y0,
                              self.x1, self.y1)

    def local_corners(self):
        return ((self.x0, self.y0), (self.x1, self.y0),
                (self.x1, self.y1), (self.x0, self.y1))

    def world_corners(self):
        out = []
        for lx, ly in self.local_corners():
            out.append((self.cx + lx * self.ct - ly * self.st,
                        self.cy + lx * self.st + ly * self.ct))
        return tuple(out)

    def world_bbox(self):
        cs = self.world_corners()
        xs = [c[0] for c in cs]
        ys = [c[1] for c in cs]
        return min(xs), min(ys), max(xs), max(ys)


def _cheb(ax, ay, bx, by):
    return max(abs(ax - bx), abs(ay - by))


def _polys_overlap(pa, pb):
    """SAT overlap test for two convex polygons (strict overlap)."""
    for poly in (pa, pb):
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            ax, ay = -(y1 - y0), (x1 - x0)
            norm = math.hypot(ax, ay)
            if norm < EPS:
                continue
            ax /= norm
            ay /= norm
            amin = amax = pa[0][0] * ax + pa[0][1] * ay
            for px, py in pa[1:]:
                v = px * ax + py * ay
                if v < amin:
                    amin = v
                elif v > amax:
                    amax = v
            bmin = bmax = pb[0][0] * ax + pb[0][1] * ay
            for px, py in pb[1:]:
                v = px * ax + py * ay
                if v < bmin:
                    bmin = v
                elif v > bmax:
                    bmax = v
            if amax <= bmin + 1e-12 or bmax <= amin + 1e-12:
                return False
    return True


class GeneratedApproach:
    MAX_D = 0.05
    MAX_DTHETA = 0.19634954

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        try:
            self.MAX_D = float(min(action_space.high[0], action_space.high[1]))
            self.MAX_DTHETA = float(action_space.high[2])
        except Exception:
            pass
        self.radius = 0.103
        self.gcen = 0.105     # distance from base centre to gripper centre
        self.ghw = 0.005      # gripper half-size along the arm
        self.ghh = 0.035      # gripper half-size across the arm
        self.margin = 0.003
        self.exec_margin = 0.002
        self.gdir = 1.0
        self.goal_th = 0.0
        self.hard = False
        self.sched = None
        self.tix = 1
        self._hard_th = None
        self._hard_idx = -1
        self._hard_key = None
        self._hard_bad = set()
        self._t0 = time.monotonic()
        self.path = []
        self.idx = 1
        self._last_pos = None
        self._stuck = 0
        self._replans = 0
        self._rects = []
        self._bounds = (0.0, 0.0, 2.5, 2.5)
        self._goal_rect = None
        self._robot_obj = None

    # ------------------------------------------------------------ parsing
    def _parse(self, state):
        robot = None
        target = None
        rects = []
        for name in sorted(state.get_object_names()):
            obj = state.get_object_from_name(name)
            tnames = {t.name for t in obj.type.get_ancestors()}
            if "crv_robot" in tnames:
                robot = obj
            elif "target_region" in tnames:
                target = obj
            elif "rectangle" in tnames:
                rects.append(obj)
        self._robot_obj = robot
        g = state.get
        rx, ry = float(g(robot, "x")), float(g(robot, "y"))
        try:
            br = float(g(robot, "base_radius"))
        except Exception:
            br = 0.1
        try:
            gw = float(g(robot, "gripper_width"))
            gh = float(g(robot, "gripper_height"))
            aj = float(g(robot, "arm_joint"))
        except Exception:
            gw, gh, aj = 0.01, 0.07, 0.1
        self.radius = br
        self.gcen = aj + gw / 2.0
        self.ghw = gw / 2.0
        self.ghh = gh / 2.0
        self._rects = [
            _Rect(float(g(o, "x")), float(g(o, "y")), float(g(o, "theta")),
                  float(g(o, "width")), float(g(o, "height")))
            for o in rects
        ]
        trect = _Rect(float(g(target, "x")), float(g(target, "y")),
                      float(g(target, "theta")), float(g(target, "width")),
                      float(g(target, "height")))
        gb = trect.world_bbox()
        self._goal_rect = gb
        xmax = max(2.5, gb[2], rx + br)
        ymax = max(2.5, gb[3], ry + br)
        self._bounds = (0.0, 0.0, xmax, ymax)
        return (rx, ry, float(g(robot, "theta")))

    # ------------------------------------------------------------ geometry
    def _gripper_poly(self, px, py, th, m):
        c, s = math.cos(th), math.sin(th)
        gx = px + self.gcen * c
        gy = py + self.gcen * s
        a = self.ghw + m
        b = self.ghh + m
        return ((gx + a * c - b * s, gy + a * s + b * c),
                (gx + a * c + b * s, gy + a * s - b * c),
                (gx - a * c + b * s, gy - a * s - b * c),
                (gx - a * c - b * s, gy - a * s + b * c))

    def _pose_free(self, px, py, th, m=None):
        """Exact-ish free test for the robot at (px, py, th) with margin m."""
        if m is None:
            m = self.margin
        r = self.radius + m
        wx0, wy0, wx1, wy1 = self._bounds
        if px < wx0 + r or px > wx1 - r or py < wy0 + r or py > wy1 - r:
            return False
        for rc in self._rects:
            if rc.pt_dist(px, py) < r:
                return False
        poly = self._gripper_poly(px, py, th, m)
        for gx, gy in poly:
            if gx < wx0 or gx > wx1 or gy < wy0 or gy > wy1:
                return False
        for rc in self._rects:
            bx0, by0, bx1, by1 = rc.bbox
            skip = True
            for gx, gy in poly:
                if bx0 <= gx <= bx1 and by0 <= gy <= by1:
                    skip = False
                    break
            if skip:
                pminx = min(p[0] for p in poly)
                pmaxx = max(p[0] for p in poly)
                pminy = min(p[1] for p in poly)
                pmaxy = max(p[1] for p in poly)
                if pmaxx <= bx0 or pminx >= bx1 or pmaxy <= by0 or pminy >= by1:
                    continue
            if rc.axis_aligned:
                rpoly = ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1))
            else:
                rpoly = rc.world_corners()
            if _polys_overlap(poly, rpoly):
                return False
        return True

    def _all_head_free(self, px, py, m=0.001):
        """True if the pose is free for *any* heading (inflated disc)."""
        r = math.hypot(self.gcen + self.ghw, self.ghh) + m
        wx0, wy0, wx1, wy1 = self._bounds
        if px < wx0 + r or px > wx1 - r or py < wy0 + r or py > wy1 - r:
            return False
        for rc in self._rects:
            if rc.pt_dist(px, py) < r:
                return False
        return True

    def _motion_free(self, x0, y0, t0, x1, y1, t1, m=None):
        d = max(math.hypot(x1 - x0, y1 - y0), abs(t1 - t0) * self.gcen * 1.6)
        n = max(1, int(d / 0.012) + 1)
        for i in range(n + 1):
            u = i / n
            if not self._pose_free(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u,
                                   t0 + (t1 - t0) * u, m):
                return False
        return True

    # -------------------------------------------- planning-model freeness
    def _plan_free_pt(self, px, py, r, th):
        return self._pose_free(px, py, th, r - self.radius)

    def _plan_free_seg(self, ax, ay, bx, by, r, th):
        x0, y0, x1, y1 = self._bounds
        if min(ax, bx) < x0 + r or max(ax, bx) > x1 - r:
            return False
        if min(ay, by) < y0 + r or max(ay, by) > y1 - r:
            return False
        for rc in self._rects:
            if rc.seg_dist(ax, ay, bx, by) < r:
                return False
        d = math.hypot(bx - ax, by - ay)
        n = max(1, int(d / 0.012) + 1)
        m = r - self.radius
        for i in range(n + 1):
            u = i / n
            if not self._pose_free(ax + (bx - ax) * u, ay + (by - ay) * u,
                                   th, m):
                return False
        return True

    def _cardinal_ok(self, th):
        if not all(rc.axis_aligned for rc in self._rects):
            return False
        return min(abs(math.cos(th)), abs(math.sin(th))) < 1e-9

    def _mask(self, X, Y, r, th):
        """Vectorised free test; X is (n,1), Y is (1,m) -> bool (n,m)."""
        m = r - self.radius
        wx0, wy0, wx1, wy1 = self._bounds
        ct, st = math.cos(th), math.sin(th)
        ok = ((X >= wx0 + r) & (X <= wx1 - r)) & ((Y >= wy0 + r)
                                                  & (Y <= wy1 - r))
        gcx = X + self.gcen * ct
        gcy = Y + self.gcen * st
        ha = self.ghw + m
        hb = self.ghh + m
        if abs(ct) > abs(st):
            hx, hy = ha, hb
        else:
            hx, hy = hb, ha
        gx0, gx1 = gcx - hx, gcx + hx
        gy0, gy1 = gcy - hy, gcy + hy
        ok = ok & (gx0 >= wx0) & (gx1 <= wx1) & (gy0 >= wy0) & (gy1 <= wy1)
        r2 = r * r
        for rc in self._rects:
            bx0, by0, bx1, by1 = rc.bbox
            dx = np.maximum(np.maximum(bx0 - X, X - bx1), 0.0)
            dy = np.maximum(np.maximum(by0 - Y, Y - by1), 0.0)
            ok = ok & ((dx * dx + dy * dy) >= r2)
            ok = ok & ~((gx1 > bx0) & (gx0 < bx1) & (gy1 > by0) & (gy0 < by1))
        return np.broadcast_to(ok, (X.shape[0], Y.shape[1])).copy()

    def _seg_mask(self, XA, XB, YA, YB, r, th, nsamp):
        out = None
        for i in range(nsamp + 1):
            t = i / nsamp
            X = XA + (XB - XA) * t
            Y = YA + (YB - YA) * t
            mk = self._mask(X, Y, r, th)
            out = mk if out is None else (out & mk)
        return out

    def _grids(self, xs, ys, r, th):
        X = np.asarray(xs, dtype=np.float64)[:, None]
        Y = np.asarray(ys, dtype=np.float64)[None, :]
        free = self._mask(X, Y, r, th)
        ns = 8
        XA, XB = X[:-1], X[1:]
        YA, YB = Y[:, :-1], Y[:, 1:]
        eh = self._seg_mask(XA, XB, Y, Y, r, th, ns)
        ev = self._seg_mask(X, X, YA, YB, r, th, ns)
        ed1 = self._seg_mask(XA, XB, YA, YB, r, th, ns)
        ed2 = self._seg_mask(XA, XB, YB, YA, r, th, ns)
        return free, eh, ev, ed1, ed2

    # ------------------------------------------------------------ lattice
    def _lattice(self, start, r):
        wx0, wy0, wx1, wy1 = self._bounds
        gx0, gy0, gx1, gy1 = self._goal_rect
        eps = 0.006
        xs = set()
        ys = set()
        for lo, hi, acc in ((wx0, wx1, xs), (wy0, wy1, ys)):
            v = lo + r
            n = 0
            while v <= hi - r + 1e-9 and n < 400:
                acc.add(round(v, 6))
                v += 0.05
                n += 1
            acc.add(round(hi - r, 6))
            acc.add(round(lo + r, 6))
        xedges = [wx0, wx1]
        yedges = [wy0, wy1]
        for rc in self._rects:
            bx0, by0, bx1, by1 = rc.bbox
            xedges += [bx0, bx1]
            yedges += [by0, by1]
            for c in (bx0 - r - self.gcen - self.ghw - eps, bx0 - r - eps,
                      bx1 + r + eps, bx1 + r + self.gcen + self.ghw + eps):
                if wx0 + r <= c <= wx1 - r:
                    xs.add(round(c, 6))
            for c in (by0 - r - eps, by1 + r + eps):
                if wy0 + r <= c <= wy1 - r:
                    ys.add(round(c, 6))
        for edges, acc, lo, hi in ((xedges, xs, wx0 + r, wx1 - r),
                                   (yedges, ys, wy0 + r, wy1 - r)):
            e = sorted(set(round(z, 6) for z in edges))
            for i in range(len(e) - 1):
                a, b = e[i], e[i + 1]
                if b - a > 2 * r:
                    for frac in (0.5, 0.3, 0.7):
                        m2 = a + frac * (b - a)
                        if lo <= m2 <= hi and min(m2 - a, b - m2) > r:
                            acc.add(round(m2, 6))
        xs.add(round(start[0], 6))
        ys.add(round(start[1], 6))
        for c in (gx0 + eps, gx1 - eps, 0.5 * (gx0 + gx1)):
            if wx0 + r <= c <= wx1 - r:
                xs.add(round(c, 6))
        for c in (gy0 + eps, gy1 - eps, 0.5 * (gy0 + gy1)):
            if wy0 + r <= c <= wy1 - r:
                ys.add(round(c, 6))
        return sorted(xs), sorted(ys)

    # ------------------------------------------------------------ planning
    def _plan(self, start, r, th):
        xs, ys = self._lattice(start, r)
        nx, ny = len(xs), len(ys)
        gx0, gy0, gx1, gy1 = self._goal_rect
        eps = 0.004

        if self._cardinal_ok(th):
            free_a, eh, ev, ed1, ed2 = self._grids(xs, ys, r, th)
            free = free_a.tolist()
            eh = eh.tolist()
            ev = ev.tolist()
            ed1 = ed1.tolist()
            ed2 = ed2.tolist()
            fast = True
        else:
            free = [[self._plan_free_pt(xs[i], ys[j], r, th)
                     for j in range(ny)] for i in range(nx)]
            eh = ev = ed1 = ed2 = None
            fast = False
        si = min(range(nx), key=lambda i: abs(xs[i] - start[0]))
        sj = min(range(ny), key=lambda j: abs(ys[j] - start[1]))
        free[si][sj] = True

        def is_goal(i, j):
            return (gx0 + eps <= xs[i] <= gx1 - eps
                    and gy0 + eps <= ys[j] <= gy1 - eps)

        def h(i, j):
            x, y = xs[i], ys[j]
            return max(max(gx0 + eps - x, 0.0, x - (gx1 - eps)),
                       max(gy0 + eps - y, 0.0, y - (gy1 - eps)))

        def edge_ok(i, j, di, dj):
            if not fast:
                return self._plan_free_seg(xs[i], ys[j], xs[i + di],
                                           ys[j + dj], r, th)
            if dj == 0:
                return eh[i][j] if di > 0 else eh[i - 1][j]
            if di == 0:
                return ev[i][j] if dj > 0 else ev[i][j - 1]
            if di > 0 and dj > 0:
                return ed1[i][j]
            if di < 0 and dj < 0:
                return ed1[i - 1][j - 1]
            if di > 0 and dj < 0:
                return ed2[i][j - 1]
            return ed2[i - 1][j]

        start_node = (si, sj)
        gscore = {start_node: 0.0}
        came = {}
        pq = [(h(si, sj), 0.0, start_node)]
        goal_found = None
        while pq:
            f, g, node = heapq.heappop(pq)
            if g > gscore.get(node, 1e18) + 1e-12:
                continue
            i, j = node
            if is_goal(i, j):
                goal_found = node
                break
            for di in (-1, 0, 1):
                ni = i + di
                if ni < 0 or ni >= nx:
                    continue
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    nj = j + dj
                    if nj < 0 or nj >= ny:
                        continue
                    if not free[ni][nj]:
                        continue
                    ng = g + _cheb(xs[i], ys[j], xs[ni], ys[nj])
                    nn = (ni, nj)
                    if ng < gscore.get(nn, 1e18) - 1e-12:
                        if not edge_ok(i, j, di, dj):
                            continue
                        gscore[nn] = ng
                        came[nn] = node
                        heapq.heappush(pq, (ng + h(ni, nj), ng, nn))
        if goal_found is None:
            return None
        pts = []
        node = goal_found
        while node != start_node:
            pts.append((xs[node[0]], ys[node[1]]))
            node = came[node]
        pts.append((xs[si], ys[sj]))
        pts.reverse()
        pts[0] = (start[0], start[1])
        pts = self._shortcut(pts, r, th)
        pts = self._refine(pts, r, th)
        return self._shortcut(pts, r, th)

    def _refine(self, pts, r, th):
        """Coordinate-descent smoothing of interior waypoints."""
        if len(pts) < 3:
            return pts
        gx0, gy0, gx1, gy1 = self._goal_rect
        eps = 0.004
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1))
        for step in (0.04, 0.02, 0.01, 0.005, 0.0025):
            improved = True
            rounds = 0
            while improved and rounds < 6:
                improved = False
                rounds += 1
                for i in range(1, len(pts)):
                    last = (i == len(pts) - 1)
                    px, py = pts[i]
                    ax, ay = pts[i - 1]
                    if last:
                        base = _cheb(ax, ay, px, py)
                    else:
                        bx, by = pts[i + 1]
                        base = (_cheb(ax, ay, px, py)
                                + _cheb(px, py, bx, by))
                    for ux, uy in dirs:
                        nx2 = px + ux * step
                        ny2 = py + uy * step
                        if last:
                            if not (gx0 + eps <= nx2 <= gx1 - eps
                                    and gy0 + eps <= ny2 <= gy1 - eps):
                                continue
                            cost = _cheb(ax, ay, nx2, ny2)
                        else:
                            cost = (_cheb(ax, ay, nx2, ny2)
                                    + _cheb(nx2, ny2, bx, by))
                        if cost >= base - 1e-9:
                            continue
                        if not self._plan_free_seg(ax, ay, nx2, ny2, r, th):
                            continue
                        if not last and not self._plan_free_seg(
                                nx2, ny2, bx, by, r, th):
                            continue
                        pts[i] = (nx2, ny2)
                        px, py = nx2, ny2
                        base = cost
                        improved = True
        return pts

    def _shortcut(self, pts, r, th):
        for _ in range(3):
            out = [pts[0]]
            i = 0
            n = len(pts)
            while i < n - 1:
                j = n - 1
                while j > i + 1:
                    cur = sum(_cheb(pts[k][0], pts[k][1],
                                    pts[k + 1][0], pts[k + 1][1])
                              for k in range(i, j))
                    direct = _cheb(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
                    if direct <= cur + 1e-9 and self._plan_free_seg(
                            pts[i][0], pts[i][1], pts[j][0], pts[j][1], r, th):
                        break
                    j -= 1
                out.append(pts[j])
                i = j
            if len(out) == len(pts):
                return out
            pts = out
        return pts

    def _make_plan(self, start, th0, allow_vertical=True):
        base = self.radius
        ladder = (0.008, 0.005, 0.003, 0.0015, 0.0005, 0.0)
        prim = 0.0 if math.cos(th0) >= -1e-9 else math.pi
        sec = math.pi - prim
        results = []

        def try_heading(th):
            for m in ladder:
                p = self._plan(start, base + m, th)
                if p:
                    results.append((th, m, p))
                    return True
            return False

        for th in (prim, sec):
            try_heading(th)
        if allow_vertical and not [r for r in results if r[1] >= 0.0015]:
            for th in (math.pi / 2, -math.pi / 2):
                try_heading(th)
        if not results:
            return None

        def score(item):
            th, m, p = item
            L = sum(_cheb(p[i][0], p[i][1], p[i + 1][0], p[i + 1][1])
                    for i in range(len(p) - 1)) / self.MAX_D
            rot = abs((th - th0 + math.pi) % (2 * math.pi) - math.pi)
            return L + 0.2 * (rot / self.MAX_DTHETA) + (0.0 if m >= 0.0015
                                                        else 1.0)

        th, m, p = min(results, key=score)
        self.goal_th = th
        self.gdir = 1.0 if math.cos(th) >= 0 else -1.0
        self.margin = m
        self.exec_margin = min(0.002, m)
        return p


    # ------------------------------------------------- heading scheduling
    def _schedule(self, pts, th0):
        """Time-indexed (x, y, theta) trajectory along *pts*.

        Discretises headings at exactly one rotation step apart and runs a
        Dijkstra search over (sample, heading) so the gripper can turn to
        fit through parts of the route that no single heading clears.
        """
        step = self.MAX_D
        samples = [pts[0]]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            d = _cheb(a[0], a[1], b[0], b[1])
            n = max(1, int(math.ceil(d / step - 1e-9)))
            for k in range(1, n + 1):
                samples.append((a[0] + (b[0] - a[0]) * k / n,
                                a[1] + (b[1] - a[1]) * k / n))
        ns = len(samples)
        if ns > 400:
            return None
        nh = max(4, int(round(2 * math.pi / self.MAX_DTHETA)))
        dth = 2 * math.pi / nh
        base = th0 - dth * round(th0 / dth)
        heads = [base + i * dth for i in range(nh)]
        freeh = [[self._pose_free(samples[k][0], samples[k][1], heads[i], 0.0)
                  for i in range(nh)] for k in range(ns)]
        cache = {}

        def edge(k, i, j):
            key = (k, i, j)
            v = cache.get(key)
            if v is None:
                a, b = samples[k], samples[k + 1]
                v = self._motion_free(a[0], a[1], heads[i], b[0], b[1],
                                      heads[j], 0.0)
                cache[key] = v
            return v

        i0 = int(round((th0 - base) / dth)) % nh
        if not freeh[0][i0]:
            return None
        dist = {(0, i0): 0}
        pq = [(0, 0, i0)]
        goal = None
        prev = {}
        while pq:
            g, k, i = heapq.heappop(pq)
            if g > dist.get((k, i), 1 << 30):
                continue
            if k == ns - 1:
                goal = (k, i)
                break
            for di in (-1, 0, 1):
                j = (i + di) % nh
                if not freeh[k + 1][j]:
                    continue
                if not edge(k, i, j):
                    continue
                nk = (k + 1, j)
                if g + 1 < dist.get(nk, 1 << 30):
                    dist[nk] = g + 1
                    prev[nk] = (k, i)
                    heapq.heappush(pq, (g + 1, k + 1, j))
            for di in (-1, 1):
                j = (i + di) % nh
                if not freeh[k][j]:
                    continue
                if not self._motion_free(samples[k][0], samples[k][1],
                                         heads[i], samples[k][0],
                                         samples[k][1], heads[j], 0.0):
                    continue
                nk = (k, j)
                if g + 1 < dist.get(nk, 1 << 30):
                    dist[nk] = g + 1
                    prev[nk] = (k, i)
                    heapq.heappush(pq, (g + 1, k, j))
        if goal is None:
            return None
        out = []
        node = goal
        while node in prev:
            k, i = node
            out.append((samples[k][0], samples[k][1], heads[i]))
            node = prev[node]
        out.append((samples[0][0], samples[0][1], heads[i0]))
        out.reverse()
        return out

    # ---------------------------------------------------------------- API
    def reset(self, state, info=None):
        self._t0 = time.monotonic()
        self.margin = 0.003
        self.hard = False
        self.sched = None
        self.tix = 1
        rx, ry, th0 = self._parse(state)
        p = self._make_plan((rx, ry), th0)
        if p is None:
            # No single-heading plan exists: fall back to a disc-only plan
            # and let the executor pick a heading that fits locally.
            self.hard = True
            gcen, ghw, ghh = self.gcen, self.ghw, self.ghh
            self.gcen, self.ghw, self.ghh = 0.0, 0.0, 0.0
            try:
                for m in (0.005, 0.003, 0.0015, 0.0005, 0.0):
                    p = self._plan((rx, ry), self.radius + m, 0.0)
                    if p:
                        self.margin = m
                        self.exec_margin = 0.0
                        break
            finally:
                self.gcen, self.ghw, self.ghh = gcen, ghw, ghh
            if p is not None:
                sched = self._schedule(p, th0)
                if sched is not None:
                    self.sched = sched
                    self.tix = 1
            self.goal_th = th0
            self.gdir = 1.0 if math.cos(th0) >= 0 else -1.0
        if p is None:
            gb = self._goal_rect
            self.goal_th = 0.0 if math.cos(th0) >= 0 else math.pi
            self.gdir = 1.0 if math.cos(th0) >= 0 else -1.0
            p = [(rx, ry), (0.5 * (gb[0] + gb[2]), 0.5 * (gb[1] + gb[3]))]
        self.path = p
        self.idx = 1
        self._last_pos = None
        self._stuck = 0
        self._replans = 0
        self._hard_th = None
        self._hard_idx = -1
        self._hard_key = None
        self._hard_bad = set()
        return None

    def _robot(self, state):
        if self._robot_obj is not None:
            return self._robot_obj
        for name in sorted(state.get_object_names()):
            obj = state.get_object_from_name(name)
            if "crv_robot" in {t.name for t in obj.type.get_ancestors()}:
                self._robot_obj = obj
                return obj
        return state.get_object_from_name("robot")

    def get_action(self, state):
        robot = self._robot(state)
        x = float(state.get(robot, "x"))
        y = float(state.get(robot, "y"))
        theta = float(state.get(robot, "theta"))

        if self._last_pos is not None:
            if _cheb(x, y, self._last_pos[0], self._last_pos[1]) < 1e-9:
                self._stuck += 1
            else:
                self._stuck = 0
        self._last_pos = (x, y, theta)
        elapsed = time.monotonic() - self._t0

        if (self._stuck >= 3 and self._replans < 10 and elapsed < 20.0
                and not self.hard):
            self._stuck = 0
            self._replans += 1
            p = self._make_plan((x, y), self.goal_th,
                                allow_vertical=(self._replans < 3))
            if p is not None:
                self.path = p
                self.idx = 1

        if (not self.hard and self._stuck >= 8 and self._replans >= 5
                and elapsed < 30.0):
            # give up on a single fixed heading and switch strategies
            self.hard = True
            self._stuck = 0
            self._resched(x, y, theta)

        if (self.hard and self._stuck >= 4 and self._replans < 14
                and elapsed < 30.0):
            self._stuck = 0
            self._replans += 1
            self._resched(x, y, theta)

        path = self.path
        budget = self.MAX_D
        cx, cy = x, y
        k = min(self.idx, len(path) - 1)
        tight = self.margin < 0.0015
        while k < len(path) and budget > 1e-12:
            wx, wy = path[k]
            d = _cheb(cx, cy, wx, wy)
            if d <= budget + 1e-12:
                budget -= d
                cx, cy = wx, wy
                k += 1
            else:
                t = budget / d
                cx += (wx - cx) * t
                cy += (wy - cy) * t
                budget = 0.0
            if tight:
                break
        self.idx = min(k, len(path) - 1)

        goal_th = self.goal_th
        err = (goal_th - theta + math.pi) % (2 * math.pi) - math.pi
        dth_full = float(np.clip(err, -self.MAX_DTHETA, self.MAX_DTHETA))

        cands = [(cx, cy)]
        wx, wy = path[self.idx]
        d = _cheb(x, y, wx, wy)
        if d > 1e-9:
            t = min(1.0, self.MAX_D / d)
            for f in (1.0, 0.7, 0.45, 0.25):
                cands.append((x + (wx - x) * t * f, y + (wy - y) * t * f))

        chosen = None
        if self.hard:
            if self.sched is not None:
                chosen = self._sched_choice(x, y, theta)
            if chosen is None:
                chosen = self._hard_choice(x, y, theta, cands, cx, cy)
            return self._act(x, y, chosen)

        em = self.exec_margin
        turning = abs(err) > 0.02
        for dth in (dth_full, 0.55 * dth_full, 0.2 * dth_full, 0.0):
            for c in cands:
                if turning and not self._all_head_free(c[0], c[1]):
                    continue
                if self._motion_free(x, y, theta, c[0], c[1], theta + dth, em):
                    chosen = (c[0], c[1], dth)
                    break
            if chosen is not None:
                break
            if abs(dth) < 1e-12:
                break
        if chosen is None and turning:
            # rotate on the spot rather than wedging in with a bad heading
            for dth in (dth_full, 0.5 * dth_full):
                if self._motion_free(x, y, theta, x, y, theta + dth, em):
                    chosen = (x, y, dth)
                    break
        if chosen is None:
            for dth in (dth_full, 0.55 * dth_full, 0.2 * dth_full, 0.0):
                for c in cands:
                    if self._motion_free(x, y, theta, c[0], c[1],
                                         theta + dth, em):
                        chosen = (c[0], c[1], dth)
                        break
                if chosen is not None:
                    break
                if abs(dth) < 1e-12:
                    break

        if chosen is None:
            wider = list(cands)
            for m in (0.0008, 0.0, -0.0015):
                for dth in (dth_full, 0.5 * dth_full, 0.0, -0.5 * dth_full):
                    for c in wider:
                        if (abs(dth) < 1e-12 and abs(c[0] - x) < 1e-12
                                and abs(c[1] - y) < 1e-12):
                            continue
                        if self._motion_free(x, y, theta, c[0], c[1],
                                             theta + dth, m):
                            chosen = (c[0], c[1], dth)
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break

        if chosen is None and elapsed < 40.0:
            gx, gy = path[min(self.idx, len(path) - 1)]
            best = None
            for m in (0.0008, -0.0015):
                for i in range(16):
                    ang = 2 * math.pi * i / 16.0
                    ux, uy = math.cos(ang), math.sin(ang)
                    s = self.MAX_D / max(abs(ux), abs(uy))
                    for mag in (s, 0.5 * s):
                        nx2, ny2 = x + ux * mag, y + uy * mag
                        for dth in (dth_full, 0.0):
                            if self._motion_free(x, y, theta, nx2, ny2,
                                                 theta + dth, m):
                                sc = _cheb(nx2, ny2, gx, gy)
                                if best is None or sc < best[0]:
                                    best = (sc, nx2, ny2, dth)
                                break
                if best is not None:
                    break
            if best is not None:
                chosen = (best[1], best[2], best[3])

        if chosen is None:
            chosen = (x, y, dth_full if abs(dth_full) > 1e-12
                      else self.MAX_DTHETA)
        return self._act(x, y, chosen)

    def _act(self, x, y, chosen):
        dx = float(np.clip(chosen[0] - x, -self.MAX_D, self.MAX_D))
        dy = float(np.clip(chosen[1] - y, -self.MAX_D, self.MAX_D))
        dth = float(np.clip(chosen[2], -self.MAX_DTHETA, self.MAX_DTHETA))
        return np.array([dx, dy, dth, -0.1, 0.0], dtype=np.float32)

    def _clearance(self, px, py, th):
        """Distance from the gripper box to the nearest obstacle/wall."""
        poly = self._gripper_poly(px, py, th, 0.0)
        gx = sum(p[0] for p in poly) / 4.0
        gy = sum(p[1] for p in poly) / 4.0
        rad = math.hypot(self.ghw, self.ghh)
        wx0, wy0, wx1, wy1 = self._bounds
        best = min(gx - wx0, wx1 - gx, gy - wy0, wy1 - gy) - rad
        for rc in self._rects:
            d = rc.pt_dist(gx, gy) - rad
            if d < best:
                best = d
        return best

    NH = 24

    def _hard_choice(self, x, y, theta, cands, cx, cy):
        """Greedy (dx, dy, dtheta) when no single heading works everywhere.

        Searches an absolute grid of headings for one that admits progress
        toward the carrot, commits to it while rotating, and blacklists
        headings that turn out not to help at this position.
        """
        base = _cheb(x, y, cx, cy)
        key = (round(x, 4), round(y, 4))
        if key != self._hard_key:
            self._hard_key = key
            self._hard_bad = set()
        pos = list(cands)
        if base > 1e-9:
            for f in (0.15, 0.06):
                pos.append((x + (cx - x) * f, y + (cy - y) * f))

        def best_move(th_new):
            for c in pos:
                if _cheb(c[0], c[1], cx, cy) >= base - 1e-9:
                    continue
                if self._motion_free(x, y, theta, c[0], c[1], th_new, 0.0):
                    return c
            return None

        c = best_move(theta)
        if c is not None:
            self._hard_th = None
            return (c[0], c[1], 0.0)

        if self._hard_th is not None:
            err = (self._hard_th - theta + math.pi) % (2 * math.pi) - math.pi
            dth = float(np.clip(err, -self.MAX_DTHETA, self.MAX_DTHETA))
            c = best_move(theta + dth)
            if c is not None:
                return (c[0], c[1], dth)
            if abs(err) <= self.MAX_DTHETA + 1e-9:
                self._hard_bad.add(self._hard_idx)
                self._hard_th = None
            elif self._motion_free(x, y, theta, x, y, theta + dth, 0.0):
                return (x, y, dth)
            else:
                self._hard_bad.add(self._hard_idx)
                self._hard_th = None

        best = None
        for c in pos:
            prog = base - _cheb(c[0], c[1], cx, cy)
            if prog <= 1e-9:
                continue
            for i in range(self.NH):
                if i in self._hard_bad:
                    continue
                th_c = 2.0 * math.pi * i / self.NH
                rot = abs((th_c - theta + math.pi) % (2 * math.pi) - math.pi)
                sc = prog - 0.004 * (rot / self.MAX_DTHETA)
                if best is not None and sc <= best[0]:
                    continue
                if not self._motion_free(x, y, th_c, c[0], c[1], th_c, 0.0):
                    continue
                best = (sc, c[0], c[1], th_c, i)
        if best is not None:
            self._hard_th = best[3]
            self._hard_idx = best[4]
            err = (best[3] - theta + math.pi) % (2 * math.pi) - math.pi
            dth = float(np.clip(err, -self.MAX_DTHETA, self.MAX_DTHETA))
            c = best_move(theta + dth)
            if c is not None:
                return (c[0], c[1], dth)
            if self._motion_free(x, y, theta, x, y, theta + dth, 0.0):
                return (x, y, dth)
            self._hard_bad.add(best[4])
            self._hard_th = None

        # cannot progress: sidestep, keeping the heading flexible
        for m in (0.0, -0.0015):
            best = None
            for mag in (self.MAX_D, 0.5 * self.MAX_D):
                for i in range(16):
                    ang = 2 * math.pi * i / 16.0
                    nx2 = x + math.cos(ang) * mag
                    ny2 = y + math.sin(ang) * mag
                    for dth in (0.0, self.MAX_DTHETA, -self.MAX_DTHETA):
                        if self._motion_free(x, y, theta, nx2, ny2,
                                             theta + dth, m):
                            sc = _cheb(nx2, ny2, cx, cy)
                            if best is None or sc < best[0]:
                                best = (sc, nx2, ny2, dth)
                            break
            if best is not None:
                return (best[1], best[2], best[3])
            for dth in (self.MAX_DTHETA, -self.MAX_DTHETA):
                if self._motion_free(x, y, theta, x, y, theta + dth, m):
                    return (x, y, dth)
        return (x, y, self.MAX_DTHETA)

    def _sched_choice(self, x, y, theta):
        """Follow the (position, heading) schedule computed at reset."""
        sched = self.sched
        n = len(sched)
        while self.tix < n - 1:
            tx, ty, tth = sched[self.tix]
            e = abs((tth - theta + math.pi) % (2 * math.pi) - math.pi)
            if _cheb(x, y, tx, ty) < 2e-4 and e < 2e-3:
                self.tix += 1
            else:
                break
        tx, ty, tth = sched[self.tix]
        err = (tth - theta + math.pi) % (2 * math.pi) - math.pi
        dth = float(np.clip(err, -self.MAX_DTHETA, self.MAX_DTHETA))
        d = _cheb(x, y, tx, ty)
        if d > self.MAX_D:
            t = self.MAX_D / d
            nx2, ny2 = x + (tx - x) * t, y + (ty - y) * t
        else:
            nx2, ny2 = tx, ty
        if (abs(nx2 - x) < 1e-6 and abs(ny2 - y) < 1e-6
                and abs(dth) < 1e-6 and self.tix < n - 1):
            self.tix += 1
            tx, ty, tth = sched[self.tix]
            err = (tth - theta + math.pi) % (2 * math.pi) - math.pi
            dth = float(np.clip(err, -self.MAX_DTHETA, self.MAX_DTHETA))
            d = _cheb(x, y, tx, ty)
            if d > self.MAX_D:
                t = self.MAX_D / d
                nx2, ny2 = x + (tx - x) * t, y + (ty - y) * t
            else:
                nx2, ny2 = tx, ty
        if self._motion_free(x, y, theta, nx2, ny2, theta + dth, 0.0):
            return (nx2, ny2, dth)
        for m in (0.0, -0.0015):
            for f in (0.6, 0.3, 0.0):
                for dd in (dth, 0.0):
                    if f == 0.0 and abs(dd) < 1e-12:
                        continue
                    px = x + (nx2 - x) * f
                    py = y + (ny2 - y) * f
                    if self._motion_free(x, y, theta, px, py, theta + dd, m):
                        return (px, py, dd)
        return None

    def _resched(self, x, y, theta):
        """Recompute the disc-only path and heading schedule from here."""
        gcen, ghw, ghh = self.gcen, self.ghw, self.ghh
        self.gcen, self.ghw, self.ghh = 0.0, 0.0, 0.0
        p = None
        try:
            for m in (0.003, 0.0015, 0.0005, 0.0):
                p = self._plan((x, y), self.radius + m, 0.0)
                if p:
                    break
        finally:
            self.gcen, self.ghw, self.ghh = gcen, ghw, ghh
        if not p:
            return
        sched = self._schedule(p, theta)
        if sched is not None:
            self.sched = sched
            self.tix = 1
            self.path = p
            self.idx = 1
        else:
            self.sched = None
            self.path = p
            self.idx = 1
