"""Motion2D approach.

The robot keeps its arm retracted.  We plan in (x, y, mode) where mode is one
of four axis-aligned arm orientations (right, up, left, down).  For each mode
the footprint (base circle + gripper rectangle) is checked exactly; switching
modes is only allowed where the full rotation disc is free.  A* uses the
Chebyshev metric (matching the per-step dx/dy limits) on a non-uniform grid
that includes positions centred in narrow gaps for every mode.  The path is
followed greedily by jumping to the furthest path point reachable in one step.
"""
import heapq
import math

import numpy as np

STEP = 0.0499
DTH = math.pi / 16 * 0.999
EPS = 2e-5
MODES = [0.0, math.pi / 2, math.pi, -math.pi / 2]
SWITCH_COST = 0.1  # per 90 degrees (rotation mostly overlaps translation)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.path = None
        self.idx = 0
        self.last_pos = None
        self.last_moved = False
        self.stuck = 0
        self.total_stuck = 0
        self.rng = np.random.default_rng(0)

    # ---------------- state reading -----------------
    @staticmethod
    def _read(state):
        robot = target = None
        rects = []
        for obj in state:
            t = obj.type.name
            if t == "crv_robot":
                robot = obj
            elif t == "target_region":
                target = obj
            elif t == "rectangle":
                rects.append(obj)
        return robot, target, rects

    @staticmethod
    def _rp(state, o):
        return tuple(float(state.get(o, f)) for f in ("x", "y", "theta", "width", "height"))

    # ---------------- geometry -----------------
    def _setup_geometry(self, state):
        robot, target, rects = self._read(state)
        self.target = self._rp(state, target)
        self.br = float(state.get(robot, "base_radius"))
        aj = float(state.get(robot, "arm_joint"))
        gw = float(state.get(robot, "gripper_width"))
        gh = float(state.get(robot, "gripper_height"))
        self.aj, self.gw, self.gh = aj, gw, gh
        self.reach = max(self.br, math.hypot(aj + gw / 2, gh / 2))
        # per-mode gripper box: (offset x, offset y, half-ext x, half-ext y)
        self.gbox = []
        for a in MODES:
            ox, oy = aj * round(math.cos(a)), aj * round(math.sin(a))
            if abs(math.cos(a)) > 0.5:
                ex, ey = gw / 2, gh / 2
            else:
                ex, ey = gh / 2, gw / 2
            self.gbox.append((ox, oy, ex, ey))
        self.xmin, self.xmax, self.ymin, self.ymax = 0.0, 2.5, 0.0, 2.5
        self.boxes = []
        self.rot = []
        for r in rects:
            x, y, th, w, h = self._rp(state, r)
            if w <= 0 or h <= 0:
                continue
            if abs(_wrap(th)) < 1e-9:
                self.boxes.append((x, y, x + w, y + h))
            elif abs(abs(_wrap(th)) - math.pi) < 1e-9:
                self.boxes.append((x - w, y - h, x, y))
            else:
                self.rot.append((x, y, th, w, h))
        t = 1.0
        self.walls = [(self.xmin - t, self.ymin - t, self.xmin, self.ymax + t),
                      (self.xmax, self.ymin - t, self.xmax + t, self.ymax + t),
                      (self.xmin - t, self.ymin - t, self.xmax + t, self.ymin),
                      (self.xmin - t, self.ymax, self.xmax + t, self.ymax + t)]
        self.allboxes = self.boxes + self.walls

    def _free(self, px, py, mode):
        """Vectorized collision check. mode=-1 means full rotation disc."""
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        ok = np.ones(px.shape, dtype=bool)
        rotating = mode < 0
        R = (self.reach if rotating else self.br) + EPS
        if not rotating:
            ox, oy, ex, ey = self.gbox[mode]
            hx = ex + EPS; hy = ey + EPS
            gcx = px + ox; gcy = py + oy
        for (x0, y0, x1, y1) in self.allboxes:
            dx = np.maximum(np.maximum(x0 - px, px - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - py, py - y1), 0.0)
            ok &= (dx * dx + dy * dy) > R * R
            if not rotating:
                ov = (gcx + hx > x0) & (gcx - hx < x1) & (gcy + hy > y0) & (gcy - hy < y1)
                ok &= ~ov
        for (x, y, th, w, h) in self.rot:
            c, s = math.cos(th), math.sin(th)
            rx, ry = px - x, py - y
            lx = c * rx + s * ry
            ly = -s * rx + c * ry
            dx = np.maximum(np.maximum(-lx, lx - w), 0.0)
            dy = np.maximum(np.maximum(-ly, ly - h), 0.0)
            ok &= (dx * dx + dy * dy) > (self.reach + EPS) ** 2
        return ok

    def _free_pt(self, x, y, mode):
        return bool(self._free(np.array([x]), np.array([y]), mode)[0])

    def _seg_free(self, a, b, mode):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(math.ceil(L / 0.003)) + 1)
        t = np.linspace(0, 1, n)
        return bool(np.all(self._free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, mode)))

    def _target_local(self, px, py):
        tx, ty, th, w, h = self.target
        c, s = math.cos(th), math.sin(th)
        rx, ry = px - tx, py - ty
        return c * rx + s * ry, -s * rx + c * ry, w, h

    # ---------------- planning -----------------
    def _axis_coords(self, lo, hi, faces, start_v, offsets):
        vals = list(np.arange(lo, hi + 1e-9, 0.01))
        faces = sorted(set(faces))
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                d = faces[j] - faces[i]
                if 0 < d < 0.35:
                    mid = 0.5 * (faces[i] + faces[j])
                    for o in offsets:
                        vals.append(mid + o)
        vals.append(start_v)
        vals = np.unique(np.round(np.array(vals), 9))
        vals = vals[(vals >= lo) & (vals <= hi)]
        return vals

    def _plan(self, start, th_now):
        xf = [self.xmin, self.xmax]
        yf = [self.ymin, self.ymax]
        for (x0, y0, x1, y1) in self.boxes:
            xf += [x0, x1]
            yf += [y0, y1]
        offx, offy = {0.0}, {0.0}
        for (ox, oy, ex, ey) in self.gbox:
            xlo = max(self.br, -(ox - ex)); xhi = max(self.br, ox + ex)
            ylo = max(self.br, -(oy - ey)); yhi = max(self.br, oy + ey)
            offx.add(round((xlo - xhi) / 2, 9)); offy.add(round((ylo - yhi) / 2, 9))
        X = self._axis_coords(self.xmin, self.xmax, xf, start[0], offx)
        Y = self._axis_coords(self.ymin, self.ymax, yf, start[1], offy)
        nx, ny = len(X), len(Y)
        GX, GY = np.meshgrid(X, Y, indexing="ij")
        free = [self._free(GX, GY, m) for m in range(4)]
        rfree = self._free(GX, GY, -1)
        lx, ly, w, h = self._target_local(GX, GY)
        mg = 0.003
        intgt = (lx >= mg) & (lx <= w - mg) & (ly >= mg) & (ly <= h - mg)
        anyfree = free[0] | free[1] | free[2] | free[3]
        if not (intgt & anyfree).any():
            intgt = (lx >= 0) & (lx <= w) & (ly >= 0) & (ly <= h)
            for m in range(4):
                free[m] = free[m] | intgt
        si = int(np.argmin(np.abs(X - start[0])))
        sj = int(np.argmin(np.abs(Y - start[1])))
        gp = np.argwhere(intgt)
        gx0, gx1 = X[gp[:, 0]].min(), X[gp[:, 0]].max()
        gy0, gy1 = Y[gp[:, 1]].min(), Y[gp[:, 1]].max()

        Xl = X.tolist(); Yl = Y.tolist()
        freel = [f.tolist() for f in free]
        rfl = rfree.tolist()
        goall = intgt.tolist()

        def hfn(i, j):
            x = Xl[i]; y = Yl[j]
            dx = max(gx0 - x, 0.0, x - gx1)
            dy = max(gy0 - y, 0.0, y - gy1)
            return max(dx, dy)

        g = {}
        parent = {}
        pq = []
        h0 = hfn(si, sj)
        for m in range(4):
            # initial rotation (mostly overlaps translation in open space)
            q = abs(_wrap(MODES[m] - th_now)) / (math.pi / 2)
            c0 = q * SWITCH_COST
            if not rfl[si][sj] and q > 1e-6:
                continue
            g[(m, si, sj)] = c0
            heapq.heappush(pq, (c0 + h0, c0, m, si, sj))
            parent[(m, si, sj)] = None
        if not pq:
            for m in range(4):
                g[(m, si, sj)] = 0.0
                parent[(m, si, sj)] = None
                heapq.heappush(pq, (h0, 0.0, m, si, sj))
        closed = set()
        found = None
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        while pq:
            f, gc, m, i, j = heapq.heappop(pq)
            key = (m, i, j)
            if key in closed:
                continue
            closed.add(key)
            if goall[i][j]:
                found = key
                break
            xi = Xl[i]; yj = Yl[j]
            fm = freel[m]
            for di, dj in nb:
                a = i + di; b = j + dj
                if a < 0 or b < 0 or a >= nx or b >= ny or not fm[a][b]:
                    continue
                if di and dj and not (fm[a][j] and fm[i][b]):
                    continue
                cst = max(abs(Xl[a] - xi), abs(Yl[b] - yj)) + 1e-6
                ng = gc + cst
                k2 = (m, a, b)
                if ng < g.get(k2, 1e18):
                    g[k2] = ng
                    parent[k2] = key
                    heapq.heappush(pq, (ng + hfn(a, b), ng, m, a, b))
            if rfl[i][j]:
                for m2 in range(4):
                    if m2 == m or not freel[m2][i][j]:
                        continue
                    q = 2 if (m2 - m) % 4 == 2 else 1
                    ng = gc + q * SWITCH_COST
                    k2 = (m2, i, j)
                    if ng < g.get(k2, 1e18):
                        g[k2] = ng
                        parent[k2] = key
                        heapq.heappush(pq, (ng + hfn(i, j), ng, m2, i, j))
        if found is None:
            return None
        nodes = [found]
        while parent[nodes[-1]] is not None:
            nodes.append(parent[nodes[-1]])
        nodes.reverse()
        pts = [(Xl[i], Yl[j], m) for (m, i, j) in nodes]
        pts[0] = (start[0], start[1], pts[0][2])
        return self._shortcut(pts)

    def _shortcut(self, pts):
        # split into runs of constant mode, shortcut each run
        out = []
        runs = []
        cur = [pts[0]]
        for p in pts[1:]:
            if p[2] != cur[-1][2]:
                runs.append(cur)
                cur = [p]
            else:
                cur.append(p)
        runs.append(cur)
        for run in runs:
            out.extend(self._shortcut_run(run))
        return out

    def _shortcut_run(self, pts):
        m = pts[0][2]
        out = [pts[0]]
        i = 0
        N = len(pts)
        while i < N - 1:
            lo = i + 1
            k = 1
            hi = None
            while True:
                j = i + 2 * k
                if j >= N - 1:
                    if self._seg_free(pts[i], pts[N - 1], m):
                        lo = N - 1
                    else:
                        hi = N - 1
                    break
                if self._seg_free(pts[i], pts[j], m):
                    lo = j
                    k *= 2
                else:
                    hi = j
                    break
            if hi is not None:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if self._seg_free(pts[i], pts[mid], m):
                        lo = mid
                    else:
                        hi = mid
            out.append(pts[lo])
            i = lo
        return out

    def _set_path(self, path):
        self.idx = 0
        if not path:
            self.path = None
            return
        pts = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            k = max(1, int(math.ceil(L / 0.002)))
            if L == 0:
                pts.append(b)
                continue
            for t in range(1, k + 1):
                pts.append((a[0] + (b[0] - a[0]) * t / k, a[1] + (b[1] - a[1]) * t / k, b[2]))
        self.path = pts

    # ---------------- interface -----------------
    def reset(self, state, info):
        self._setup_geometry(state)
        robot, _, _ = self._read(state)
        start = (float(state.get(robot, "x")), float(state.get(robot, "y")))
        self._set_path(self._plan(start, float(state.get(robot, "theta"))))
        self.last_pos = None
        self.last_moved = False
        self.stuck = 0
        self.total_stuck = 0

    def get_action(self, state):
        robot, _, _ = self._read(state)
        x = float(state.get(robot, "x")); y = float(state.get(robot, "y"))
        th = float(state.get(robot, "theta"))
        act = np.zeros(5, dtype=np.float32)
        if self.last_pos is not None and self.last_moved and abs(self.last_pos[0] - x) < 1e-12 \
                and abs(self.last_pos[1] - y) < 1e-12 and abs(self.last_pos[2] - th) < 1e-12:
            self.stuck += 1
            self.total_stuck += 1
        else:
            self.stuck = 0
        if self.stuck >= 1:
            self._set_path(self._plan((x, y), th))
        if self.path is not None:
            P = self.path
            n = len(P)
            i = self.idx
            # advance over zero-length mode switches at current location
            while i + 1 < n and abs(P[i + 1][0] - x) < 1e-9 and abs(P[i + 1][1] - y) < 1e-9:
                i += 1
            self.idx = i
            m = P[i][2]
            dth = _wrap(MODES[m] - th)
            act[2] = float(np.clip(dth, -DTH, DTH))
            rot_after = abs(dth) > DTH
            chk = -1 if rot_after else m
            best = None
            j = i + 1
            while j < n and P[j][2] == m:
                d = max(abs(P[j][0] - x), abs(P[j][1] - y))
                if d <= STEP:
                    best = j
                elif best is not None and d > 3 * STEP:
                    break
                j += 1
            if best is not None:
                while best > i and not (self._free_pt(P[best][0], P[best][1], chk)
                                        and self._free_pt(P[best][0], P[best][1], m)):
                    best -= 1
                if best == i:
                    best = None
            if best is not None:
                self.idx = best
                act[0] = P[best][0] - x
                act[1] = P[best][1] - y
            elif not rot_after and i + 1 < n:
                wx, wy = P[i + 1][0], P[i + 1][1]
                dx, dy = wx - x, wy - y
                mx = max(abs(dx), abs(dy))
                sc = min(1.0, STEP / mx) if mx > 0 else 0.0
                act[0] = dx * sc; act[1] = dy * sc
        if self.stuck >= 2:
            act[0] = float(self.rng.uniform(-0.01, 0.01))
            act[1] = float(self.rng.uniform(-0.01, 0.01))
        self.last_moved = bool(abs(act[0]) > 0 or abs(act[1]) > 0 or abs(act[2]) > 0)
        self.last_pos = (x, y, th)
        return np.clip(act, self.action_space.low, self.action_space.high).astype(np.float32)
