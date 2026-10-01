"""ClutteredStorage2D approach: push in-shelf blocks deep, then fetch floor blocks
and stack them (horizontal) into the shelf from below."""
import heapq
import math

import numpy as np

SHELF_BOTTOM = 2.625
SHELF_TOP = 3.0
WORLD_X = (0.0, 5.0)
BW, BH = 0.28, 0.04
COL_W = 0.3175
DEBUG = False
STACK_DY = 0.048
LAZY_PUSH_Y = 1.8


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def safe_th(t):
    """wrap and keep away from +-pi (float32 +-pi falls outside [-pi, pi])."""
    t = wrap(t)
    if abs(t) > math.pi - 1e-4:
        return math.copysign(math.pi - 2e-4, t)
    return t


def rect_corners(x, y, th, w, h):
    c, s = math.cos(th), math.sin(th)
    return np.array([[x, y], [x + w * c, y + w * s],
                     [x + w * c - h * s, y + w * s + h * c], [x - h * s, y + h * c]])


def seg_dist(px, py, a, b):
    """distance from points (arrays) to segment ab."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / max(L2, 1e-12), 0, 1)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy)


def poly_dist(px, py, poly):
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    d = None
    inside = np.ones(np.broadcast(px, py).shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        di = seg_dist(px, py, a, b)
        d = di if d is None else np.minimum(d, di)
        cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        inside &= cross >= 0
    return np.where(inside, 0.0, d)


def polys_overlap(p, q):
    """SAT test for convex polygons."""
    for poly in (p, q):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            ax = np.array([-e[1], e[0]])
            a = p @ ax
            b = q @ ax
            if a.max() < b.min() - 1e-9 or b.max() < a.min() - 1e-9:
                return False
    return True


def poly_min_y_in_strip(poly, xa, xb, n=21):
    """lowest y of polygon's boundary within x in [xa, xb]."""
    best = None
    for x in np.linspace(xa, xb, n):
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            if (a[0] - x) * (b[0] - x) <= 0 and abs(b[0] - a[0]) > 1e-12:
                t = (x - a[0]) / (b[0] - a[0])
                y = a[1] + t * (b[1] - a[1])
                best = y if best is None else min(best, y)
    if best is None:
        best = poly[:, 1].min()
    return best


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.low = np.array(action_space.low, dtype=np.float32)
        self.high = np.array(action_space.high, dtype=np.float32)
        self.res = 0.05

    # ------------------------------------------------------------------ utils
    def _parse(self, state):
        self.state = state
        names = state.get_object_names()
        self.robot = state.get_object_from_name('robot')
        self.shelf = state.get_object_from_name('shelf')
        g = lambda o, f: float(state.get(o, f))
        r = self.robot
        self.rx, self.ry, self.rth = g(r, 'x'), g(r, 'y'), g(r, 'theta')
        self.arm = g(r, 'arm_joint')
        self.arm_max = g(r, 'arm_length')
        self.base_r = g(r, 'base_radius')
        self.vac = g(r, 'vacuum')
        s = self.shelf
        self.sx1, self.sw1 = g(s, 'x1'), g(s, 'width1')
        self.sy1, self.sh1 = g(s, 'y1'), g(s, 'height1')
        self.blocks = {}
        for n in names:
            o = state.get_object_from_name(n)
            if n in ('robot', 'shelf'):
                continue
            try:
                w, h = g(o, 'width'), g(o, 'height')
            except Exception:
                continue
            poly = rect_corners(g(o, 'x'), g(o, 'y'), g(o, 'theta'), w, h)
            self.blocks[n] = dict(x=g(o, 'x'), y=g(o, 'y'), th=g(o, 'theta'), w=w, h=h,
                                  poly=poly, center=poly.mean(axis=0))

    def _act(self, dx=0.0, dy=0.0, dth=0.0, darm=0.0, vac=0.0):
        a = np.array([dx, dy, dth, darm, vac], dtype=np.float32)
        return np.clip(a, self.low, self.high)

    def _inside(self, b):
        return b['poly'][:, 1].min() > SHELF_BOTTOM + 0.001

    # --------------------------------------------------------------- planning
    def _obstacles(self, exclude):
        return [b['poly'] for n, b in self.blocks.items()
                if n not in exclude and b['poly'][:, 1].min() < SHELF_BOTTOM]

    def _clearance(self, x, y, exclude):
        d = min(x - WORLD_X[0], WORLD_X[1] - x, y, SHELF_BOTTOM - y)
        for p in self._obstacles(exclude):
            d = min(d, float(poly_dist(x, y, p)))
        return d

    def _plan(self, start, goal, R, exclude):
        res = self.res
        xs = np.arange(0.0, 5.0 + 1e-9, res)
        ys = np.arange(0.0, SHELF_BOTTOM + 1e-9, res)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        clear = np.minimum.reduce([X - WORLD_X[0], WORLD_X[1] - X, Y, SHELF_BOTTOM - Y])
        for p in self._obstacles(exclude):
            clear = np.minimum(clear, poly_dist(X, Y, p))
        free = clear >= R
        nx, ny = free.shape

        def idx(p):
            return (int(np.clip(round(p[0] / res), 0, nx - 1)),
                    int(np.clip(round(p[1] / res), 0, ny - 1)))

        def nearest_free(c):
            if free[c]:
                return c
            best, bd = None, 1e9
            fi = np.argwhere(free)
            if len(fi) == 0:
                return c
            d = (fi[:, 0] - c[0]) ** 2 + (fi[:, 1] - c[1]) ** 2
            k = int(np.argmin(d))
            return tuple(fi[k])

        s = idx(start)
        gcell = nearest_free(idx(goal))
        # A* allowing blocked cells at high cost (to escape start)
        pen = np.where(free, 0.0, 20.0)
        dist = {s: 0.0}
        prev = {}
        pq = [(0.0, s)]
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        found = False
        while pq:
            f, c = heapq.heappop(pq)
            if c == gcell:
                found = True
                break
            dc = dist[c]
            if f - max(abs(c[0] - gcell[0]), abs(c[1] - gcell[1])) > dc + 1e-9:
                continue
            for di, dj in nbrs:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < nx and 0 <= n[1] < ny):
                    continue
                if not free[n] and free[c]:
                    continue  # never re-enter blocked region
                nd = dc + (1.0 if di == 0 or dj == 0 else 1.02) + pen[n]
                if nd < dist.get(n, 1e18):
                    dist[n] = nd
                    prev[n] = c
                    heapq.heappush(pq, (nd + max(abs(n[0] - gcell[0]), abs(n[1] - gcell[1])), n))
        if not found:
            return None
        cells = [gcell]
        while cells[-1] != s:
            cells.append(prev[cells[-1]])
        cells.reverse()
        pts = [(c[0] * res, c[1] * res) for c in cells]
        pts[0] = tuple(start)
        # shortcut smoothing
        def los(a, b):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(2, int(L / 0.01))
            for t in np.linspace(0, 1, n):
                c = idx((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                if not free[c]:
                    return False
            return True
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not (free[idx(pts[i])] and los(pts[i], pts[j])):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    # ------------------------------------------------------------ task logic
    def reset(self, state, info):
        self._parse(state)
        self.phase = None
        self.task = None
        self.path = None
        self.last_pose = None
        self.stuck = 0
        self.failed = {}
        self.bad_cols = {}
        self.cap_zero = {}
        self.idle = 0
        self.rng = np.random.default_rng(0)
        self.allowed_cols = None
        self.lazy_cols = set()

    def _columns(self):
        K = max(1, int(round(self.sw1 / COL_W)))
        return [self.sx1 + self.sw1 * (j + 0.5) / K for j in range(K)], self.sw1 / K

    def _col_ceiling(self, cx, halfw, exclude=()):
        top = SHELF_TOP
        for n, b in self.blocks.items():
            if n in exclude or not self._inside(b):
                continue
            p = b['poly']
            if p[:, 0].max() > cx - halfw and p[:, 0].min() < cx + halfw:
                top = min(top, p[:, 1].min())
        return top

    def _need_bottom(self):
        cols, _ = self._columns()
        n_out = sum(1 for b in self.blocks.values() if not self._inside(b))
        per = -(-n_out // len(cols))
        return SHELF_BOTTOM + per * 0.056 + 0.02

    def _bx_interval(self, cx, cw, ox_rel, gdx=0.0):
        """feasible block-center x interval in column centered cx when block center is
        ox_rel right of robot center and gripper is gdx right of robot center."""
        ox = self.sx1 + 0.004
        oX = self.sx1 + self.sw1 - 0.004
        slack = max(0.0, cw / 2 - BW / 2 - 0.006)
        lo = max(cx - slack, ox + BW / 2 + 0.002, self.base_r + 0.002 + ox_rel,
                 ox + 0.072 - gdx + ox_rel)
        hi = min(cx + slack, oX - BW / 2 - 0.002, 5.0 - self.base_r - 0.002 + ox_rel,
                 oX - 0.072 - gdx + ox_rel)
        return lo, hi

    def _ins_target_top(self, bx, gx, exclude=()):
        """highest allowed top y for a horizontal block centered at bx pushed by gripper at gx."""
        cb = self._col_ceiling(bx, BW / 2 + 0.004, exclude)
        cg = self._col_ceiling(gx, 0.075, exclude)
        return min(cb - 0.006, cg - 0.008 + BH)

    def _cap(self, ceil):
        return max(0, int(math.floor((ceil - (SHELF_BOTTOM + 0.004)) / STACK_DY)))

    def _plan_push(self):
        """Return (block, target_bottom) of the next in-shelf block to lift, or None.
        Also sets self.plan_total (current capacity) and self.lazy_cols (columns that
        can be filled now without losing needed potential capacity)."""
        cols, cw = self._columns()
        n_out = sum(1 for b in self.blocks.values() if not self._inside(b))
        self.plan_total = 0
        self.lazy_cols = set(range(len(cols)))
        if n_out == 0:
            return None
        info = []
        total = 0
        tot_pot = 0
        for j, cx in enumerate(cols):
            if all(lo > hi for lo, hi in (self._bx_interval(cx, cw, ox, 0.0)
                                          for ox in (0.0, 0.07, -0.07, 0.11, -0.11))):
                continue
            ceil = self._col_ceiling(cx, cw / 2 - 0.005)
            cap = self._cap(ceil)
            zero = j in self.cap_zero and abs(self.cap_zero[j] - ceil) < 1e-3
            if zero:
                cap = 0
            members = [(n, b) for n, b in self.blocks.items() if self._inside(b)
                       and abs(b['center'][0] - cx) <= cw / 2]
            pot, push = cap, None
            if members:
                n, b = min(members, key=lambda nb: nb[1]['poly'][:, 1].min())
                if self.failed.get(n, 0) < 3:
                    bot, top = float(b['poly'][:, 1].min()), float(b['poly'][:, 1].max())
                    above = self._col_ceiling(b['center'][0], 0.145, exclude=(n,))
                    max_bot = above - 0.008 - (top - bot)
                    if max_bot >= bot + 0.02:
                        pc = self._cap(max_bot - 0.004)
                        if zero and max_bot < bot + STACK_DY:
                            pc = 0
                        if pc > cap:
                            pot = pc
                            push = (n, bot, max_bot)
            total += cap
            tot_pot += pot
            info.append((j, cx, cap, pot, zero, push))
        deficit = n_out - total
        self.plan_total = total
        if deficit <= 0:
            return None
        self.lazy_cols = {j for j, cx, cap, pot, zero, push in info
                          if cap > 0 and tot_pot - pot + cap >= n_out}
        best = None
        for j, cx, cap, pot, zero, push in info:
            if push is None:
                continue
            n, bot, max_bot = push
            k = min(pot, cap + deficit)
            R = SHELF_BOTTOM + 0.004 + k * STACK_DY + 0.004
            if zero:
                R = max(R, bot + STACK_DY)
            R = min(R, max_bot)
            gain = k - cap
            trips = sorted(max(abs(bo['center'][0] - cx), abs(2.3 - bo['center'][1]))
                           for bo in self.blocks.values() if not self._inside(bo))
            trip = 2 * sum(trips[:gain]) / 0.05
            cost = -gain * 100 + abs(cx - self.rx) / 0.05 + trip
            if best is None or cost < best[0]:
                best = (cost, n, R)
        if best is None:
            self.lazy_cols = set(range(len(cols)))
            return None
        return best[1], best[2]

    def _choose_task(self):
        pp = self._plan_push()
        self.allowed_cols = self.lazy_cols if pp is not None else None
        if pp is not None and (not self.lazy_cols or self.ry > LAZY_PUSH_Y):
            self.push_target_bottom = pp[1]
            return ('push', pp[0])
        outs = [(n, b) for n, b in self.blocks.items() if not self._inside(b)]
        if not outs:
            return None
        def fcost(nb):
            opts = self._grasp_options(nb[0])
            base = opts[0][0] if opts else 1e4
            return self.failed.get(nb[0], 0) * 200 + base
        def rough(nb):
            c = nb[1]['center']
            sgx = min(max(c[0], self.sx1 + 0.15), self.sx1 + self.sw1 - 0.15)
            return self.failed.get(nb[0], 0) * 10 + max(abs(c[0] - self.rx), abs(c[1] - self.ry)) \
                + max(abs(c[0] - sgx), abs(c[1] - 2.3))
        outs.sort(key=rough)
        outs = outs[:4]
        outs.sort(key=fcost)
        return ('fetch', outs[0][0])

    def get_action(self, state):
        self._parse(state)
        pose = (self.rx, self.ry, self.rth, self.arm, self.vac)
        if self.last_pose is not None and np.allclose(pose, self.last_pose, atol=1e-7) \
                and self.last_nonzero:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_pose = pose
        a = self._policy()
        self.last_nonzero = bool(np.any(np.abs(a[:4]) > 1e-6))
        # watchdog: many consecutive idle steps -> random nudge to break deadlocks
        self.idle = 0 if (self.last_nonzero or abs(float(a[4]) - self.vac) > 1e-6) else self.idle + 1
        if self.idle > 10:
            ang = self.rng.uniform(-math.pi, math.pi)
            a = self._act(dx=0.05 * math.cos(ang), dy=0.05 * math.sin(ang), darm=-0.1, vac=0.0)
            self.last_nonzero = True
            self.task = None
            self.phase = None
        return a

    def _new_task(self):
        self.task = self._choose_task()
        self.path = None
        self.phase = 'start'
        self.stuck = 0

    def _abort(self):
        if self.task is not None:
            self.failed[self.task[1]] = self.failed.get(self.task[1], 0) + 1
        self.task = None
        self.phase = 'retreat'
        self.retreat_n = 0
        self.path = None

    def _policy(self):
        if self.phase == 'retreat':
            # drop vacuum, retract arm, then pick new task
            if self.vac > 0.0:
                return self._act(vac=0.0)
            if self.arm > self.base_r + 1e-3 and self.retreat_n < 6:
                self.retreat_n += 1
                dy = 0.0
                if abs(wrap(self.rth - math.pi / 2)) < 0.3 and self.ry > 1.0:
                    nt = self._choose_task()
                    if nt is not None and nt[0] == 'fetch':
                        dy = -0.05
                return self._act(dy=dy, darm=-0.1, vac=0.0)
            self.phase = None
        if self.task is None or self.phase is None:
            self._new_task()
            if self.task is None:
                return self._act()
        if self.stuck > 4:
            self.stuck = 0
            self._abort()
            return self._act(vac=0.0)
        kind, name = self.task
        if name not in self.blocks:
            self._new_task()
            return self._act()
        if kind == 'push':
            return self._push(name)
        return self._fetch(name)

    # --------------------------------------------------------- motion helpers
    def _goto(self, goal, R, exclude, th_goal=None, rot_R=None, vac=0.0):
        """Follow planned path to goal; returns action or None when arrived."""
        if self.path is None:
            self.path = self._plan((self.rx, self.ry), goal, R, exclude)
            if self.path is None:
                return None
            self.path = self.path[1:] + [tuple(goal)]
        while self.path and math.hypot(self.path[0][0] - self.rx, self.path[0][1] - self.ry) < 1e-4:
            self.path.pop(0)
        dth = 0.0
        if th_goal is not None:
            e = wrap(th_goal - self.rth)
            if abs(e) > 1e-6:
                if rot_R is None or self._clearance(self.rx, self.ry, exclude) >= rot_R:
                    dth = e
        if not self.path:
            if abs(dth) < 1e-6:
                return 'done'
            return self._act(dth=dth, vac=vac)
        wx, wy = self.path[0]
        vx, vy = wx - self.rx, wy - self.ry
        m = max(abs(vx), abs(vy))
        if m > 0.05:
            vx, vy = vx * 0.05 / m, vy * 0.05 / m
        return self._act(dx=vx, dy=vy, dth=dth, vac=vac)

    def _straight_up_ready(self, th_goal):
        if self.path is None:
            return False
        gx, gy = self.goal
        return (abs(self.rx - gx) < 0.002 and abs(wrap(th_goal - self.rth)) < 1e-4
                and self.ry <= gy + 1e-6
                and all(abs(p[0] - gx) < 0.002 for p in self.path))

    # ------------------------------------------------------------ push task
    def _push(self, name):
        b = self.blocks[name]
        R = self.base_r + 0.05
        if self.phase == 'start':
            self.failed[name] = self.failed.get(name, 0) + 0.5
            self.phase = 'nav'
            self.path = None
            cx = b['center'][0]
            lo = max(self.sx1 + 0.074, b['poly'][:, 0].min() - 0.03, self.base_r + 0.002)
            hi = min(self.sx1 + self.sw1 - 0.074, b['poly'][:, 0].max() + 0.03, 5.0 - self.base_r - 0.002)
            cx = min(max(cx, lo), hi) if lo <= hi else min(max(cx, self.base_r + 0.002), 5.0 - self.base_r - 0.002)
            self.goal = (cx, SHELF_BOTTOM - R - 0.01)
        if self.phase == 'nav':
            if self.arm > self.base_r + 1e-3:
                return self._act(darm=-0.1)
            a = 'done' if self._straight_up_ready(math.pi / 2) else \
                self._goto(self.goal, R, (), th_goal=math.pi / 2, rot_R=R - 0.02)
            if a is None:
                self._abort()
                return self._act()
            if isinstance(a, str):
                self.phase = 'extend'
            else:
                return a
        if self.phase == 'extend':
            gx = self.rx + self.arm * math.cos(self.rth)
            bot = poly_min_y_in_strip(b['poly'], gx - 0.07, gx + 0.07)
            face = self.ry + self.arm + 0.01
            gap = bot - face
            if gap > 0.02:
                need = gap - 0.008
                da = min(0.1, need, self.arm_max - self.arm)
                room = SHELF_BOTTOM - self.base_r - 0.004 - self.ry
                dy = max(0.0, min(0.05, room, need - da))
                return self._act(dx=self.goal[0] - self.rx, dy=dy, darm=da, vac=1.0)
            self.phase = 'lift'
            self.lift_ref = b['poly'][:, 1].max()
            self.lift_steps = 0
        if self.phase == 'lift':
            top = b['poly'][:, 1].max()
            ceil = self._col_ceiling(b['center'][0], 0.14, exclude=(name,))
            target = min(ceil - 0.008, top + self.push_target_bottom - b['poly'][:, 1].min())
            self.lift_steps += 1
            if self.lift_steps > 3 and abs(top - self.lift_ref) < 1e-6:
                # not grasped; creep a bit
                if self.lift_steps > 12:
                    self._abort()
                    return self._act(vac=0.0)
                return self._act(darm=0.003, vac=1.0)
            d = target - top
            if d > 1e-3 and self.arm < self.arm_max - 1e-4:
                if d <= 0.1 and self.arm + d <= self.arm_max:
                    self.task = None
                    self.phase = 'retreat'
                    self.retreat_n = 0
                    return self._act(darm=d, vac=0.0)
                return self._act(darm=min(0.1, d), vac=1.0)
            self.phase = 'release'
        if self.phase == 'release':
            self.task = None
            self.phase = 'retreat'
            self.retreat_n = 0
            return self._act(vac=0.0)
        return self._act()

    # ------------------------------------------------------------ fetch task
    def _grasp_options(self, name):
        feas_cache = {}
        opts_noroom = []
        b = self.blocks[name]
        th = b['th']
        u = np.array([math.cos(th), math.sin(th)])
        nrm = np.array([-math.sin(th), math.cos(th)])
        c = b['center']
        opts = []
        Rn = self.base_r + 0.035
        for s in (1, -1):
            n = nrm * s
            for off in (0.0, 0.07, -0.07, 0.11, -0.11):
                for D in (0.25, 0.32, 0.42, 0.55):
                    side = c + u * off + n * (b['h'] / 2)
                    p = side + n * D
                    if self._clearance(p[0], p[1], ()) < Rn:
                        continue
                    # gripper swept rect (from grasp pose back to standoff) must be free
                    sweep = np.array([side + u * 0.075 + n * 0.003, side - u * 0.075 + n * 0.003,
                                      side - u * 0.075 + n * D, side + u * 0.075 + n * D])
                    ok = True
                    for m, ob in self.blocks.items():
                        if m == name or self._inside(ob):
                            continue
                        if polys_overlap(sweep, ob['poly']):
                            ok = False
                            break
                    if not ok:
                        continue
                    hd = -n
                    phi = math.pi / 2 - math.atan2(hd[1], hd[0])
                    ux = math.cos(phi) * u[0] - math.sin(phi) * u[1]
                    ox_rel = -off * ux
                    key = round(ox_rel, 3)
                    if key not in feas_cache:
                        cols, cw = self._columns()
                        fz, rm = False, False
                        for j_, cx_ in enumerate(cols):
                            if self.allowed_cols is not None and j_ not in self.allowed_cols:
                                continue
                            lo, hi = self._bx_interval(cx_, cw, ox_rel)
                            if lo > hi:
                                continue
                            fz = True
                            bx_ = min(max(cx_, lo), hi)
                            if self._ins_target_top(bx_, bx_ - ox_rel) - BH >= SHELF_BOTTOM + 0.004:
                                rm = True
                        feas_cache[key] = (fz, rm)
                    fz, rm = feas_cache[key]
                    if not fz:
                        continue
                    sgx = min(max(p[0], self.sx1 + 0.15), self.sx1 + self.sw1 - 0.15)
                    cost = (max(abs(p[0] - self.rx), abs(p[1] - self.ry))
                            + max(abs(p[0] - sgx), abs(p[1] - 2.3))) / 0.05 \
                        + 2 * math.ceil((D - 0.23) / 0.1) + abs(off) * 50
                    (opts if rm else opts_noroom).append((cost, p, safe_th(math.atan2(-n[1], -n[0])), D))
        if not opts:
            opts = opts_noroom
        opts.sort(key=lambda o: o[0])
        return opts

    def _fetch(self, name):
        b = self.blocks[name]
        Rn = self.base_r + 0.035
        if self.phase == 'start':
            opts = self._grasp_options(name)
            if not opts:
                self._abort()
                return self._act()
            _, p, th, D = opts[0]
            self.goal = (float(p[0]), float(p[1]))
            self.goal_th = th
            self.D = D
            self.opts = opts
            self.opt_i = 0
            self.phase = 'nav'
            self.path = None
        if self.phase == 'nav':
            if self.arm > self.base_r + 1e-3:
                return self._act(darm=-0.1)
            a = self._goto(self.goal, Rn, (), th_goal=self.goal_th, rot_R=self.base_r + 0.03)
            tries = 0
            while a is None and self.opt_i + 1 < len(self.opts) and tries < 6:
                # current standoff unreachable: try the next grasp option
                self.opt_i += 1
                tries += 1
                _, p, th, D = self.opts[self.opt_i]
                self.goal = (float(p[0]), float(p[1]))
                self.goal_th = th
                self.D = D
                self.path = None
                a = self._goto(self.goal, Rn, (), th_goal=self.goal_th, rot_R=self.base_r + 0.03)
            if a is None:
                self._abort()
                return self._act()
            if isinstance(a, str):
                self.phase = 'extend'
                self.ext_steps = 0
            else:
                # final step: extend the arm concurrently
                gx, gy = self.goal
                if (self.path is not None and len(self.path) == 1
                        and abs(self.rx + float(a[0]) - gx) < 1e-5 and abs(self.ry + float(a[1]) - gy) < 1e-5
                        and abs(wrap(self.rth + float(a[2]) - self.goal_th)) < 1e-5):
                    h = np.array([math.cos(self.goal_th), math.sin(self.goal_th)])
                    gap = float(((b['poly'] - np.array([gx, gy])) @ h).min()) - (self.arm + 0.01)
                    if gap > 0.02:
                        a = a.copy()
                        a[3] = min(0.1, gap - 0.008)
                        a[4] = 1.0
                        self.phase = 'extend'
                        self.ext_steps = 1
                return a
        if self.phase == 'extend':
            if math.hypot(self.rx - self.goal[0], self.ry - self.goal[1]) > 1e-4:
                self.phase = 'nav'
                self.path = None
                return self._act(darm=-(self.arm - self.base_r), vac=0.0)
            # distance from gripper face to block along heading
            h = np.array([math.cos(self.rth), math.sin(self.rth)])
            proj = (b['poly'] - np.array([self.rx, self.ry])) @ h
            gap = proj.min() - (self.arm + 0.01)
            self.ext_steps += 1
            if gap > 0.02 and self.ext_steps < 20:
                return self._act(darm=min(0.1, gap - 0.008), vac=1.0)
            self.phase = 'retract'
            self.grab_ref = b['center'].copy()
            self.ret_steps = 0
        if self.phase == 'retract':
            self.ret_steps += 1
            if self.arm > self.base_r + 1e-3:
                return self._act(darm=-(self.arm - self.base_r), vac=1.0)
            if np.linalg.norm(b['center'] - self.grab_ref) < 1e-4 and self.ret_steps > 1 \
                    and self.D > 0.26:
                pass
            # check held: block relative pose must be close to gripper
            h = np.array([math.cos(self.rth), math.sin(self.rth)])
            proj = (b['poly'] - np.array([self.rx, self.ry])) @ h
            if proj.min() - (self.arm + 0.01) > 0.03:
                self._abort()
                return self._act(vac=0.0)
            self.phase = 'carry_plan'
        if self.phase == 'carry_plan':
            rel = wrap(b['th'] - self.rth)
            # choose robot theta so block is horizontal, arm near pi/2
            cands = [wrap(k * math.pi - rel) for k in range(-2, 3)]
            th_t = min(cands, key=lambda t: abs(wrap(t - math.pi / 2)))
            blk = wrap(th_t + rel)
            th_t = safe_th(th_t + (safe_th(blk) - blk))
            # relative corners to robot
            rel_c = b['poly'] - np.array([self.rx, self.ry])
            self.Rc = max(float(np.linalg.norm(rel_c, axis=1).max()), self.base_r + 0.025) + 0.03
            dth = wrap(th_t - self.rth)
            c, s = math.cos(dth), math.sin(dth)
            off = (b['center'] - np.array([self.rx, self.ry]))
            off = np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])
            cols, cw = self._columns()
            ox = self.sx1 + 0.004
            oX = self.sx1 + self.sw1 - 0.004
            gdx = 0.5 * math.cos(th_t)  # gripper center x offset from robot (approx)
            best = None
            for j, cx in enumerate(cols):
                if j in self.bad_cols.get(name, ()):
                    continue
                if self.allowed_cols is not None and j not in self.allowed_cols:
                    continue
                lo, hi = self._bx_interval(cx, cw, off[0], gdx)
                if DEBUG:
                    bx_ = min(max(cx, lo), hi)
                    print('col', j, round(lo, 3), round(hi, 3), 'off', off.round(3), 'ttop', round(self._ins_target_top(bx_, bx_ - off[0] + gdx), 3))
                if lo > hi:
                    continue
                bx = min(max(cx, lo), hi)
                gx = bx - off[0]
                t_top = self._ins_target_top(bx, gx + gdx)
                if t_top - BH < SHELF_BOTTOM + 0.004:
                    continue
                space = t_top - SHELF_BOTTOM
                gy = SHELF_BOTTOM - self.Rc - 0.01
                score = -space * 2 + max(abs(gx - self.rx), abs(gy - self.ry))
                if best is None or score < best[0]:
                    best = (score, bx, gx, gy, j, t_top)
            if best is None:
                cols_, cw_ = self._columns()
                for j_, cx_ in enumerate(cols_):
                    self.cap_zero[j_] = self._col_ceiling(cx_, cw_ / 2 - 0.005)
                self._abort()
                return self._act(vac=0.0)
            _, self.cx, gx, gy, self.col_j, t_top = best
            gx = float(np.clip(gx, self.Rc, 5 - self.Rc))
            # lowest robot y from which the arm alone can insert (block top offset off[1]+BH/2)
            gy_low = t_top - (off[1] + BH / 2) - (self.arm_max - self.base_r - 0.03)
            gy_try = min(gy, max(gy_low, self.ry + abs(gx - self.rx)))
            if DEBUG: print("carry", round(self.ry,3), round(gx - self.rx,3), "gy", round(gy,3), "low", round(gy_low,3), "try", round(gy_try,3), "t_top", round(t_top,3), off.round(3))
            if gy_try < gy - 0.02:
                ok = all(self._clearance(gx, yy, (name,)) >= self.Rc
                         for yy in np.linspace(gy_try, gy, 12))
                # corridor above robot for arm+block must be free of floor blocks
                if ok:
                    corr = np.array([[gx - 0.16 + min(0, off[0]), gy_try],
                                     [gx + 0.16 + max(0, off[0]), gy_try],
                                     [gx + 0.16 + max(0, off[0]), SHELF_BOTTOM],
                                     [gx - 0.16 + min(0, off[0]), SHELF_BOTTOM]])
                    for m, ob in self.blocks.items():
                        if m != name and not self._inside(ob) and polys_overlap(corr, ob['poly']):
                            ok = False
                            break
                if ok:
                    gy = float(gy_try)
            self.goal = (gx, gy)
            self.goal_th = th_t
            self.path = None
            self.phase = 'carry'
        if self.phase == 'carry':
            a = 'done' if self._straight_up_ready(self.goal_th) else \
                self._goto(self.goal, self.Rc, (name,), th_goal=self.goal_th,
                           rot_R=self.Rc, vac=1.0)
            if a is None:
                self._abort()
                return self._act(vac=0.0)
            if not isinstance(a, str):
                return a
            self.phase = 'align'
        if self.phase == 'align':
            ex = self.cx - b['center'][0]
            eth = wrap(self.goal_th - self.rth)
            if abs(ex) > 1e-4 or abs(eth) > 1e-5:
                return self._act(dx=ex, dth=eth, vac=1.0)
            self.phase = 'insert'
            self.ins_last = None
        if self.phase == 'backout':
            if self.arm > self.base_r + 1e-3 or self.ry > SHELF_BOTTOM - self.Rc - 0.01:
                dy = -min(0.05, max(0.0, self.ry - (SHELF_BOTTOM - self.Rc - 0.01)))
                return self._act(dy=dy, darm=-0.1, vac=1.0)
            self.phase = 'carry_plan'
            return self._act(vac=1.0)
        if self.phase == 'insert':
            top = b['poly'][:, 1].max()
            bot = b['poly'][:, 1].min()
            gxc = self.rx + self.arm * math.cos(self.rth)
            target = self._ins_target_top(b['center'][0], gxc, exclude=(name,))
            d = target - top
            rejected = self.ins_last is not None and abs(top - self.ins_last) < 1e-7
            self.ins_last = top
            if d > 1e-3 and not rejected:
                s = math.sin(self.rth)
                da = max(0.0, min(0.1, d / max(s, 0.5), self.arm_max - self.arm))
                room = SHELF_BOTTOM - self.base_r - 0.004 - self.ry
                dy = max(0.0, min(0.05, room, d - da * s))
                if da > 1e-5 or dy > 1e-5:
                    if da * s + dy >= d - 1e-4 and bot + da * s + dy > SHELF_BOTTOM + 0.004:
                        # final step: move then release in the same step
                        self.task = None
                        self.phase = 'retreat'
                        self.retreat_n = 0
                        return self._act(dy=dy, darm=da, vac=0.0)
                    return self._act(dy=dy, darm=da, vac=1.0)
            if bot <= SHELF_BOTTOM + 0.002:
                # could not insert: back out holding the block, try another column
                self.bad_cols.setdefault(name, set()).add(self.col_j)
                self.phase = 'backout'
                self.stuck = 0
                return self._act(darm=-0.1, vac=1.0)
            self.task = None
            self.phase = 'retreat'
            self.retreat_n = 0
            return self._act(vac=0.0)
        return self._act()
