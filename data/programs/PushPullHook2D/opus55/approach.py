import numpy as np
import time as _t

L1 = 1.25   # long arm length
L2 = 0.625  # short arm length
HW = 0.05   # hook width
BR = 0.05   # button radius
XMIN, XMAX, YMIN, YMAX = 0.1, 3.4, 0.1, 1.15  # robot center bounds
WX0, WX1, WY0, WY1 = 0.0, 3.5, 0.0, 2.5       # world walls (hook)
GRIP = 0.105  # gripper face distance from robot center (arm=0.1)
GGAP = 0.006
ROBOT_R = 0.14
STEP = 0.05
DTH_MAX = 0.196
NTH = 66


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def hook_frame(th):
    u = np.array([-np.cos(th), -np.sin(th)])
    nn = np.array([np.sin(th), -np.cos(th)])
    return u, nn


def hook_dist(px, py, vx, vy, th):
    """Distance from points to the hook (vectorized)."""
    ux, uy = -np.cos(th), -np.sin(th)
    nx, ny = np.sin(th), -np.cos(th)
    rx, ry = px - vx, py - vy
    a = rx * ux + ry * uy
    b = rx * nx + ry * ny
    d1 = np.hypot(np.maximum(0, np.abs(a - L1 / 2) - L1 / 2), np.maximum(0, np.abs(b - HW / 2) - HW / 2))
    d2 = np.hypot(np.maximum(0, np.abs(a - HW / 2) - HW / 2), np.maximum(0, np.abs(b - L2 / 2) - L2 / 2))
    return np.minimum(d1, d2)


def hook_corners_ok(vx, vy, th, m=0.02):
    ux, uy = -np.cos(th), -np.sin(th)
    nx, ny = np.sin(th), -np.cos(th)
    ok = True
    for (a, b) in [(0, 0), (L1, 0), (L1, HW), (0, L2), (HW, L2)]:
        cx = vx + a * ux + b * nx
        cy = vy + a * uy + b * ny
        ok = ok & (cx > WX0 + m) & (cx < WX1 - m) & (cy > WY0 + m) & (cy < WY1 - m)
    return ok


def robot_disk_free(x, y, vx, vy, th, r=ROBOT_R):
    return hook_dist(x, y, vx, vy, th) > r


def bfs(valid, start, conn, valid2=None, conn2=()):
    """Unit-cost BFS on boolean grid from start index. conn: list of offset tuples.
    Returns distance array (large=unreached)."""
    INF = 10 ** 6
    dist = np.full(valid.shape, INF, dtype=np.int32)
    dist[start] = 0
    front = np.zeros(valid.shape, bool)
    front[start] = True
    visited = front.copy()
    d = 0
    nd = valid.ndim
    while front.any():
        d += 1
        new = np.zeros_like(front)
        for off in conn:
            sh = front
            src = [slice(None)] * nd
            dst = [slice(None)] * nd
            rolled = sh
            for ax, o in enumerate(off):
                if o == 0:
                    continue
                if nd == 3 and ax == 2:
                    rolled = np.roll(rolled, o, axis=2)
                else:
                    if o > 0:
                        src[ax] = slice(0, -o)
                        dst[ax] = slice(o, None)
                    else:
                        src[ax] = slice(-o, None)
                        dst[ax] = slice(0, o)
            new[tuple(dst)] |= rolled[tuple(src)]
        if valid2 is not None:
            for off in conn2:
                src = [slice(None)] * nd
                dst = [slice(None)] * nd
                rolled = front
                for ax, o in enumerate(off):
                    if o == 0:
                        continue
                    if ax == 2:
                        rolled = np.roll(rolled, o, axis=2)
                    else:
                        if o > 0:
                            src[ax] = slice(0, -o)
                            dst[ax] = slice(o, None)
                        else:
                            src[ax] = slice(-o, None)
                            dst[ax] = slice(0, o)
                new[tuple(dst)] |= rolled[tuple(src)] & valid2[tuple(dst)]
        new &= valid & ~visited
        dist[new] = d
        visited |= new
        front = new
        if d > 3000:
            break
    return dist


def descend(dist, goal, conn, valid2=None, conn2=()):
    """Path from start (dist 0) to goal following decreasing dist."""
    path = [goal]
    cur = goal
    nd = dist.ndim
    while dist[cur] > 0:
        best = None
        allc = sorted(list(conn), key=lambda c: (abs(c[-1]) if nd == 3 else 0, sum(abs(z) for z in c[:-1]) if nd == 3 else 0))
        if valid2 is not None and valid2[cur]:
            allc = allc + list(conn2)
        for off in allc:
            nxt = []
            okk = True
            for ax, o in enumerate(off):
                v = cur[ax] - o
                if nd == 3 and ax == 2:
                    v %= dist.shape[2]
                elif v < 0 or v >= dist.shape[ax]:
                    okk = False
                    break
                nxt.append(v)
            if not okk:
                continue
            nxt = tuple(nxt)
            if dist[nxt] == dist[cur] - 1:
                best = nxt
                break
        if best is None:
            break
        cur = best
        path.append(cur)
    path.reverse()
    return path


def _mk_params():
    rows = []
    for sg in (0.15, 0.1, 0.06, 0.035):
        for b in np.arange(0.08, 0.56, 0.04):
            rows.append((0.0, 0.0, b, sg))
        for b in np.arange(0.15, 0.56, 0.04):
            rows.append((np.pi, HW, b, sg))
        for a in np.arange(0.2, 1.2, 0.05):
            rows.append((np.pi / 2, a, HW, sg))
        for a in np.arange(0.08, 1.2, 0.05):
            rows.append((-np.pi / 2, a, 0.0, sg))
    return np.array(rows)


PARAMS = _mk_params()
_g = np.arange(-0.5, 0.501, 0.05)
WAY_OFFS = sorted([np.array([a, b]) for a in _g for b in _g if 0.04 < np.hypot(a, b) <= 0.5], key=lambda w: np.hypot(*w))

CONN2 = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
CONN3B = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-2, 2)]
CONN3 = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.obs = None
        self.gen = None

    def reset(self, state, info):
        self.obs = np.asarray(state, dtype=float)
        self.gen = self._run()
        self.bad_grasps = set()
        self.t_total = 0.0

    def get_action(self, state):
        self.obs = np.asarray(state, dtype=float)
        _t0 = _t.time()
        try:
            a = next(self.gen)
        except StopIteration:
            self.gen = self._run()
            try:
                a = next(self.gen)
            except StopIteration:
                a = np.zeros(5)
        self.t_total += _t.time() - _t0
        a = np.asarray(a, dtype=float)
        a[0] = np.clip(a[0], -STEP, STEP)
        a[1] = np.clip(a[1], -STEP, STEP)
        a[2] = np.clip(a[2], -DTH_MAX, DTH_MAX)
        a[3] = np.clip(a[3], -0.1, 0.1)
        a[4] = np.clip(a[4], 0, 1)
        return a.astype(np.float32)

    # ---------------------------------------------------------------- helpers
    def hook_pose(self):
        o = self.obs
        return o[9], o[10], o[11]

    def move_toward(self, x, y, th, vac, darm=0.0):
        o = self.obs
        x = min(max(x, XMIN + 1e-4), XMAX - 1e-4)
        y = min(max(y, YMIN + 1e-4), YMAX - 1e-4)
        dx = np.clip(x - o[0], -STEP, STEP)
        dy = np.clip(y - o[1], -STEP, STEP)
        dt = np.clip(wrap(th - o[2]), -DTH_MAX, DTH_MAX)
        return np.array([dx, dy, dt, darm, vac])

    def goto(self, x, y, th, vac, tol=1e-4, maxn=60):
        """Generator: move to pose. Returns True if reached."""
        for _ in range(maxn):
            o = self.obs
            if abs(x - o[0]) < tol and abs(y - o[1]) < tol and abs(wrap(th - o[2])) < tol:
                return True
            prev = o[:3].copy()
            yield self.move_toward(x, y, th, vac)
            if np.allclose(prev, self.obs[:3], atol=1e-7):
                return False
        return False

    def robot_for_hook(self, vx, vy, hth, off, dth):
        rth = hth - dth
        c, s = np.cos(rth), np.sin(rth)
        return vx - (c * off[0] - s * off[1]), vy - (s * off[0] + c * off[1]), wrap(rth)

    # ---------------------------------------------------------------- grasp
    def grasp_options(self):
        vx, vy, th = self.hook_pose()
        V = np.array([vx, vy])
        u, nn = hook_frame(th)
        opts = []
        for a in np.arange(1.2, 0.25, -0.05):
            for side in (0, 1):
                if side == 0:  # outer face, robot on -nn side
                    P = V + a * u
                    dirv = nn  # robot facing direction
                    R = P - nn * (GRIP + GGAP)
                else:
                    P = V + a * u + HW * nn
                    dirv = -nn
                    R = P + nn * (GRIP + GGAP)
                S = R - dirv * 0.05
                rth = np.arctan2(dirv[1], dirv[0])
                ok = (round(a, 3), side) not in self.bad_grasps
                for q in (R, S):
                    if not (XMIN + 0.02 <= q[0] <= XMAX - 0.02 and YMIN + 0.02 <= q[1] <= YMAX - 0.02):
                        ok = False
                if not ok:
                    continue
                if not robot_disk_free(S[0], S[1], vx, vy, th, ROBOT_R + 0.005):
                    continue
                # predicted transform after grasp
                d = V - R
                c, s = np.cos(rth), np.sin(rth)
                off = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
                dth = wrap(th - rth)
                opts.append(dict(key=(round(a, 3), side), a=a, side=side, R=R, S=S, rth=rth, dirv=dirv, off=off, dth=dth))
        return opts

    def nav_grid2(self):
        o = self.obs
        x0, y0 = o[0], o[1]
        kx0 = int(np.floor((x0 - XMIN) / STEP + 1e-9))
        kx1 = int(np.floor((XMAX - x0) / STEP + 1e-9))
        ky0 = int(np.floor((y0 - YMIN) / STEP + 1e-9))
        ky1 = int(np.floor((YMAX - y0) / STEP + 1e-9))
        xs = x0 + STEP * np.arange(-kx0, kx1 + 1)
        ys = y0 + STEP * np.arange(-ky0, ky1 + 1)
        return xs, ys, (kx0, ky0)

    # ---------------------------------------------------------------- push candidates
    def push_candidates(self, off, dth, B, T, final=True, first_only=False):
        d = T - B
        dn = np.linalg.norm(d)
        nh = d / max(dn, 1e-9)
        phi = np.arctan2(nh[1], nh[0])
        P = PARAMS
        hth = wrap(phi + P[:, 0])
        ux, uy = -np.cos(hth), -np.sin(hth)
        nx, ny = np.sin(hth), -np.cos(hth)
        rth = hth - dth
        c, s = np.cos(rth), np.sin(rth)
        ox = c * off[0] - s * off[1]
        oy = s * off[0] + c * off[1]
        ok = np.ones(len(P), bool)
        stage = None
        endoff = 0.13 if final else 0.08
        for idx, cpt in enumerate((B[None, :] - nh[None, :] * (BR + P[:, 3:4]), np.tile(T - nh * endoff, (len(P), 1)))):
            vx = cpt[:, 0] - P[:, 1] * ux - P[:, 2] * nx
            vy = cpt[:, 1] - P[:, 1] * uy - P[:, 2] * ny
            rx = vx - ox
            ry = vy - oy
            ok &= (rx >= XMIN + 0.005) & (rx <= XMAX - 0.005) & (ry >= YMIN + 0.005) & (ry <= YMAX - 0.005)
            ok &= hook_corners_ok(vx, vy, hth, 0.03)
            if idx == 0:
                stage = (rx, ry, wrap(rth))
        cands = []
        for k in np.nonzero(ok)[0]:
            cands.append(dict(rel=P[k, 0], pa=P[k, 1], pb=P[k, 2], sg=P[k, 3], hth=hth[k],
                              stage=(stage[0][k], stage[1][k], stage[2][k])))
            if first_only:
                break
        return cands

    def plan_leg(self, off, dth, B, T, quick=False):
        """Returns (goal, final, cands) for the first leg or None."""
        c = self.push_candidates(off, dth, B, T, True, quick)
        if c:
            return (T, True, c)
        best = None
        t0 = _t.time()
        tl = 1.5 if self.t_total < 30 else (0.3 if self.t_total < 45 else 0.0)
        for W in WAY_OFFS:
            if _t.time() - t0 > tl:
                break
            Wp = B + W
            if not (0.3 < Wp[0] < 3.2 and 1.32 < Wp[1] < 2.3):
                continue
            L = np.linalg.norm(W) + np.linalg.norm(T - Wp)
            if best is not None and L >= best[0]:
                continue
            if not self.push_candidates(off, dth, Wp, T, True, True):
                continue
            c1 = self.push_candidates(off, dth, B, Wp, False, quick)
            if c1:
                best = (L, Wp, c1)
        if best is None:
            return None
        return (best[1], False, best[2])

    def transit_plan(self, x0, y0, th0, off, dth, B, cands):
        kx0 = int(np.floor((x0 - XMIN) / STEP + 1e-9))
        kx1 = int(np.floor((XMAX - x0) / STEP + 1e-9))
        ky0 = int(np.floor((y0 - YMIN) / STEP + 1e-9))
        ky1 = int(np.floor((YMAX - y0) / STEP + 1e-9))
        xs = x0 + STEP * np.arange(-kx0, kx1 + 1)
        ys = y0 + STEP * np.arange(-ky0, ky1 + 1)
        ths = wrap(th0 + 2 * np.pi / NTH * np.arange(NTH))
        X, Y, TH = np.meshgrid(xs, ys, ths, indexing='ij')
        hth = TH + dth
        c, s = np.cos(TH), np.sin(TH)
        VX = X + c * off[0] - s * off[1]
        VY = Y + s * off[0] + c * off[1]
        valid = hook_corners_ok(VX, VY, hth, 0.03)
        valid &= hook_dist(B[0], B[1], VX, VY, hth) > BR + 0.03
        start = (kx0, ky0, 0)
        valid[start] = True
        valid2 = valid & np.roll(valid, 1, axis=2) & np.roll(valid, -1, axis=2)
        dist = bfs(valid, start, CONN3, valid2, CONN3B)
        best = None
        for cd in cands:
            rx, ry, rth = cd['stage']
            i = int(round((rx - xs[0]) / STEP))
            j = int(round((ry - ys[0]) / STEP))
            k = int(round(wrap(rth - ths[0]) / (2 * np.pi / NTH))) % NTH
            if not (0 <= i < len(xs) and 0 <= j < len(ys)):
                continue
            if dist[i, j, k] >= 10 ** 6:
                continue
            cost = dist[i, j, k] + (cd['sg'] - 0.06) / 0.05
            if best is None or cost < best[0]:
                best = (cost, cd, (i, j, k))
        if best is None:
            return None
        self.valid2 = valid2
        return best, dist, xs, ys, ths

    # ---------------------------------------------------------------- main
    def _run(self):
        for _ in range(6):
            got = yield from self._grasp()
            if got:
                break
        while True:
            res = yield from self._transit_and_push()
            if res == 'regrasp':
                if self.obs[6] > 0.5 and self.transform_ok():
                    yield from self._park()
                yield np.array([0, 0, 0, 0, 0.0])
                for _ in range(6):
                    got = yield from self._grasp()
                    if got:
                        break

    def _park(self):
        """Carry the hook to a hanging (long arm down) pose before release."""
        o = self.obs
        B = o[20:22].copy()
        off, dth = self.off, self.dth
        hth = np.pi / 2
        cands = []
        for vx in np.arange(0.2, 3.3, 0.1):
            for vy in (1.35, 1.45, 1.55, 1.65):
                rx, ry, rth = self.robot_for_hook(vx, vy, hth, off, dth)
                if not (XMIN + 0.01 <= rx <= XMAX - 0.01 and YMIN + 0.01 <= ry <= YMAX - 0.01):
                    continue
                if not hook_corners_ok(vx, vy, hth, 0.05):
                    continue
                if hook_dist(B[0], B[1], vx, vy, hth) < BR + 0.1:
                    continue
                cands.append(dict(stage=(rx, ry, rth), sg=0.06))
        if not cands:
            return False
        tp = self.transit_plan(o[0], o[1], o[2], off, dth, B, cands)
        if tp is None:
            return False
        self.phase = 'park'
        best, dist, xs, ys, ths = tp
        _, cd, goal = best
        path = descend(dist, goal, CONN3, self.valid2, CONN3B)
        for (i, j, k) in path[1:]:
            ok = yield from self.goto(xs[i], ys[j], ths[k], 1.0)
            if not ok:
                return False
        rx, ry, rth = cd['stage']
        yield from self.goto(rx, ry, rth, 1.0)
        return True

    def _grasp(self):
        o = self.obs
        vx, vy, th = self.hook_pose()
        B = o[20:22].copy()
        T = o[29:31].copy()
        self.phase = 'grasp_nav'
        opts = self.grasp_options()
        # arm fully retracted
        if o[4] > 0.1 + 1e-6:
            yield np.array([0, 0, 0, -0.1, 0])
        xs, ys, start = self.nav_grid2()
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        valid = robot_disk_free(X, Y, vx, vy, th, ROBOT_R)
        valid[start] = True
        dist = bfs(valid, start, CONN2)
        best = None
        scored = []
        for op in opts:
            S = op['S']
            i = int(round((S[0] - xs[0]) / STEP))
            j = int(round((S[1] - ys[0]) / STEP))
            if not (0 <= i < len(xs) and 0 <= j < len(ys)):
                continue
            if dist[i, j] >= 10 ** 6:
                continue
            scored.append((dist[i, j], op, (i, j)))
        scored.sort(key=lambda z: z[0])
        t0 = _t.time()
        nev = 0
        for dcost, op, ij in scored:
            if best is not None and dcost >= best[0]:
                break
            tl = 4.0 if self.t_total < 15 else (1.0 if self.t_total < 35 else 0.0)
            if nev >= 14 or _t.time() - t0 > tl or (self.t_total > 35 and nev >= 1):
                break
            leg = self.plan_leg(op['off'], op['dth'], B, T)
            if leg is None:
                continue
            nev += 1
            R = op['R']
            tp = self.transit_plan(R[0], R[1], op['rth'], op['off'], op['dth'], B, leg[2])
            if tp is None:
                continue
            cost = dcost + tp[0][0] + (0 if leg[1] else 25)
            if best is None or cost < best[0]:
                best = (cost, op, ij)
        if best is None:
            # fallback: any reachable option
            for op in opts:
                S = op['S']
                i = int(round((S[0] - xs[0]) / STEP))
                j = int(round((S[1] - ys[0]) / STEP))
                if 0 <= i < len(xs) and 0 <= j < len(ys) and dist[i, j] < 10 ** 6:
                    best = (dist[i, j], op, (i, j))
                    break
        if best is None:
            yield np.array([0, 0, 0, 0, 0.0])
            return False
        _, op, goal = best
        path = descend(dist, goal, CONN2)
        rth = op['rth']
        for (i, j) in path[1:]:
            prev = self.obs[:2].copy()
            yield self.move_toward(xs[i], ys[j], rth, 0.0)
            if np.allclose(prev, self.obs[:2], atol=1e-7):
                return False
        yield from self.goto(op['S'][0], op['S'][1], rth, 0.0)
        self.phase = 'grasp_approach'
        # approach until (near) contact with vacuum on
        dv = op['dirv']
        R = op['R']
        yield from self.goto(R[0], R[1], rth, 1.0, maxn=4)
        o = self.obs
        V = o[9:11]
        u, nn = hook_frame(o[11])
        b = np.dot(o[0:2] - V, nn)
        # distance of gripper face to hook face
        gapd = (b - HW - GRIP) if op['side'] == 1 else (-b - GRIP)
        if gapd > 0.012:
            for _ in range(6):
                prev = self.obs[:2].copy()
                yield np.array([dv[0] * 0.005, dv[1] * 0.005, 0, 0, 1.0])
                if np.allclose(prev, self.obs[:2], atol=1e-7):
                    break
            o = self.obs
            V = o[9:11]
            u, nn = hook_frame(o[11])
            b = np.dot(o[0:2] - V, nn)
            gapd = (b - HW - GRIP) if op['side'] == 1 else (-b - GRIP)
            if gapd > 0.014:
                self.bad_grasps.add(op['key'])
                return False
        self.update_transform()
        self.grasp_key = op['key']
        self.attached_checked = False
        return True

    def update_transform(self):
        o = self.obs
        c, s = np.cos(o[2]), np.sin(o[2])
        d = o[9:11] - o[0:2]
        self.off = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        self.dth = wrap(o[11] - o[2])

    def transform_ok(self):
        o = self.obs
        c, s = np.cos(o[2]), np.sin(o[2])
        d = o[9:11] - o[0:2]
        off = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        return np.linalg.norm(off - self.off) < 0.01 and abs(wrap(o[11] - o[2] - self.dth)) < 0.01

    def _transit_and_push(self):
        o = self.obs
        B = o[20:22].copy()
        T = o[29:31].copy()
        if o[6] < 0.5:
            return 'regrasp'
        self.update_transform()
        off, dth = self.off, self.dth
        leg = self.plan_leg(off, dth, B, T)
        if leg is None:
            return 'regrasp'
        W, final, cands = leg
        tp = self.transit_plan(o[0], o[1], o[2], off, dth, B, cands)
        if tp is None:
            return 'regrasp'
        self.phase = 'transit'
        best, dist, xs, ys, ths = tp
        _, cd, goal = best
        path = descend(dist, goal, CONN3, self.valid2, CONN3B)
        for (i, j, k) in path[1:]:
            ok = yield from self.goto(xs[i], ys[j], ths[k], 1.0)
            if not self.transform_ok():
                if not self.attached_checked:
                    self.bad_grasps.add(self.grasp_key)
                    return 'regrasp'
                return 'replan'
            self.attached_checked = True
            if not ok:
                return 'replan'
        rx, ry, rth = cd['stage']
        yield from self.goto(rx, ry, rth, 1.0)
        self.phase = 'push'
        # push phase (closed loop)
        rel, pa, pb = cd['rel'], cd['pa'], cd['pb']
        nh0 = (W - B) / max(np.linalg.norm(W - B), 1e-9)
        stuck = 0
        for _ in range(400):
            o = self.obs
            B = o[20:22]
            T = o[29:31]
            if final:
                d = T - B
                nh = d / max(np.linalg.norm(d), 1e-9)
            else:
                if np.dot(B - W, nh0) > -0.03:
                    return 'replan'
                nh = nh0
            phi = np.arctan2(nh[1], nh[0])
            hth = wrap(phi + rel)
            u, nn = hook_frame(hth)
            # current face point distance to button
            cur_v = o[9:11]
            cur_u, cur_n = hook_frame(o[11])
            cur_c = cur_v + pa * cur_u + pb * cur_n
            gap = np.dot(B - cur_c, nh) - BR
            lat = np.dot(B - cur_c, np.array([-nh[1], nh[0]]))
            ang_err = abs(wrap(o[11] - hth))
            if gap > 0.02 and (abs(lat) > 0.01 or ang_err > 0.02):
                g = max(gap, 0.02)
            else:
                g = -0.005
            c = B - nh * (BR + g)
            V = c - pa * u - pb * nn
            rx, ry, rth = self.robot_for_hook(V[0], V[1], hth, off, dth)
            prev = o[:3].copy()
            yield self.move_toward(rx, ry, rth, 1.0)
            if np.allclose(prev, self.obs[:3], atol=1e-7):
                stuck += 1
                if stuck > 3:
                    return 'replan'
            else:
                stuck = 0
            if abs(self.obs[6]) < 0.5:
                return 'regrasp'
        return 'replan'
