import numpy as np

TWO_PI = 2 * np.pi


def wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


def rect_poly(x, y, th, w, h):
    c, s = np.cos(th), np.sin(th)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


def _axes(P):
    e = np.roll(P, -1, axis=0) - P
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def poly_overlap(P, Q, margin=0.0):
    """SAT test for convex polygons; True if closer than margin."""
    for A in (_axes(P), _axes(Q)):
        pp = P @ A.T
        qq = Q @ A.T
        if np.any(pp.max(0) + margin < qq.min(0)) or np.any(qq.max(0) + margin < pp.min(0)):
            return False
    return True


_CIRC = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)])


def convex_hull(pts):
    pts = sorted(set((float(p[0]), float(p[1])) for p in pts))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


class GeneratedApproach:
    Y_MAX = 1.482   # max robot base y (below middle wall)
    Y_MIN = 0.258
    X_MIN = 0.266
    X_MAX = 3.234
    WALL_Y = 1.75
    PUSH_EXTRA = 0.25

    def __init__(self, action_space, observation_space, primitives):
        self.lo = np.array(action_space.low, dtype=float) * 0.99
        self.hi = np.array(action_space.high, dtype=float) * 0.99

    # ------------------------------------------------------------------ parse
    def _parse(self, state):
        d = {}
        r = state.get_object_from_name('robot')
        g = lambda o, f: float(state.get(o, f))
        d['r'] = np.array([g(r, 'x'), g(r, 'y')])
        d['rth'] = g(r, 'theta')
        d['arm'] = g(r, 'arm_joint')
        d['gap'] = g(r, 'finger_gap')
        h = state.get_object_from_name('hook')
        d['h'] = np.array([g(h, 'x'), g(h, 'y')])
        d['hth'] = g(h, 'theta')
        d['held'] = g(h, 'held') > 0.5
        self.hw = g(h, 'width')
        self.l1 = g(h, 'length_side1')
        self.l2 = g(h, 'length_side2')
        t = state.get_object_from_name('target_block')
        d['t'] = np.array([g(t, 'x'), g(t, 'y')])
        d['tth'] = g(t, 'theta')
        d['tw'] = g(t, 'width')
        d['th_'] = g(t, 'height')
        d['tpoly'] = rect_poly(d['t'][0], d['t'][1], d['tth'], d['tw'], d['th_'])
        obs = []
        for nm in sorted(state.get_object_names()):
            nm = str(nm)
            if nm in ('target_block', 'robot', 'hook'):
                continue
            o = state.get_object_from_name(nm)
            try:
                obs.append(rect_poly(g(o, 'x'), g(o, 'y'), g(o, 'theta'), g(o, 'width'), g(o, 'height')))
            except Exception:
                pass
        d['obs'] = obs
        return d

    # ---------------------------------------------------------------- geometry
    def hook_frame(self, C, phi):
        u = np.array([np.cos(phi), np.sin(phi)])
        n = np.array([np.sin(phi), -np.cos(phi)])
        return u, n

    def hook_end(self, d):
        u, n = self.hook_frame(d['h'], d['hth'])
        return d['h'] - self.l1 * u + 0.5 * self.hw * n, u, n

    def hook_polys(self, C, phi):
        u, n = self.hook_frame(C, phi)
        d1 = -u
        s1 = np.array([C, C + self.l1 * d1, C + self.l1 * d1 + self.hw * n, C + self.hw * n])
        s2 = np.array([C, C + self.l2 * n, C + self.l2 * n + self.hw * d1, C + self.hw * d1])
        return s1, s2

    def corner_from_robot(self, rpos, rth, arm):
        """Predicted hook corner & angle for held hook given robot pose."""
        c, s = np.cos(rth), np.sin(rth)
        L = self.L0 + np.array([arm - self.arm0, 0.0])
        return rpos + np.array([c * L[0] - s * L[1], s * L[0] + c * L[1]]), rth + self.dphi

    def robot_for_corner(self, C, phi, arm):
        rth = phi - self.dphi
        c, s = np.cos(rth), np.sin(rth)
        L = self.L0 + np.array([arm - self.arm0, 0.0])
        return C - np.array([c * L[0] - s * L[1], s * L[0] + c * L[1]]), rth

    # ------------------------------------------------------- collision model
    def robot_polys(self, x, y, th, arm, gap=0.32, base=True):
        u = np.array([np.cos(th), np.sin(th)])
        v = np.array([-u[1], u[0]])
        c = np.array([x, y])
        lat = gap / 2 + 0.03
        g0, g1 = arm - 0.03, arm + 0.23
        grip = np.array([c + g0 * u - lat * v, c + g1 * u - lat * v, c + g1 * u + lat * v, c + g0 * u + lat * v])
        out = [grip]
        if base:
            out.append(c + 0.24 * _CIRC)
        return out

    def pose_shapes(self, pose, held):
        x, y, th, arm = pose
        if held:
            C, phi = self.corner_from_robot(np.array([x, y]), th, arm)
            s1, s2 = self.hook_polys(C, phi)
            return [s1, s2] + self.robot_polys(x, y, th, arm, 0.12, base=False)
        return self.robot_polys(x, y, th, arm, 0.32, base=True)

    def path_free(self, p0, p1, obstacles, held, margin=0.03):
        """p = (x, y, th_unwrapped, arm); straight-line interpolation check."""
        if not obstacles:
            return True
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        dist = np.abs(p1 - p0) / np.array([0.02, 0.02, 0.012, 0.02])
        n = int(np.ceil(dist.max())) + 1
        boxes = [(P.min(0), P.max(0)) for P in obstacles]
        for i in range(n + 1):
            p = p0 + (p1 - p0) * i / n
            for S in self.pose_shapes(p, held):
                smin, smax = S.min(0), S.max(0)
                for P, (bmin, bmax) in zip(obstacles, boxes):
                    if np.any(smin > bmax + margin) or np.any(bmin > smax + margin):
                        continue
                    if poly_overlap(S, P, margin):
                        return False
        return True

    def choose_route(self, d, candidates, obstacles, held):
        """candidates: list of routes; route = list of (x,y,th,arm). Returns first free route
        (after unwrapping angles along the shortest direction) else the last candidate."""
        cur = (d['r'][0], d['r'][1], d['rth'], d['arm'])
        best = None
        for route in candidates:
            prev = np.array(cur, float)
            ok = True
            out = []
            for wp in route:
                wp = np.array(wp, float)
                wp[0] = np.clip(wp[0], self.X_MIN, self.X_MAX)
                wp[1] = np.clip(wp[1], self.Y_MIN, self.Y_MAX)
                wp[2] = prev[2] + wrap(wp[2] - prev[2])
                if ok and not self.path_free(prev, wp, obstacles, held):
                    ok = False
                out.append(wp)
                prev = wp
            if ok:
                return out, True
            best = out
        return best, False

    def choose_route_cost(self, d, candidates, obstacles, held, skip_last=0):
        """Unwrap each candidate, sort by estimated time, return first collision-free."""
        cur = np.array((d['r'][0], d['r'][1], d['rth'], d['arm']), float)
        items = []
        for cand in candidates:
            if isinstance(cand, tuple):
                route, nskip, marg = cand
            else:
                route, nskip, marg = cand, skip_last, 0.03
            prev = cur
            out = []
            cost = 0.0
            for wp in route:
                wp = np.array(wp, float)
                wp[0] = np.clip(wp[0], self.X_MIN, self.X_MAX)
                wp[1] = np.clip(wp[1], self.Y_MIN, self.Y_MAX)
                wp[2] = prev[2] + wrap(wp[2] - prev[2])
                cost += max(np.hypot(wp[0] - prev[0], wp[1] - prev[1]) / 0.0495,
                            abs(wp[2] - prev[2]) / 0.0647, abs(wp[3] - prev[3]) / 0.099) + 1.5
                out.append(wp)
                prev = wp
            items.append((cost, len(items), out, nskip, marg))
        items.sort(key=lambda z: (z[0], z[1]))
        for cost, _, out, nskip, marg in items:
            prev = cur
            ok = True
            for wp in out[:len(out) - nskip]:
                if not self.path_free(prev, wp, obstacles, held, margin=marg):
                    ok = False
                    break
                prev = wp
            if ok:
                return out, True
        return None, False

    # ----------------------------------------------------------------- control
    def act(self, d, pos, th=None, arm=None, grip=0.0, rotdir=0):
        pos = np.array([np.clip(pos[0], self.X_MIN, self.X_MAX), np.clip(pos[1], self.Y_MIN, self.Y_MAX)])
        dx, dy = pos - d['r']
        if th is None:
            dth = 0.0
        else:
            dth = wrap(th - d['rth'])
            if rotdir > 0 and dth < -0.02:
                dth += TWO_PI
            elif rotdir < 0 and dth > 0.02:
                dth -= TWO_PI
        darm = 0.0 if arm is None else arm - d['arm']
        a = np.array([dx, dy, dth, darm, grip], dtype=float)
        # scale translation to keep direction
        # proportional scaling of motion dims (straight line in x,y,theta,arm)
        ratio = np.max(np.abs(a[:4]) / self.hi[:4])
        if ratio > 1:
            a[:4] /= ratio
        return np.clip(a, self.lo, self.hi)

    def at(self, d, pos, th=None, arm=None, tol=0.004, ttol=0.01):
        pos = np.array([np.clip(pos[0], self.X_MIN, self.X_MAX), np.clip(pos[1], self.Y_MIN, self.Y_MAX)])
        ok = np.hypot(*(pos - d['r'])) < tol
        if th is not None:
            ok = ok and abs(wrap(th - d['rth'])) < ttol
        if arm is not None:
            ok = ok and abs(arm - d['arm']) < 0.005
        return ok

    # -------------------------------------------------------------- interface
    def reset(self, state, info):
        self._rtype = None
        self.phase = 'init'
        self.k = 0
        self.wp = None
        self.L0 = None
        self.fail = 0
        self.npush = 0
        self.nreplan = 0

    def set_phase(self, p):
        self.phase = p
        self.k = 0
        self.wp = None

    def get_action(self, state):
        d = self._parse(state)
        self.k += 1
        try:
            a = self._policy(d)
        except Exception:
            a = np.zeros(5)
        return np.clip(np.asarray(a, dtype=float), self.lo, self.hi)

    def _policy(self, d):
        ph = self.phase
        if ph == 'init':
            if d['held']:
                self._record_grasp(d)
                self.set_phase('orient')
            else:
                self.set_phase('plan_grasp')
            return self._policy(d)

        if ph == 'plan_grasp':
            E, u, n = self.hook_end(d)
            G = E - 0.30 * u
            pre = E - 0.55 * u
            if (pre[0] < self.X_MIN + 0.01 or pre[1] > self.Y_MAX - 0.01 or G[1] > self.Y_MAX - 0.01
                    or pre[1] < 0.3) and self.npush < 6:
                # push hook's short side rightwards near its bottom (translates + rotates ccw)
                self.npush += 1
                shift = max(self.X_MIN + 0.05 - pre[0], 0) + self.PUSH_EXTRA
                s1, s2 = self.hook_polys(d['h'], d['hth'])
                s2minx = s2[:, 0].min()
                s2miny = s2[:, 1].min()
                yc = max(s2miny + 0.12, self.Y_MIN + 0.01)
                x0 = s2minx - 0.24 - 0.06
                ylow = self.Y_MIN + 0.02
                if d['r'][0] <= x0:
                    self.wp = [(np.array([d['r'][0], ylow]), None, 0),
                               (np.array([x0, ylow]), np.pi, 1)]
                else:
                    self.wp = [(np.array([x0, ylow]), None, 0),
                               (np.array([x0, ylow]), np.pi, 1)]
                self.wp += [
                           (np.array([x0, yc]), np.pi, 1),
                           (np.array([x0 + 0.06 + shift, yc]), np.pi, 1),
                           (np.array([x0 + shift - 0.12, yc]), np.pi, 1)]
                self.phase = 'push'
                self.k = 0
                return self._policy(d)
            ylow = 0.5
            th = d['hth']
            cur = wrap(d['rth'])
            rd = -1 if cur >= th else 1
            sc = self._grasp_shortcut(d, pre, th)
            if sc is not None:
                self.wp = sc
                self.phase = 'grasp_go'
                self.k = 0
                return self._policy(d)
            self.wp = [(np.array([d['r'][0], max(min(d['r'][1], ylow), self.Y_MIN + 0.02)]), None, 0),
                       (np.array([max(pre[0], self.X_MIN), ylow]), None, 0),
                       (np.array([max(pre[0], self.X_MIN), ylow]), th, rd),
                       (np.array([max(pre[0], self.X_MIN), min(pre[1], self.Y_MAX)]), th, 0)]
            self.phase = 'grasp_go'
            self.k = 0
            return self._policy(d)

        if ph == 'push':
            while self.wp:
                p, th, rd = self.wp[0]
                if self.at(d, p, th) or self.k > 60:
                    self.wp.pop(0)
                    self.k = 0
                    continue
                return self.act(d, p, th, arm=0.24, grip=0.02, rotdir=rd)
            self.set_phase('plan_grasp')
            return self._policy(d)

        if ph == 'grasp_go':
            while self.wp:
                p, th, rd = self.wp[0]
                if self.at(d, p, th) or self.k > 60:
                    self.wp.pop(0)
                    self.k = 0
                    continue
                return self.act(d, p, th, arm=0.24, grip=0.02, rotdir=rd)
            self.set_phase('grasp_pre')
            return self._policy(d)

        if ph == 'grasp_pre':
            E, u, n = self.hook_end(d)
            pre = E - 0.55 * u
            th = d['hth']
            if self.at(d, pre, th, tol=0.01, ttol=0.02) or self.k > 40:
                self.set_phase('grasp_in')
                return self._policy(d)
            return self.act(d, pre, th, arm=0.24, grip=0.02)

        if ph == 'grasp_in':
            E, u, n = self.hook_end(d)
            G = E - 0.30 * u
            th = d['hth']
            if self.at(d, G, th, tol=0.005, ttol=0.02) or self.k > 30:
                self.set_phase('grasp_close')
                return self._policy(d)
            return self.act(d, G, th, arm=0.24, grip=0.02)

        if ph == 'grasp_close':
            if d['held'] and self.k > 3:
                self._record_grasp(d)
                self.set_phase('orient')
                return self._policy(d)
            if self.k > 14:
                self.fail += 1
                self.wp = None
                self.set_phase('grasp_back')
                return self._policy(d)
            return self.act(d, d['r'], None, arm=0.24, grip=-0.02)

        if ph == 'grasp_back':
            E, u, n = self.hook_end(d)
            pre = E - 0.6 * u
            if self.k > 12:
                self.set_phase('plan_grasp')
                return self._policy(d)
            return self.act(d, pre, None, arm=0.24, grip=0.02)

        if ph == 'orient':
            return self._orient(d)
        if ph == 'man':
            return self._man(d)
        if ph == 'over':
            return self._over(d)
        if ph == 'pull':
            return self._pull(d)
        return np.zeros(5)

    def _record_grasp(self, d):
        c, s = np.cos(d['rth']), np.sin(d['rth'])
        v = d['h'] - d['r']
        self.L0 = np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
        self.arm0 = d['arm']
        self.dphi = wrap(d['hth'] - d['rth'])

    # --------------------------------------------------------- manipulation
    def _plan_lift(self, d):
        tp = d['tpoly']
        tl = tp[:, 0].min()
        tt = tp[:, 1].max()
        margin = 0.06
        best = None
        for alpha in np.linspace(0, 0.6, 31):
            phi = np.pi / 2 + alpha
            s1, s2 = self.hook_polys(np.zeros(2), phi)
            maxx = max(s1[:, 0].max(), s2[:, 0].max())
            cx = tl - margin - maxx
            rpos, rth = self.robot_for_corner(np.array([cx, 0.0]), phi, 0.24)
            if rpos[0] >= self.X_MIN + 0.02:
                best = (alpha, phi, cx)
                break
        if best is None:
            phi = np.pi / 2 + 0.6
            rpos, rth = self.robot_for_corner(np.zeros(2), phi, 0.24)
            cx = self.X_MIN + 0.02 - rpos[0]
            best = (0.6, phi, cx)
        alpha, phi, cx = best
        u, n = self.hook_frame(None, phi)
        inner_off = self.hw * (-u)
        cy_need = tt + 0.05 - inner_off[1]
        self.lift_phi = phi
        self.lift_cx = cx
        self.lift_cy = cy_need
        s1, s2 = self.hook_polys(np.array([cx, 0.0]), phi)
        self.lift_minx = min(s1[:, 0].min(), s2[:, 0].min())

    def _robot_target(self, C, phi):
        """Robot pose (pos, th, arm) placing corner at C, using arm to respect y limit."""
        for arm in np.linspace(0.24, 0.48, 13):
            rpos, rth = self.robot_for_corner(C, phi, arm)
            if rpos[1] <= self.Y_MAX:
                return rpos, rth, arm
        return rpos, rth, 0.48

    def _cy_low(self, phi):
        C, _ = self.corner_from_robot(np.array([0.0, self.Y_MIN + 0.01]), phi - self.dphi, 0.24)
        return C[1]

    def _swept_target(self, d, beta, shrink=0.0):
        tp = d['tpoly']
        c = tp.mean(0)
        tp = c + (tp - c) * (1 - shrink)
        tb = tp[:, 1].min()
        drop = max(tb - (self.WALL_Y + 0.03), 0.0)
        D = np.array([np.tan(beta) * drop, -drop])
        return convex_hull(np.vstack([tp, tp + D])), D

    def _blocking(self, d, beta=0.0):
        """Obstructions intersecting the target's swept path when pulled along beta."""
        S, D = self._swept_target(d, beta)
        res = []
        for P in d['obs']:
            if poly_overlap(S, P, -0.03):
                res.append(P)
        return res

    def _choose_beta(self, d):
        tp = d['tpoly']
        for beta in (0.0, 0.2, 0.4, 0.6, 0.8):
            S, D = self._swept_target(d, beta)
            if S[:, 0].max() > 3.44:
                break
            if not self._blocking(d, beta):
                return beta
        return None

    def _plan(self, d):
        """Decide the next maneuver; returns list of (C, phi) waypoints and final mode."""
        self._plan_lift(d)
        phi0 = np.pi / 2
        s1, s2 = self.hook_polys(np.zeros(2), phi0)
        hminx = min(s1[:, 0].min(), s2[:, 0].min())
        hmaxx = max(s1[:, 0].max(), s2[:, 0].max())
        barr = s1[:, 0].max()  # right face of bar relative to corner
        beta = self._choose_beta(d)
        blk = []
        if beta is None:
            blk = self._blocking(d, 0.0) if self.nsweep < 2 else []
            beta = 0.0
        self.pull_beta = beta
        if blk:
            tp = d['tpoly']
            tl, tr, tb = tp[:, 0].min(), tp[:, 0].max(), tp[:, 1].min()
            pminx = min(P[:, 0].min() for P in blk)
            pmaxx = max(P[:, 0].max() for P in blk)
            pminy = min(P[:, 1].min() for P in blk)
            wext = max(P[:, 0].max() - P[:, 0].min() for P in blk)
            cy = tb - 0.03
            cy = max(cy, pminy + 0.06)
            cy = max(cy, self._cy_low(phi0))
            right_ok = tr + 0.08 + wext < 3.42
            left_target = self.lift_minx - 0.04
            left_ok = left_target - wext > 0.1 and left_target - hminx >= self._min_cx(phi0)
            self.nsweep += 1
            ocx = np.mean([P[:, 0].mean() for P in blk])
            go_left = left_ok and (ocx < tp[:, 0].mean() or not right_ok)
            if not go_left:
                cx0 = pminx - 0.04 - hmaxx
                cx1 = tr + 0.08 - barr
            else:
                cx0 = pmaxx + 0.04 - hminx
                cx1 = left_target - hminx
            cx0 = max(cx0, self._min_cx(phi0))
            cl = self._cy_low(phi0)
            wps = [(np.array([cx0, cl]), phi0), (np.array([cx0, cy]), phi0),
                   (np.array([cx1, cy]), phi0), (np.array([cx1, cl]), phi0)]
            return wps, 'sweep'
        phi = self.lift_phi
        cl = self._cy_low(phi)
        wps = [(np.array([self.lift_cx, cl]), phi), (np.array([self.lift_cx, self.lift_cy]), phi)]
        return wps, 'over'

    def _min_cx(self, phi):
        rpos, rth = self.robot_for_corner(np.zeros(2), phi, 0.24)
        return self.X_MIN + 0.01 - rpos[0]

    def _grasp_shortcut(self, d, pre, th):
        obst = self._obstacles(d, walls=False, hook=True, target=True)
        px = float(np.clip(pre[0], self.X_MIN, self.X_MAX))
        py = float(np.clip(pre[1], self.Y_MIN, self.Y_MAX))
        rx, ry = float(d['r'][0]), float(d['r'][1])
        cur = float(d['rth'])
        arm = float(d['arm'])
        dth = wrap(th - cur)
        opts = [dth, dth - TWO_PI * np.sign(dth)] if abs(dth) > 1e-3 else [dth]
        cands = []
        for dd in opts:
            ang = cur + dd
            rd = 1 if dd > 0 else -1
            cands.append(([(rx, ry, cur, arm), (px, py, ang, 0.24)], [((px, py), th, rd)]))
            for yv in (0.5, 0.38, self.Y_MIN + 0.02):
                yl = min(ry, yv)
                pts = [(rx, yl), (px, yl), (px, py)]
                # fraction of rotation done in each of the 3 segments
                for split in ((0, 1, 0), (0, 0, 1), (0, .5, .5), (1 / 3, 1 / 3, 1 / 3), (0.5, 0.5, 0)):
                    route = [(rx, ry, cur, arm)]
                    wps = []
                    a = cur
                    for p, f in zip(pts, split):
                        a = a + f * dd
                        route.append((p[0], p[1], a, 0.24))
                        wps.append((p, None if f == 0 and abs(a - cur) < 1e-9 else wrap(a), rd if f > 0 else 0))
                    cands.append((route, wps))
        best = None
        for route, wps in cands:
            ok = all(self.path_free(route[i], route[i + 1], obst, False, margin=-0.03 if i == 0 else 0.02)
                     for i in range(len(route) - 1))
            if ok:
                cost = sum(max(np.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1]) / 0.0495,
                               abs(route[i + 1][2] - route[i][2]) / 0.0647) for i in range(len(route) - 1))
                if best is None or cost < best[0]:
                    best = (cost, wps)
        if best is None:
            return None
        return [(np.array(p, float), t, r) for p, t, r in best[1]]

    def _obstacles(self, d, walls=False, hook=False, target=True):
        obs = list(d['obs'])
        if target:
            obs.append(d['tpoly'])
        if hook:
            s1, s2 = self.hook_polys(d['h'], d['hth'])
            obs += [s1, s2]
        if walls:
            obs.append(np.array([[-1.0, -1.0], [4.5, -1.0], [4.5, 0.015], [-1.0, 0.015]]))
            obs.append(np.array([[-1.0, -1.0], [0.02, -1.0], [0.02, 4.5], [-1.0, 4.5]]))
            obs.append(np.array([[3.48, -1.0], [4.5, -1.0], [4.5, 4.5], [3.48, 4.5]]))
        return obs

    def _start_plan(self, d):
        wps, after = self._plan(d)
        poses = []
        for C, phi in wps:
            rpos, rth, arm = self._robot_target(C, phi)
            poses.append(np.array([rpos[0], rpos[1], rth, arm]))
        low = self.Y_MIN + 0.01
        cons = [np.array([d['r'][0], low, d['rth'], 0.24]),
                np.array([poses[0][0], low, d['rth'], 0.24]),
                np.array([poses[0][0], low, poses[0][2], 0.24])] + poses
        if after == 'sweep':
            cands = [poses, (poses[1:], len(poses) - 2, -0.02)]
        else:
            cands = [poses[1:], poses]
        cx, cyr, cth, carm = d['r'][0], d['r'][1], d['rth'], d['arm']
        p0 = poses[0]
        top = min(cyr, p0[1])
        for yv in np.arange(low, top + 1e-9, 0.08):
            cands.append([np.array([cx, yv, cth, carm]), np.array([p0[0], yv, p0[2], p0[3]])] + poses[1:])
            cands.append([np.array([cx, yv, cth, carm]), np.array([p0[0], yv, cth, carm]),
                          np.array([p0[0], yv, p0[2], p0[3]])] + poses[1:])
            cands.append([np.array([cx, yv, p0[2], p0[3]]), np.array([p0[0], yv, p0[2], p0[3]])] + poses[1:])
        route, ok = self.choose_route_cost(d, cands, self._obstacles(d), True,
                                           skip_last=(len(poses) - 1 if after == 'sweep' else 0))
        if not ok:
            route = cons
        self.man = [np.array(p, float) for p in route]
        self.after = after

    def _orient(self, d):
        self.nsweep = 0
        self.nreplan = 0
        self._start_plan(d)
        self.set_phase('man')
        return self._policy(d)

    def _man(self, d):
        while self.man:
            p = self.man[0]
            if self.at(d, p[:2], p[2], arm=p[3], tol=0.006) or self.k > 70:
                self.man.pop(0)
                self.k = 0
                continue
            return self.act(d, p[:2], p[2], arm=p[3], grip=-0.02)
        if self.after == 'sweep':
            self._start_plan(d)
            self.k = 0
            return self._man(d)
        self.set_phase('over')
        return self._policy(d)

    def _bar_right_x_at(self, C, phi, y):
        s1, s2 = self.hook_polys(C, phi)
        a = s1[3]
        b = s1[2]
        if abs(a[1] - b[1]) < 1e-9:
            return max(a[0], b[0])
        t = np.clip((y - a[1]) / (b[1] - a[1]), 0, 1)
        return a[0] + t * (b[0] - a[0])

    def _over_cx(self, d, phi, cy):
        tp = d['tpoly']
        gap = np.inf
        for p in tp:
            bx = self._bar_right_x_at(np.array([0.0, cy]), phi, p[1])
            gap = min(gap, p[0] - bx)
        return gap - 0.025

    def _over(self, d):
        cy = self.lift_cy
        cx = self._over_cx(d, self.lift_phi, cy)
        if self.k <= 1 or getattr(self, 'over_cx0', None) is None:
            self.over_cx0 = cx
        cx = min(cx, self.over_cx0 + 0.06)
        C = np.array([cx, cy])
        rpos, rth, arm = self._robot_target(C, self.lift_phi)
        if self.at(d, rpos, rth, arm=arm, tol=0.01) or self.k > 60:
            self.pull_x = cx
            self.pull_C0 = None
            self.set_phase('pull')
            return self._policy(d)
        return self.act(d, rpos, rth, arm=arm, grip=-0.02)

    def _pull(self, d):
        Ccur, phic = self.corner_from_robot(d['r'], d['rth'], d['arm'])
        if self.pull_C0 is None:
            self.pull_C0 = Ccur.copy()
            self.pull_best = np.inf
            self.pull_last = 0
        tb = float(np.min(d['tpoly'][:, 1]))
        if tb < self.pull_best - 0.004:
            self.pull_best = tb
            self.pull_last = self.k
        if self.k - self.pull_last > 25 and self.nreplan < 4:
            self.nreplan += 1
            self.over_cx0 = None
            self.nsweep = 0
            self._start_plan(d)
            self.set_phase('man')
            return self._policy(d)
        beta = self.pull_beta
        if beta <= 1e-6:
            cy = Ccur[1] - 0.05
            cx = self._over_cx(d, self.lift_phi, cy)
            cx = min(cx, self.pull_x + 0.1)
            C = np.array([cx, cy])
        else:
            dr = np.array([np.sin(beta), -np.cos(beta)])
            s = float((Ccur - self.pull_C0) @ dr)
            C = self.pull_C0 + (s + 0.07) * dr
        rpos, rth, arm = self._robot_target(C, self.lift_phi)
        return self.act(d, rpos, rth, arm=arm, grip=-0.02)
