import math
import numpy as np

TWO_PI = 2.0 * math.pi
TOL_TH = 0.1396


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def rot(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def _rect_sdf(px, py, hx, hy):
    dx = abs(px) - hx
    dy = abs(py) - hy
    ox = dx if dx > 0.0 else 0.0
    oy = dy if dy > 0.0 else 0.0
    return math.hypot(ox, oy) + min(max(dx, dy), 0.0)


class GeneratedApproach:
    """Quasi-static pusher for the 2-D T-block alignment task."""

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.amax = 0.0499
        self.R = 0.1
        self.gain = 0.87          # block travel per unit robot travel (head-on)
        self.pen = 0.14           # servo target depth behind the contact face
        self.stand = 0.19         # stand-off distance in front of the face
        self.lo = 0.0
        self.hi = 5.0
        self.h = 0.05
        self.N = int(round((self.hi - self.lo) / self.h)) + 1
        self.gx = self.lo + np.arange(self.N) * self.h
        self.infl = 0.04
        self.C2K = 0.2606
        self.nudge = True
        self.fine = True
        self.fine_mag = 0.16
        self.exec_k = 0.5
        self.rp_k = 0.06
        self.fine_min = 0.014
        self.rp_k = 0.06
        self.rp_max = 9
        self.nerr_S = 6.0
        self.Smin_end = 0.035
        self.nerr_ell = 0.0
        self.nerr_lim = 0.0
        self.stage_thresh = -9.0
        self.stage_d = 0.55
        self.navw = 0.5
        self.vguard = True
        self.vmargin = 0.05
        self.nerr_end = 0.0
        self.box = 0.55
        self.end_div = 4.0
        self.end_floor = 0.004
        self.endmode = False
        self.end_lim = 0.016
        self.rec_nerr = 6.0
        self.Smin = 0.10

    # ------------------------------------------------------------------
    def reset(self, state, info=None):
        s = np.asarray(state, dtype=np.float64)
        self.w = float(s[12])
        self.Lh = float(s[13])
        self.Lv = float(s[14])
        self.R = float(s[28])
        Ab = self.Lh * self.w
        As = self.w * self.Lv
        self.ycom = -(Ab * (self.w / 2.0) + As * (self.w + self.Lv / 2.0)) / (Ab + As)
        self.c2 = self.C2K * self.Lh * self.Lv
        self.ell = 0.5 * self.Lh
        # Block local frame: origin sits on the TOP edge of the cross bar.
        self.bar_half = (self.Lh / 2.0, self.w / 2.0)
        self.bar_c = (0.0, -self.w / 2.0)
        self.stem_half = (self.w / 2.0, self.Lv / 2.0)
        self.stem_c = (0.0, -(self.w + self.Lv / 2.0))
        self.cands = self._make_candidates()
        self.goal_adj = self._adjust_goal(float(s[29]), float(s[30]), float(s[31]))
        gv = self._violation(*self.goal_adj)
        self.vallow = max(gv, 0.0) + self.vmargin
        self.stage_on = gv > -self.stage_thresh
        d = np.array([2.5 - self.goal_adj[0], 2.5 - self.goal_adj[1]])
        nd = float(np.linalg.norm(d))
        self.stage_dir = d / nd if nd > 1e-6 else np.zeros(2)
        self.phase = "plan"
        self.push = None
        self.since = 0
        self.nav_steps = 0
        self.tick = 0
        self.ban = {}
        self.path = None
        self.grid_key = None
        self.endmode = False
        self.trav = 0.0
        self.S_exec = 1e9
        self._ctx = None
        self.S = 0.3
        self.Ebest = 1e18
        self.stall = 0
        self.mode = 0
        self.recover = 0
        self.seg0 = None
        self.trav = 0.0
        self.S_exec = 0.3
        self.ell0 = self.ell
        self.rng = np.random.RandomState(12345)

    # ------------------------------------------------------------------
    def _corners(self):
        w, Lh, Lv = self.w, self.Lh, self.Lv
        return [(-Lh / 2, 0.0), (Lh / 2, 0.0), (Lh / 2, -w), (w / 2, -w),
                (w / 2, -(w + Lv)), (-w / 2, -(w + Lv)), (-w / 2, -w),
                (-Lh / 2, -w)]

    def _violation(self, gx, gy, gt):
        c, s = math.cos(gt), math.sin(gt)
        v = 0.0
        for (x, y) in self._corners():
            px = gx + c * x - s * y
            py = gy + s * x + c * y
            v = max(v, self.lo - px, px - self.hi, self.lo - py, py - self.hi)
        return v

    def _adjust_goal(self, gx, gy, gt):
        if not self.nudge:
            return (gx, gy, gt)
        base = self._violation(gx, gy, gt)
        if base <= 0.0:
            return (gx, gy, gt)
        best = (base, 0.0, 0.0, 0.0)
        for dt in np.linspace(-0.09, 0.09, 13):
            for dx in np.linspace(-0.022, 0.022, 12):
                for dy in np.linspace(-0.022, 0.022, 12):
                    v = self._violation(gx + dx, gy + dy, gt + dt)
                    cost = max(v, 0.0) + 1e-3 * (abs(dx) + abs(dy) + 0.2 * abs(dt))
                    if cost < best[0]:
                        best = (cost, dx, dy, dt)
        return (gx + best[1], gy + best[2], gt + best[3])

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _sdf(self, x, y):
        a = _rect_sdf(x - self.bar_c[0], y - self.bar_c[1],
                      self.bar_half[0], self.bar_half[1])
        b = _rect_sdf(x - self.stem_c[0], y - self.stem_c[1],
                      self.stem_half[0], self.stem_half[1])
        return a if a < b else b

    def _sdf_grad(self, x, y):
        e = 1e-4
        f0 = self._sdf(x, y)
        gx = (self._sdf(x + e, y) - f0) / e
        gy = (self._sdf(x, y + e) - f0) / e
        n = math.hypot(gx, gy)
        if n < 1e-9:
            return 1.0, 0.0
        return gx / n, gy / n

    def _feasible(self, c_loc, n_loc):
        p = c_loc + n_loc * (self.R + 0.005)
        if self._sdf(p[0], p[1]) < self.R - 0.003:
            return False
        p2 = c_loc + n_loc * (self.R + self.infl + 0.02)
        if self._sdf(p2[0], p2[1]) < self.R + self.infl + 0.015:
            return False
        p3 = c_loc + n_loc * (self.R + self.stand)
        if self._sdf(p3[0], p3[1]) < self.R + self.infl + 0.015:
            return False
        return True

    def _make_candidates(self):
        w, Lh, Lv, R = self.w, self.Lh, self.Lv, self.R
        hw = w / 2.0
        raw = []
        for x in np.linspace(-Lh / 2 + 0.02, Lh / 2 - 0.02, 13):
            raw.append((np.array([x, 0.0]), np.array([0.0, 1.0])))
        lim = hw + R + 0.05
        if lim < Lh / 2 - 0.02:
            for x in np.linspace(lim, Lh / 2 - 0.02, 5):
                raw.append((np.array([x, -w]), np.array([0.0, -1.0])))
                raw.append((np.array([-x, -w]), np.array([0.0, -1.0])))
        raw.append((np.array([-Lh / 2, -hw]), np.array([-1.0, 0.0])))
        raw.append((np.array([Lh / 2, -hw]), np.array([1.0, 0.0])))
        ylo = -(w + Lv) + 0.06
        yhi = -w - R - 0.07
        if yhi > ylo:
            for y in np.linspace(ylo, yhi, 7):
                raw.append((np.array([hw, y]), np.array([1.0, 0.0])))
                raw.append((np.array([-hw, y]), np.array([-1.0, 0.0])))
        raw.append((np.array([0.0, -(w + Lv)]), np.array([0.0, -1.0])))
        return [(c, n, -n) for c, n in raw if self._feasible(c, n)]

    # ------------------------------------------------------------------
    # Push model
    # ------------------------------------------------------------------
    def _cost(self, dpx, dpy, dth):
        if self.endmode:
            b = self.box
            a1 = abs(dpx) / 0.03 - b
            a2 = abs(dpy) / 0.03 - b
            a3 = abs(dth) / TOL_TH - b
            r = 0.0
            if a1 > 0.0:
                r += a1 * a1
            if a2 > 0.0:
                r += a2 * a2
            if a3 > 0.0:
                r += a3 * a3
            return r
        return dpx * dpx + dpy * dpy + (self.ell * dth) ** 2

    def _predict(self, P, th, c_loc, u_loc, S, nsub=6):
        ds = self.gain * S / nsub
        c2 = self.c2
        yc = self.ycom
        ct, st = math.cos(th), math.sin(th)
        gx = P[0] - (-st * yc)
        gy = P[1] - (ct * yc)
        ax = c_loc[0] - 0.0
        ay = c_loc[1] - yc
        ux, uy = u_loc[0], u_loc[1]
        for _ in range(nsub):
            ct, st = math.cos(th), math.sin(th)
            rx = ct * ax - st * ay
            ry = st * ax + ct * ay
            wx = ct * ux - st * uy
            wy = st * ux + ct * uy
            m = rx * wy - ry * wx
            th += ds * m / (c2 + m * m)
            f = ds / (1.0 + m * m / c2)
            gx += wx * f
            gy += wy * f
        ct, st = math.cos(th), math.sin(th)
        return gx + (-st * yc), gy + (ct * yc), th

    def _best_travel(self, P, th, c_loc, u_loc, S, Pg, thg, nsub=24):
        c2 = self.c2
        yc = self.ycom
        ct, st = math.cos(th), math.sin(th)
        gx = P[0] + st * yc
        gy = P[1] - ct * yc
        ax = c_loc[0]
        ay = c_loc[1] - yc
        ux, uy = u_loc[0], u_loc[1]
        ds = self.gain * S / nsub
        step = S / nsub
        bestc = self._cost(Pg[0] - P[0], Pg[1] - P[1], wrap(thg - th))
        bests = 0.0
        trav = 0.0
        for _ in range(nsub):
            ct, st = math.cos(th), math.sin(th)
            rx = ct * ax - st * ay
            ry = st * ax + ct * ay
            wx = ct * ux - st * uy
            wy = st * ux + ct * uy
            m = rx * wy - ry * wx
            th += ds * m / (c2 + m * m)
            f = ds / (1.0 + m * m / c2)
            gx += wx * f
            gy += wy * f
            trav += step
            ct2, st2 = math.cos(th), math.sin(th)
            cc = self._cost(Pg[0] - (gx - st2 * yc), Pg[1] - (gy + ct2 * yc),
                            wrap(thg - th))
            if cc < bestc:
                bestc = cc
                bests = trav
        return bests

    # ------------------------------------------------------------------
    def _rank(self, P, th, q, Pg, thg):
        dpx, dpy = Pg[0] - P[0], Pg[1] - P[1]
        dth0 = wrap(thg - th)
        E0 = self._cost(dpx, dpy, dth0)
        mag = math.hypot(dpx, dpy) + self.ell * abs(dth0)
        S = float(min(max(1.2 * mag, self.Smin), 0.55))
        M = rot(th)
        out = []
        for idx, (c_loc, n_loc, u_loc) in enumerate(self.cands):
            if self.ban.get(idx, 0) > self.tick:
                continue
            px, py, th1 = self._predict(P, th, c_loc, u_loc, S)
            gain = E0 - self._cost(Pg[0] - px, Pg[1] - py, wrap(thg - th1))
            if gain <= 1e-9:
                continue
            if self.vguard and self._violation(px, py, th1) > self.vallow:
                continue
            cw = P + M @ c_loc
            n = M @ n_loc
            far = cw + n * (self.R + self.stand)
            if (far[0] < self.lo + self.R + 0.02 or far[0] > self.hi - self.R - 0.02
                    or far[1] < self.lo + self.R + 0.02
                    or far[1] > self.hi - self.R - 0.02):
                continue
            if idx == self.push and self.phase == "push":
                nav = 0.0
            else:
                nav = float(np.linalg.norm(far - q)) + 0.25
                if self._blocked(P, th, q, far):
                    nav += 0.8
            out.append((gain / (S + self.navw * nav + 0.05), idx))
        out.sort(reverse=True)
        return out, S

    def _blocked(self, P, th, a, b):
        Mi = rot(-th)
        al = Mi @ (a - P)
        bl = Mi @ (b - P)
        L = float(np.linalg.norm(bl - al))
        n = int(L / 0.05) + 2
        lim = self.R + self.infl
        for t in np.linspace(0.0, 1.0, n):
            x = al[0] + (bl[0] - al[0]) * t
            y = al[1] + (bl[1] - al[1]) * t
            if self._sdf(x, y) < lim:
                return True
        return False

    # ------------------------------------------------------------------
    # Grid path planning (world frame; block is static while navigating)
    # ------------------------------------------------------------------
    def _build_grid(self, P, th, target):
        N = self.N
        ct, st = math.cos(-th), math.sin(-th)
        X = self.gx[None, :] - P[0]
        Y = self.gx[:, None] - P[1]
        lx = ct * X - st * Y
        ly = st * X + ct * Y
        d1 = self._sdf_vec(lx - self.bar_c[0], ly - self.bar_c[1], self.bar_half)
        d2 = self._sdf_vec(lx - self.stem_c[0], ly - self.stem_c[1], self.stem_half)
        free = np.minimum(d1, d2) > (self.R + self.infl)
        m = self.R + 0.006
        inb = (self.gx >= self.lo + m) & (self.gx <= self.hi - m)
        free &= inb[None, :] & inb[:, None]
        ti = min(max(int(round((target[1] - self.lo) / self.h)), 0), N - 1)
        tj = min(max(int(round((target[0] - self.lo) / self.h)), 0), N - 1)
        free[ti, tj] = True
        dist = np.full((N, N), -1, dtype=np.int32)
        dist[ti, tj] = 0
        cur = np.zeros((N, N), dtype=bool)
        cur[ti, tj] = True
        for it in range(1, 500):
            nb = np.zeros((N, N), dtype=bool)
            nb[1:, :] |= cur[:-1, :]
            nb[:-1, :] |= cur[1:, :]
            nb[:, 1:] |= cur[:, :-1]
            nb[:, :-1] |= cur[:, 1:]
            nb[1:, 1:] |= cur[:-1, :-1]
            nb[1:, :-1] |= cur[:-1, 1:]
            nb[:-1, 1:] |= cur[1:, :-1]
            nb[:-1, :-1] |= cur[1:, 1:]
            nb &= free
            nb &= (dist < 0)
            if not nb.any():
                break
            dist[nb] = it
            cur = nb
        self.dist = dist
        self.free = free
        self.tcell = (ti, tj)

    @staticmethod
    def _sdf_vec(x, y, half):
        dx = np.abs(x) - half[0]
        dy = np.abs(y) - half[1]
        return (np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
                + np.minimum(np.maximum(dx, dy), 0.0))

    def _plan_path(self, P, th, q, target):
        """Return list of world waypoints from q to target, or None."""
        self._build_grid(P, th, target)
        N = self.N
        i = min(max(int(round((q[1] - self.lo) / self.h)), 0), N - 1)
        j = min(max(int(round((q[0] - self.lo) / self.h)), 0), N - 1)
        if self.dist[i, j] < 0:
            # snap to nearest reachable cell within a small window
            best = None
            bd = 1e18
            for di in range(-4, 5):
                for dj in range(-4, 5):
                    a, b = i + di, j + dj
                    if 0 <= a < N and 0 <= b < N and self.dist[a, b] >= 0:
                        dd = di * di + dj * dj
                        if dd < bd:
                            bd = dd
                            best = (a, b)
            if best is None:
                return None
            i, j = best
        cells = [(i, j)]
        for _ in range(2000):
            ci, cj = cells[-1]
            cd = self.dist[ci, cj]
            if cd == 0:
                break
            nxt = None
            nb = cd
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    a, b = ci + di, cj + dj
                    if a < 0 or b < 0 or a >= N or b >= N:
                        continue
                    d = self.dist[a, b]
                    if d >= 0 and d < nb:
                        nb = d
                        nxt = (a, b)
            if nxt is None:
                return None
            cells.append(nxt)
        pts = [np.array([self.lo + c[1] * self.h, self.lo + c[0] * self.h])
               for c in cells]
        pts.append(np.asarray(target, dtype=np.float64))
        # line-of-sight shortcut
        out = []
        k = 0
        n = len(pts)
        while k < n - 1:
            best = k + 1
            for t in range(n - 1, k, -1):
                if not self._blocked(P, th, pts[k], pts[t]):
                    best = t
                    break
            out.append(pts[best])
            k = best
        if not out:
            out = [np.asarray(target, dtype=np.float64)]
        out[-1] = np.asarray(target, dtype=np.float64)
        return out

    # ------------------------------------------------------------------
    def _far_point(self, P, th, idx):
        c_loc, n_loc, u_loc = self.cands[idx]
        M = rot(th)
        cw = P + M @ c_loc
        n = M @ n_loc
        return cw + n * (self.R + self.stand)

    def _replan(self, P, th, q, Pg, thg):
        order, S = self._rank(P, th, q, Pg, thg)
        self.S = S
        self._ctx = (P.copy(), th, Pg.copy(), thg, S)
        if not order:
            self.ban = {}
            order, S = self._rank(P, th, q, Pg, thg)
        for _, idx in order[:6]:
            if idx == self.push and self.phase == "push":
                self.push = idx
                self.since = 0
                self._set_exec(idx)
                return True
            far = self._far_point(P, th, idx)
            path = self._plan_path(P, th, q, far)
            if path is None:
                self.ban[idx] = self.tick + 60
                continue
            self.push = idx
            self._set_exec(idx)
            self.path = path
            self.pi = 0
            self.phase = "nav"
            self.nav_steps = 0
            self.since = 0
            return True
        if order:
            self.push = order[0][1]
            self._set_exec(self.push)
            self.path = [self._far_point(P, th, self.push)]
            self.pi = 0
            self.phase = "nav"
            self.nav_steps = 0
            self.since = 0
            return True
        return False

    def _set_exec(self, idx):
        self.trav = 0.0
        self.S_exec = 1e9
        if not self.endmode:
            return
        P, th, Pg, thg, S = self._ctx
        c_loc, n_loc, u_loc = self.cands[idx]
        s = self._best_travel(P, th, c_loc, u_loc, S, Pg, thg)
        self.S_exec = max(s, 0.012)

    # ------------------------------------------------------------------
    def get_action(self, state):
        try:
            a = self._get_action(state)
            a = np.asarray(a, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(a)):
                return np.zeros(2)
            return np.clip(a, -self.amax, self.amax)
        except Exception:
            try:
                return self._fallback(state)
            except Exception:
                return np.zeros(2)

    def _fallback(self, state):
        s = np.asarray(state, dtype=np.float64)
        d = s[0:2] - s[16:18]
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return np.zeros(2)
        return np.clip(d / n * self.amax, -self.amax, self.amax)

    def _get_action(self, state):
        s = np.asarray(state, dtype=np.float64)
        P = s[0:2].copy()
        th = float(s[2])
        q = s[16:18].copy()
        Pg = np.array([self.goal_adj[0], self.goal_adj[1]])
        thg = self.goal_adj[2]
        thg_raw = float(s[31])
        self.tick += 1

        glo = max(thg_raw - TOL_TH, -math.pi + 1e-4)
        ghi = min(thg_raw + TOL_TH, math.pi - 1e-4)
        thg_eff0 = 0.5 * (glo + ghi)
        glo = max(min(thg, thg_eff0 + 0.09), thg_eff0 - 0.09)
        ghi = glo
        thg_eff = 0.5 * (glo + ghi)
        dpx, dpy = Pg[0] - P[0], Pg[1] - P[1]
        if (abs(s[29] - P[0]) < 0.027 and abs(s[30] - P[1]) < 0.027
                and abs(th - thg_raw) < 0.125):
            return np.zeros(2)

        E_now = self._cost(dpx, dpy, wrap(thg_eff - th))
        if E_now < self.Ebest - 1e-4:
            self.Ebest = E_now
            self.stall = 0
        else:
            self.stall += 1
        nerr0 = max(abs(s[29] - P[0]) / 0.03, abs(s[30] - P[1]) / 0.03,
                    abs(th - float(s[31])) / TOL_TH)
        if self.stall > 200 and self.recover <= 0 and nerr0 > self.rec_nerr:
            self.stall = 0
            self.recover = 80
            self.Ebest = 1e18
            self.ban = {}
            self.phase = "plan"
        if self.recover > 0:
            self.recover -= 1
            Pg = np.array([2.5, 2.5])
            thg_eff = th
            self.ell = 0.08 * self.Lh
            dpx, dpy = Pg[0] - P[0], Pg[1] - P[1]
            if abs(dpx) < 0.35 and abs(dpy) < 0.35:
                self.recover = 0
            if self.recover == 0:
                self.ell = self.ell0
                self.Ebest = 1e18
                self.stall = 0
                self.phase = "plan"
        else:
            self.ell = self.ell0
        nerr = max(abs(s[29] - P[0]) / 0.03, abs(s[30] - P[1]) / 0.03,
                   abs(th - thg_raw) / TOL_TH)
        self.endmode = (nerr < self.nerr_end) and self.recover <= 0
        if self.endmode:
            Pg = np.array([float(s[29]), float(s[30])])
            thg_eff = thg_eff0
            dpx, dpy = Pg[0] - P[0], Pg[1] - P[1]
        if self.recover <= 0 and nerr < self.nerr_ell:
            self.ell = 0.215
        self.Smin = self.Smin_end if nerr < self.nerr_S else 0.10
        if self.stage_on and self.recover <= 0:
            ad = abs(wrap(thg_eff - th))
            ramp = min(max((ad - 0.10) / 0.35, 0.0), 1.0)
            if ramp > 0.0:
                Pg = Pg + self.stage_dir * (self.stage_d * ramp)
                dpx, dpy = Pg[0] - P[0], Pg[1] - P[1]
        mag = math.hypot(dpx, dpy) + self.ell * abs(wrap(thg_eff - th))
        replan_every = int(min(max(mag / self.rp_k, 2), self.rp_max))

        if self.phase == "push":
            if self.seg0 is None:
                self.seg0 = (P.copy(), th, q.copy())
        else:
            self.seg0 = None

        if self.push is None or self.phase == "plan" or (
                self.phase == "push" and (self.since >= replan_every
                                          or self.trav >= self.S_exec)):
            if self.phase == "push" and self.seg0 is not None:
                b0, t0, q0 = self.seg0
                moved = (math.hypot(P[0] - b0[0], P[1] - b0[1])
                         + self.ell * abs(wrap(th - t0)))
                rmoved = math.hypot(q[0] - q0[0], q[1] - q0[1])
                if rmoved > 0.12 and moved < 0.35 * self.gain * rmoved:
                    self.ban[self.push] = self.tick + 90
                    self.phase = "plan"
                self.seg0 = None
            self._replan(P, th, q, Pg, thg_eff)
            self.trav = 0.0
        self.since += 1

        if self.push is None:
            return np.zeros(2)

        c_loc, n_loc, u_loc = self.cands[self.push]
        M = rot(th)
        cw = P + M @ c_loc
        n = M @ n_loc

        if self.phase == "nav":
            self.nav_steps += 1
            ql = rot(-th) @ (q - P)
            if self._sdf(ql[0], ql[1]) < self.R + 0.02:
                gx, gy = self._sdf_grad(ql[0], ql[1])
                out = M @ np.array([gx, gy])
                return self._clamp(q, out * self.amax)
            far = cw + n * (self.R + self.stand)
            if self.nav_steps > 90:
                self.ban[self.push] = self.tick + 80
                self.phase = "plan"
                self._replan(P, th, q, Pg, thg_eff)
                if self.phase != "nav":
                    self.phase = "nav"
                c_loc, n_loc, u_loc = self.cands[self.push]
                M = rot(th)
                cw = P + M @ c_loc
                n = M @ n_loc
                far = cw + n * (self.R + self.stand)
            if self.path is None:
                self.path = [far]
                self.pi = 0
            # advance waypoints
            while self.pi < len(self.path) - 1 and \
                    np.linalg.norm(self.path[self.pi] - q) < 0.05:
                self.pi += 1
            self.path[-1] = far
            tgt = self.path[self.pi]
            if self.pi == len(self.path) - 1 and np.linalg.norm(tgt - q) < 0.035:
                self.phase = "push"
                self.since = 0
            else:
                return self._toward(q, tgt)

        desired = cw + n * (self.R - self.pen)
        act = self._toward(q, desired)
        if self.fine and (mag < self.fine_mag or nerr < self.nerr_lim):
            lim = min(max(0.30 * mag, self.fine_min), self.amax)
            if nerr < self.nerr_lim:
                lim = min(lim, self.end_lim)
            nn = float(np.linalg.norm(act))
            if nn > lim:
                act = act * (lim / nn)
        if self.endmode:
            lim = min(self.amax, max(self.end_floor, self.S_exec / self.end_div))
            nn = float(np.linalg.norm(act))
            if nn > lim:
                act = act * (lim / nn)
        self.trav += float(np.linalg.norm(act))
        return act

    # ------------------------------------------------------------------
    def _toward(self, q, target):
        d = np.asarray(target, dtype=np.float64) - q
        nn = float(np.linalg.norm(d))
        if nn < 1e-9:
            return np.zeros(2)
        return self._clamp(q, d / nn * min(self.amax, nn))

    def _clamp(self, q, step):
        tgt = np.clip(q + step, self.lo + self.R + 0.004, self.hi - self.R - 0.004)
        return np.clip(tgt - q, -self.amax, self.amax)
