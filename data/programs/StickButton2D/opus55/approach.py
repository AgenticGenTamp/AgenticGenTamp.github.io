import math
import time
import itertools
import numpy as np

XMIN, XMAX, YMIN, YMAX = 0.0, 3.5, 0.0, 2.5
TABLE_Y = 1.25
MAXD = 0.05
MAXTH = 0.196
INFL = 0.125  # inflation of the free stick for base navigation


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def seg_hits_rect(p, q, rect):
    """Liang-Barsky: does segment p->q intersect open rect (x0,x1,y0,y1)?"""
    x0, x1, y0, y1 = rect
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if abs(pk) < 1e-12:
            if qk <= 0:
                return False
        else:
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 >= t1:
                return False
    return t1 - t0 > 1e-9


def in_rect(p, rect):
    return rect[0] < p[0] < rect[1] and rect[2] < p[1] < rect[3]


def cheb_region(fp, R, rr):
    """Chebyshev-closest point to fp in rectangle R=(x0,x1,y0,y1) dilated by rr."""
    fx, fy = fp
    x0, x1, y0, y1 = R

    def gaps(t):
        gx = max(x0 - (fx + t), (fx - t) - x1, 0.0)
        gy = max(y0 - (fy + t), (fy - t) - y1, 0.0)
        return gx, gy

    gx, gy = gaps(0.0)
    if gx * gx + gy * gy <= rr * rr:
        return fp
    ax, ay = (gx, gy) if gx >= gy else (gy, gx)
    if ax - rr >= ay:
        hi = ax - rr
    else:
        ssum = ax + ay
        disc = ssum * ssum - 2.0 * (ax * ax + ay * ay - rr * rr)
        hi = (ssum - math.sqrt(max(disc, 0.0))) / 2.0
    hi = max(hi, 0.0) + 1e-12
    t = hi
    sx0, sx1, sy0, sy1 = fx - t, fx + t, fy - t, fy + t
    if sx1 < x0:
        qx = sx1
    elif sx0 > x1:
        qx = sx0
    else:
        qx = min(max(fx, x0), x1)
    if sy1 < y0:
        qy = sy1
    elif sy0 > y1:
        qy = sy0
    else:
        qy = min(max(fy, y0), y1)
    return (qx, qy)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------------------------
    def reset(self, state, info):
        self.grasped = False
        self.rel = None
        self.loc = None
        self.last_cmd = None
        self.last_pos = None
        self.stuck = 0
        self.grasp_stage = 0
        self.vac_steps = 0
        self.plan = None
        self.plan_key = None
        self.fail_grasps = 0
        self.pending = None
        self.next_pos = None
        self.pm = 0.012
        self.idle = 0
        self.cur_target = None
        self.next_th = None
        self.releases = 0

    # ------------------------------------------------------------------
    def _read(self, state):
        robot = None
        stick = None
        buttons = []
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            tn = o.type.name
            if tn == "crv_robot":
                robot = o
            elif tn == "rectangle" and stick is None:
                stick = o
            elif tn == "circle":
                buttons.append(o)
        g = state.get
        self.rx, self.ry, self.rth = g(robot, "x"), g(robot, "y"), g(robot, "theta")
        self.arm = g(robot, "arm_joint")
        self.arm_max = g(robot, "arm_length")
        self.br = g(robot, "base_radius")
        self.gw = g(robot, "gripper_width")
        self.gh = g(robot, "gripper_height")
        self.vac = g(robot, "vacuum")
        self.stick = None
        if stick is not None:
            self.stick = tuple(g(stick, f) for f in ("x", "y", "theta", "width", "height"))
        self.buttons = []
        for b in sorted(buttons, key=lambda o: o.name):
            pressed = g(b, "color_g") > 0.5 and g(b, "color_r") < 0.5
            if not pressed:
                self.buttons.append((b.name, g(b, "x"), g(b, "y"), g(b, "radius")))

    def _stick_corners(self):
        sx, sy, sth, sw, sh = self.stick
        c, s = math.cos(sth), math.sin(sth)
        pts = []
        for u, v in ((0, 0), (sw, 0), (sw, sh), (0, sh)):
            pts.append((sx + c * u - s * v, sy + s * u + c * v))
        return pts

    def _stick_aabb(self):
        pts = self._stick_corners()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), max(xs), min(ys), max(ys)

    def _obstacle(self):
        if self.grasped or self.stick is None:
            return None
        x0, x1, y0, y1 = self._stick_aabb()
        if y0 > TABLE_Y + INFL:
            return None
        return (x0 - INFL, x1 + INFL, y0 - INFL, 1e3)

    # ------------------------------------------------------------------
    def _bounds(self, margin=0.004):
        xlo = XMIN + self.br + margin
        xhi = XMAX - self.br - margin
        ylo = YMIN + self.br + margin
        yhi = TABLE_Y - self.br - margin
        if self.grasped and self.rel is not None:
            x0, x1, y0, y1 = self.rel
            yhi = min(yhi, YMAX - y1 - 0.004)
            ylo = max(ylo, YMIN - y0 + 0.004)
            xlo = max(xlo, XMIN - x0 + 0.004)
            xhi = min(xhi, XMAX - x1 - 0.004)
        return xlo, xhi, ylo, yhi

    def _clampxy(self, x, y, margin=0.004):
        xlo, xhi, ylo, yhi = self._bounds(margin)
        return min(max(x, xlo), xhi), min(max(y, ylo), yhi)

    # ------------------------------------------------------------------
    def _nav_nodes(self, obs):
        if obs is None:
            return []
        x0, x1, y0, _ = obs
        d = 0.01
        nodes = []
        for x in (x0 - d, x1 + d):
            p = (x, y0 - d)
            q = self._clampxy(*p)
            if abs(q[0] - p[0]) < 1e-9 and abs(q[1] - p[1]) < 1e-9:
                nodes.append(p)
        return nodes

    def _nav(self, p, q):
        """returns (cost, next_waypoint)"""
        obs = self._obstacle()
        if obs is None or not seg_hits_rect(p, q, obs):
            return cheb(p, q), q
        if in_rect(p, obs):
            # escape: sideways away from the stick, or straight down
            x0, x1, y0, _ = obs
            cx = (x0 + x1) / 2
            ex = x0 - 0.01 if p[0] < cx else x1 + 0.01
            escapes = [(ex, p[1]), (p[0], y0 - 0.01)]
            best = (1e9, q)
            for e in escapes:
                e2 = self._clampxy(*e)
                if in_rect(e2, obs):
                    continue
                c = cheb(p, e2) + self._nav(e2, q)[0]
                if c < best[0]:
                    best = (c, e2)
            return best
        nodes = self._nav_nodes(obs)
        best = (1e9, q)
        # paths through 1 or 2 nodes
        for i, a in enumerate(nodes):
            if seg_hits_rect(p, a, obs):
                continue
            if not seg_hits_rect(a, q, obs):
                c = cheb(p, a) + cheb(a, q)
                if c < best[0]:
                    best = (c, a)
            for j, b in enumerate(nodes):
                if i == j or seg_hits_rect(a, b, obs) or seg_hits_rect(b, q, obs):
                    continue
                c = cheb(p, a) + cheb(a, b) + cheb(b, q)
                if c < best[0]:
                    best = (c, a)
        return best

    # ------------------------------------------------------------------
    def _gripper_top(self, bx, by, th, x0, x1):
        """max y of gripper polygon over x in [x0,x1] (None if no overlap)."""
        c, s_ = math.cos(th), math.sin(th)
        L = self.arm_max
        gcx, gcy = bx + L * c, by + L * s_
        hw, hh = self.gw / 2, self.gh / 2
        pts = [(gcx + c * u - s_ * v, gcy + s_ * u + c * v)
               for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
        best = None
        for i in range(4):
            p, q = pts[i], pts[(i + 1) % 4]
            lo, hi = min(p[0], q[0]), max(p[0], q[0])
            a, b = max(lo, x0), min(hi, x1)
            if a > b:
                continue
            for x in (a, b):
                if hi - lo < 1e-12:
                    y = max(p[1], q[1])
                else:
                    t = (x - p[0]) / (q[0] - p[0])
                    y = p[1] + t * (q[1] - p[1])
                if best is None or y > best:
                    best = y
        return best

    def _grasp_pose(self):
        x0, x1, y0, y1 = self._stick_aabb()
        scx = (x0 + x1) / 2
        xlo, xhi, _, _ = self._bounds(margin=0.002)
        gx = min(max(scx, xlo), xhi)
        gcx = min(max(scx, XMIN + 0.045), XMAX - 0.045)
        shift = gcx - gx
        th = math.pi / 2
        if abs(shift) > 0.004:
            th = math.pi / 2 - math.asin(max(-0.6, min(0.6, shift / self.arm_max)))
        top = self._gripper_top(gx, 0.0, th, x0 - 0.002, x1 + 0.002)
        if top is None:
            top = self.arm_max + self.gw / 2
        y_touch = y0 - top - 0.0008
        y_pre = y_touch - 0.03
        return gx, th, y_pre, y_touch

    def _base_targets(self, bx, by, r):
        rr = self.br + r - self.pm
        obs = self._obstacle()
        pts = []
        n = 16
        for k in range(n):
            a = 2 * math.pi * k / n
            for f in (1.0, 0.5):
                p = (bx + rr * f * math.cos(a), by + rr * f * math.sin(a))
                p = self._clampxy(*p, margin=0.012)
                if math.hypot(p[0] - bx, p[1] - by) > self.br + r - 0.003:
                    continue
                if obs is not None and in_rect(p, obs):
                    continue
                pts.append(p)
        return pts

    def _base_target(self, b, fp):
        """(steps, pos) for pressing with the robot base."""
        _, bx, by, r = b
        rr = self.br + r - self.pm
        c = cheb_region(fp, (bx, bx, by, by), rr - 0.001)
        c = self._clampxy(*c, margin=0.012)
        obs = self._obstacle()
        cands = []
        if math.hypot(c[0] - bx, c[1] - by) < self.br + r - 0.003 and (obs is None or not in_rect(c, obs)):
            cands.append(c)
        else:
            cands = self._base_targets(bx, by, r)
        best = None
        for c in cands:
            cost = self._nav(fp, c)[0]
            if best is None or cost < best[0]:
                best = (cost, c)
        if best is None:
            return None
        return best[0] / MAXD, best[1]

    def _gripper_pose_ok(self, p, th):
        c, s = math.cos(th), math.sin(th)
        L = self.arm_max
        gcx, gcy = p[0] + L * c, p[1] + L * s
        hw, hh = self.gw / 2, self.gh / 2
        for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            x = gcx + c * u - s * v
            y = gcy + s * u + c * v
            if x < XMIN + 0.003 or x > XMAX - 0.003 or y < YMIN + 0.003 or y > YMAX - 0.003:
                return False
        if not self.grasped and self.stick is not None:
            x0, x1, y0, y1 = self._stick_aabb()
            for f in (1.0, 0.75, 0.5):
                px, py = p[0] + L * f * c, p[1] + L * f * s
                dx = max(x0 - px, 0.0, px - x1)
                dy = max(y0 - py, 0.0, py - y1)
                if math.hypot(dx, dy) < 0.045:
                    return False
        return True

    def _grip_contact(self, p, th, bx, by):
        c, s = math.cos(th), math.sin(th)
        L = self.arm_max
        gx, gy = p[0] + L * c, p[1] + L * s
        u = (bx - gx) * c + (by - gy) * s
        v = -(bx - gx) * s + (by - gy) * c
        du = max(abs(u) - self.gw / 2, 0.0)
        dv = max(abs(v) - self.gh / 2, 0.0)
        return math.hypot(du, dv)

    def _grip_target(self, b, fp, th0, slack, ths=None):
        """(steps, pos, th) for pressing with the extended gripper."""
        if self.grasped:
            return None
        _, bx, by, r = b
        L = self.arm_max
        rho = self.gw / 2 + r - self.pm - 0.002
        sp = self.gh / 2 - self.gw / 2
        obs = self._obstacle()
        if ths is None:
            ths = [math.atan2(by - fp[1], bx - fp[0]), th0]
            ths += [th0 + k * MAXTH for k in (-2, -1, 1, 2)]
            ths += [2 * math.pi * k / 12 for k in range(12)]
        best = None
        for th in ths:
            R = max(abs(wrap(th - th0)) / MAXTH - slack, 0.0)
            if best is not None and max(R, 1.0) >= best[0]:
                continue
            c, s_ = math.cos(th), math.sin(th)
            P0 = (bx - L * c + sp * s_, by - L * s_ - sp * c)
            P1 = (bx - L * c - sp * s_, by - L * s_ + sp * c)
            dxs, dys = P1[0] - P0[0], P1[1] - P0[1]
            ex, ey = fp[0] - P0[0], fp[1] - P0[1]
            fs = [0.0, 0.5, 1.0]
            for den, num in ((dxs, ex), (dys, ey), (dxs - dys, ex - ey), (dxs + dys, ex + ey)):
                if abs(den) > 1e-9:
                    f = num / den
                    if 0.0 < f < 1.0:
                        fs.append(f)
            for f in fs:
                q = (P0[0] + f * dxs, P0[1] + f * dys)
                p = cheb_region(fp, (q[0], q[0], q[1], q[1]), rho)
                p = self._clampxy(*p, margin=0.012)
                T = cheb(fp, p) / MAXD
                if best is not None and max(T, R, 1.0) >= best[0]:
                    continue
                if self._grip_contact(p, th, bx, by) > r - 0.003:
                    continue
                if obs is not None and in_rect(p, obs):
                    continue
                if not self._gripper_pose_ok(p, th):
                    continue
                T = self._nav(fp, p)[0] / MAXD
                steps = max(T, R, 1.0)
                if best is None or steps < best[0]:
                    best = (steps, p, th)
        return best

    def _stick_target(self, b, fp, th0=None, slack=0.0, thetas=None, extend=True):
        if not self.grasped or self.loc is None:
            return None
        _, bx, by, r = b
        if th0 is None:
            th0 = self.rth
        if thetas is None:
            res = getattr(self, "th_res", 2)
            thetas = [th0 + k * MAXTH / res for k in sorted(range(-7 * res, 7 * res + 1), key=abs)]
        saved_rel = self.rel
        rho = self.stick[3] / 2 + r - self.pm
        best = None
        try:
            for th in thetas:
                W = self._world_off(th)
                self.rel = self._rel_at(th)
                xlo, xhi, ylo, yhi = self._bounds()
                if xlo > xhi or ylo > yhi:
                    continue
                m0 = ((W[0][0] + W[1][0]) / 2, (W[0][1] + W[1][1]) / 2)
                m1 = ((W[2][0] + W[3][0]) / 2, (W[2][1] + W[3][1]) / 2)
                L = math.hypot(m1[0] - m0[0], m1[1] - m0[1])
                ux, uy = (m1[0] - m0[0]) / L, (m1[1] - m0[1]) / L
                hw = self.stick[3] / 2
                a0 = (m0[0] + ux * hw, m0[1] + uy * hw)
                a1 = (m1[0] - ux * hw, m1[1] - uy * hw)
                P0 = (bx - a0[0], by - a0[1])
                P1 = (bx - a1[0], by - a1[1])
                R = max(abs(wrap(th - th0)) / MAXTH - slack, 0.0)
                if best is not None and R >= best[0]:
                    continue
                cands = []
                for i in range(13):
                    f = i / 12.0
                    cands.append((P0[0] + f * (P1[0] - P0[0]), P0[1] + f * (P1[1] - P0[1])))
                # exact chebyshev-closest on segment: candidate breakpoints
                dxs, dys = P1[0] - P0[0], P1[1] - P0[1]
                ex, ey = fp[0] - P0[0], fp[1] - P0[1]
                for den, num in ((dxs, ex), (dys, ey), (dxs - dys, ex - ey), (dxs + dys, ex + ey)):
                    if abs(den) > 1e-9:
                        f = num / den
                        if 0.0 <= f <= 1.0:
                            cands.append((P0[0] + f * dxs, P0[1] + f * dys))
                for q in cands:
                    c = cheb_region(fp, (q[0], q[0], q[1], q[1]), rho - 0.001)
                    c = (min(max(c[0], xlo), xhi), min(max(c[1], ylo), yhi))
                    T = cheb(fp, c) / MAXD
                    steps = max(T, R)
                    if best is not None and steps >= best[0]:
                        continue
                    if self._stick_dist(b, c, W) > r - 0.003:
                        continue
                    if not self._stick_path_ok(fp, th0, c, th):
                        continue
                    best = (steps, c, th)
        finally:
            self.rel = saved_rel
        if best is None and extend:
            wide = [th0 + k * MAXTH for k in range(8, 17)] + [th0 - k * MAXTH for k in range(8, 17)]
            return self._stick_target(b, fp, th0, slack, thetas=wide, extend=False)
        return best

    def _stick_dist(self, b, p, W):
        _, bx, by, r = b
        ax, ay = p[0] + W[0][0], p[1] + W[0][1]
        ex, ey = W[1][0] - W[0][0], W[1][1] - W[0][1]
        fx, fy = W[3][0] - W[0][0], W[3][1] - W[0][1]
        le, lf = math.hypot(ex, ey), math.hypot(fx, fy)
        u = ((bx - ax) * ex + (by - ay) * ey) / le
        v = ((bx - ax) * fx + (by - ay) * fy) / lf
        du = max(-u, 0.0, u - le)
        dv = max(-v, 0.0, v - lf)
        return math.hypot(du, dv)

    def _button_target(self, b, fp, th0=None, slack=0.0):
        """returns (steps, pos, th_or_None, mode) or None"""
        if th0 is None:
            th0 = self.rth
        cands = []
        t = self._base_target(b, fp)
        if t is not None:
            cands.append((t[0], t[1], None, "base"))
        t = self._stick_target(b, fp, th0, slack)
        if t is not None:
            th_s = None if abs(wrap(t[2] - th0)) < 1e-9 else t[2]
            cands.append((t[0], t[1], th_s, "stick"))
        t = self._grip_target(b, fp, th0, slack)
        if t is not None:
            # tie-break towards the simpler base press
            cands.append((t[0] + 0.3, t[1], t[2], "grip"))
        if not cands:
            return None
        return min(cands, key=lambda c: c[0])

    def _needs_stick(self, b):
        return False

    # ------------------------------------------------------------------
    def _flex_grasp(self, side, pos, th):
        Ls = (0.15, 0.2)
        low = False
        if isinstance(side, tuple):
            side, mode = side
            if mode == "reach":
                Ls = (0.2,)
                low = True
        """time-optimal non-flush grasp pose for a stick side (vertical stick)."""
        if self.stick is None or abs(wrap(self.stick[2])) > 1e-3:
            return None
        x0, x1, y0, y1 = self._stick_aabb()
        xlo, xhi, ylo, yhi = self._bounds(margin=0.004)
        obs = self._obstacle()
        best = None
        if side == "below":
            n = (0.0, -1.0)
            tilts = (-0.3, 0.0, 0.3)
        elif side == "left":
            n = (-1.0, 0.0)
            tilts = (-0.6, -0.3, 0.0, 0.3, 0.6)
        else:
            n = (1.0, 0.0)
            tilts = (-0.6, -0.3, 0.0, 0.3, 0.6)
        if low:
            tilts = (0.0,)
        base_ang = math.atan2(-n[1], -n[0])
        hh = self.gh / 2
        hw = self.gw / 2
        for a in tilts:
            thg = base_ang + a
            dx, dy = math.cos(thg), math.sin(thg)
            gap = 0.004 if a == 0.0 else 0.003
            d_f = gap + hh * abs(math.sin(a))
            for L in Ls:
                reach = L + hw
                if side == "below":
                    fy = y0 - d_f
                    # choose face x near stick centre so base x is close to pos
                    xc = (x0 + x1) / 2
                    lim = 0.01 if a != 0.0 else 0.012
                    fx = min(max(pos[0] + reach * dx, xc - lim), xc + lim)
                    px, py = fx - reach * dx, fy - reach * dy
                    if not (xlo <= px <= xhi and ylo <= py <= yhi):
                        px, py = min(max(px, xlo), xhi), min(max(py, ylo), yhi)
                        fx, fy2 = px + reach * dx, py + reach * dy
                        if abs(fy2 - fy) > 1e-6 or abs(fx - xc) > lim + 1e-9:
                            continue
                else:
                    fx = (x0 - d_f) if side == "left" else (x1 + d_f)
                    fy = min(max(pos[1] + reach * dy, y0 + 0.03), y1 - 0.03)
                    if low:
                        fy = y0 - 0.013
                    px, py = fx - reach * dx, fy - reach * dy
                    if not (xlo <= px <= xhi):
                        continue
                    if not (ylo <= py <= yhi):
                        py = min(max(py, ylo), yhi)
                        fy = py + reach * dy
                        if not ((y0 - 0.014 if low else y0 + 0.03) <= fy <= y1 - 0.03):
                            continue
                p = (px, py)
                if obs is not None and in_rect(p, obs):
                    continue
                # gripper corners inside the world
                gcx, gcy = px + L * dx, py + L * dy
                ok = True
                for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
                    cx = gcx + dx * u - dy * v
                    cy = gcy + dy * u + dx * v
                    if cx < XMIN + 0.002 or cx > XMAX - 0.002 or cy < YMIN + 0.002 or cy > YMAX - 0.002:
                        ok = False
                if not ok:
                    continue
                T = self._nav(pos, p)[0] / MAXD
                R = abs(wrap(thg - th)) / MAXTH
                c = max(T, R) - 0.01 * L
                if best is None or c < best[0]:
                    best = (c, (px, py, thg, side, L))
        return None if best is None else best[1]

    def _grasp_options(self, pos):
        opts = []
        x0, x1, y0, y1 = self._stick_aabb()
        gx, th, y_pre, y_touch = self._grasp_pose()
        opts.append((gx, y_touch, th, "below"))
        reach = self.arm_max + self.gw / 2 + 0.0008
        xlo, xhi, ylo, yhi = self._bounds(margin=0.002)
        gy = min(max(pos[1], y0 - 0.027), y0 - 0.013)
        if ylo <= gy <= yhi:
            if x0 - reach >= xlo:
                opts.append((x0 - reach, gy, 0.0, "left"))
            if x1 + reach <= xhi:
                opts.append((x1 + reach, gy, math.pi, "right"))
        return opts

    def _seq_cost(self, start, th0, seq, grasp_at, opt=None, trace=None, ub=1e18):
        pos = start
        th = th0
        slack = 0.0
        cost = 0.0
        saved = (self.grasped, self.rel, self.loc)
        try:
            for i, b in enumerate(seq):
                if i == grasp_at:
                    extra = 0.5
                    if isinstance(opt, (str, tuple)) and len(opt) == 2:
                        fm = getattr(self, "_fmemo", None)
                        fk = (opt, round(pos[0], 7), round(pos[1], 7), round(th, 7))
                        if fm is not None and fk in fm:
                            o2 = fm[fk]
                        else:
                            o2 = self._flex_grasp(opt, pos, th)
                            if fm is not None:
                                fm[fk] = o2
                        if o2 is None:
                            return 1e9
                        if trace is not None:
                            self._first_opt = o2
                        gx, gy, th_g, side, _L = o2
                        extra = 0.0
                    else:
                        gx, gy, th_g, side = opt[:4]
                    T = self._nav(pos, (gx, gy))[0] / MAXD
                    R = abs(wrap(th_g - th)) / MAXTH - slack
                    cost += max(T, R) + extra
                    pos = (gx, gy)
                    th = th_g
                    slack = 0.0
                    self.grasped = True
                    self.loc = self._loc_from(gx, gy, th_g)
                    self.rel = self._rel_at(th_g)
                    if trace is not None:
                        trace.append(th_g)
                        self._tpos.append((gx, gy))
                memo = getattr(self, "_memo", None)
                if memo is not None:
                    key = (b[0], round(pos[0], 7), round(pos[1], 7), round(th, 7), round(slack, 4),
                           self.grasped, self.loc)
                    if key in memo:
                        t = memo[key]
                    else:
                        t = self._button_target(b, pos, th, slack)
                        memo[key] = t
                else:
                    t = self._button_target(b, pos, th, slack)
                if t is None:
                    return 1e9
                cost += t[0]
                if cost >= ub:
                    return 1e9
                pos = t[1]
                if t[2] is None:
                    slack = 0.0 if self.grasped else slack + t[0]
                else:
                    th = t[2]
                    slack = 0.0
                    if self.grasped:
                        self.rel = self._rel_at(th)
                if trace is not None:
                    trace.append(t[2])
                    self._tpos.append(t[1])
        finally:
            self.grasped, self.rel, self.loc = saved
        return cost

    def _loc_from(self, bx, by, th):
        c, s_ = math.cos(-th), math.sin(-th)
        out = []
        for (x, y) in self._stick_corners():
            dx, dy = x - bx, y - by
            out.append((c * dx - s_ * dy, s_ * dx + c * dy))
        return tuple(out)

    def _world_off(self, th, loc=None):
        loc = self.loc if loc is None else loc
        c, s_ = math.cos(th), math.sin(th)
        return [(c * u - s_ * v, s_ * u + c * v) for (u, v) in loc]

    def _rel_at(self, th):
        W = self._world_off(th)
        xs = [w[0] for w in W]
        ys = [w[1] for w in W]
        return (min(xs), max(xs), min(ys), max(ys))

    def _stick_path_ok(self, fp, th0, p, th):
        d = cheb(fp, p)
        dth = wrap(th - th0)
        n = int(math.ceil(max(d / MAXD, abs(dth) / MAXTH) - 1e-9))
        for k in range(1, n + 1):
            f = min(k * MAXD / d, 1.0) if d > 1e-12 else 1.0
            px, py = fp[0] + f * (p[0] - fp[0]), fp[1] + f * (p[1] - fp[1])
            t = th0 + max(-k * MAXTH, min(k * MAXTH, dth))
            for (wx, wy) in self._world_off(t):
                x, y = px + wx, py + wy
                if x < XMIN + 0.002 or x > XMAX - 0.002 or y < YMIN + 0.002 or y > YMAX - 0.002:
                    return False
        return True

    def _pred_rel(self, bx, by):
        x0, x1, y0, y1 = self._stick_aabb()
        return (x0 - bx, x1 - bx, y0 - by, y1 - by)

    def _make_plan(self):
        start = (self.rx, self.ry)
        th0 = self.rth
        btns = list(self.buttons)
        n = len(btns)
        can_grasp = not self.grasped and self.stick is not None
        best = None
        limit = 5 if can_grasp else 6
        self.th_res = 2 if n <= 4 else 1
        if n <= limit:
            if can_grasp and self.fail_grasps == 0 and abs(wrap(self.stick[2])) < 1e-3:
                gopts = [("below", "fast"), ("left", "fast"), ("right", "fast"), ("below", "reach"), ("left", "reach"), ("right", "reach")]
            else:
                gopts = self._grasp_options(start) if can_grasp else []
            self._memo = {}
            self._fmemo = {}
            try:
                for perm in itertools.permutations(btns):
                    ub = best[0] if best is not None else 1e18
                    c = self._seq_cost(start, th0, perm, None, ub=ub)
                    if best is None or c < best[0]:
                        best = (c, perm, None, None)
                    for g in range(n):
                        for opt in gopts:
                            c = self._seq_cost(start, th0, perm, g, opt, ub=best[0])
                            if c < best[0]:
                                best = (c, perm, g, opt)
            finally:
                self._memo = None
                self._fmemo = None
                self.th_res = 2
        if best is None and n > limit:
            best = self._local_search(start, th0, btns, can_grasp)
        if best is None or best[0] >= 1e9:
            best = self._greedy(start, btns, can_grasp)
        if best[0] >= 1e9 and self.grasped and getattr(self, "releases", 0) < 3:
            self.next_th = None
            return "release"
        _, perm, g, opt = best
        if g is not None and opt is None:
            opt = self._grasp_options(start)[0]
        trace = []
        self._first_opt = None
        self._tpos = []
        self._seq_cost(start, th0, perm, g, opt, trace)
        self.next_pos = self._tpos[1] if len(self._tpos) > 1 else None
        if isinstance(opt, (str, tuple)) and len(opt) == 2:
            if g == 0 and self._first_opt is not None:
                opt = self._first_opt
            elif g == 0:
                opt = self._grasp_options(start)[0]
        # next theta requirement after the first element
        self.next_th = None
        rest = trace[1:] if trace else []
        for t in rest:
            if t is not None:
                self.next_th = t
                break
        if g is not None and g == 0:
            return ("grasp", opt)
        return perm[0][0]

    def _local_search(self, start, th0, btns, can_grasp, budget=0.6):
        t_end = time.time() + budget
        if can_grasp and self.fail_grasps == 0 and abs(wrap(self.stick[2])) < 1e-3:
            gopts = [("below", "fast"), ("left", "fast"), ("right", "fast"),
                     ("left", "reach"), ("right", "reach")]
        else:
            gopts = self._grasp_options(start) if can_grasp else []
        self._memo = {}
        self._fmemo = {}
        self.th_res = 1
        try:
            # nearest-neighbour initial order by straight-line distance
            order = []
            rem = list(btns)
            pos = start
            while rem:
                b = min(rem, key=lambda b: cheb(pos, (b[1], b[2])))
                order.append(b)
                rem.remove(b)
                pos = (b[1], b[2])
            n = len(order)
            configs = [(None, None)] + [(g, o) for g in range(n) for o in gopts]

            def evalc(perm, cfg, ub=1e18):
                return self._seq_cost(start, th0, tuple(perm), cfg[0], cfg[1], ub=ub)

            best = None
            for cfg in configs:
                c = evalc(order, cfg, best[0] if best else 1e18)
                if best is None or c < best[0]:
                    best = (c, list(order), cfg)
                if time.time() > t_end:
                    break
            improved = True
            while improved and time.time() < t_end:
                improved = False
                c0, perm, cfg = best
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        p2 = list(perm)
                        x = p2.pop(i)
                        p2.insert(j, x)
                        cands = [cfg]
                        if cfg[0] is not None:
                            gi = cfg[0]
                            cands += [(g, cfg[1]) for g in (gi - 1, gi + 1) if 0 <= g < n]
                        for cf in cands:
                            c = evalc(p2, cf, best[0])
                            if c < best[0] - 1e-6:
                                best = (c, p2, cf)
                                improved = True
                        if time.time() > t_end:
                            break
                    if time.time() > t_end:
                        break
            c, perm, cfg = best
            return (c, tuple(perm), cfg[0], cfg[1])
        finally:
            self._memo = None
            self._fmemo = None
            self.th_res = 2

    def _greedy(self, start, btns, can_grasp):
        best = None
        for b in btns:
            t = self._button_target(b, start)
            if t is not None and (best is None or t[0] < best[0]):
                best = (t[0], b)
        if best is not None:
            rest = [b for b in btns if b is not best[1]]
            return (0, tuple([best[1]] + rest), None, None)
        if can_grasp:
            return (0, tuple(btns), 0, None)
        return (1e9, tuple(btns), None, None)

    # ------------------------------------------------------------------
    def _move_action(self, tx, ty, dth=0.0, darm=0.0):
        dx = tx - self.rx
        dy = ty - self.ry
        m = max(abs(dx), abs(dy))
        if m > MAXD:
            dx, dy = dx / m * MAXD, dy / m * MAXD
        if self.stuck >= 1:
            k = self.stuck % 4
            if k == 1:
                dx, dy = dx * 0.3, dy * 0.3
            elif k == 2:
                dy = 0.0
            elif k == 3:
                dx = 0.0
            dth = 0.0
            darm = 0.0
        return [dx, dy, dth, darm]

    def _act(self, a, vac):
        act = np.array([a[0], a[1], a[2], a[3], vac], dtype=np.float32)
        lo = np.asarray(self.action_space.low, dtype=np.float32)
        hi = np.asarray(self.action_space.high, dtype=np.float32)
        act = np.clip(act, lo, hi)
        self.last_cmd = act
        return act

    # ------------------------------------------------------------------
    def get_action(self, state):
        self._read(state)
        pos = (self.rx, self.ry, self.rth, self.arm)
        if self.last_cmd is not None and self.last_pos is not None:
            moved = max(abs(pos[0] - self.last_pos[0]), abs(pos[1] - self.last_pos[1]),
                        abs(wrap(pos[2] - self.last_pos[2])), abs(pos[3] - self.last_pos[3]))
            cmd = float(np.max(np.abs(self.last_cmd[:4])))
            if cmd > 1e-4 and moved < 1e-6:
                self.stuck += 1
            else:
                self.stuck = 0
            if moved < 1e-6:
                self.idle += 1
            else:
                self.idle = 0
        self.last_pos = pos

        pg = getattr(self, "pending", None)
        if pg is not None:
            self.pending = None
            gx, gy, thg, side, L = pg
            if (not self.grasped and self.vac > 0.5 and abs(self.rx - gx) < 1e-4 and abs(self.ry - gy) < 1e-4
                    and abs(wrap(self.rth - thg)) < 1e-4 and abs(self.arm - L) < 1e-4):
                self.grasped = True
                self.rel = self._pred_rel(self.rx, self.ry)
                self.loc = self._loc_from(self.rx, self.ry, self.rth)
                self.plan_key = None
                self.plan = None
            elif not self.grasped:
                # final approach step was rejected
                self.fail_grasps += 1
                self.plan_key = None
        if self.grasped:
            loc = self._loc_from(self.rx, self.ry, self.rth)
            if self.loc is not None and max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                                            for a, b in zip(loc, self.loc)) > 0.004:
                self.fail_grasps += 1
                self.grasped = False
                self.rel = None
                self.loc = None
                self.grasp_stage = 0
                self.plan_key = None
            else:
                self.loc = loc
                self.rel = self._pred_rel(self.rx, self.ry)

        if not self.buttons:
            return self._act([0, 0, 0, 0], 1.0 if self.grasped else 0.0)

        key = (tuple(b[0] for b in self.buttons), self.grasped)
        if key != self.plan_key or self.plan is None:
            self.plan_key = key
            self.plan = self._make_plan()

        if self.idle >= 3 and isinstance(self.plan, str) and self.plan != "release":
            # pressing apparently failed: use a larger safety margin
            self.pm = min(self.pm + 0.006, 0.03)
            self.idle = 0
            self.cur_target = None
        if isinstance(self.plan, tuple) and not self.grasped:
            if len(self.plan[1]) == 5:
                return self._grasp_flex(self.plan[1])
            return self._grasp(self.plan[1])
        if self.plan == "release" and self.grasped:
            return self._release()
        target = None
        for b in self.buttons:
            if b[0] == self.plan:
                target = b
        if target is None:
            self.plan_key = None
            return self._act([0, 0, 0, 0], 1.0 if self.grasped else 0.0)
        return self._goto_button(target)

    # ------------------------------------------------------------------
    def _lookahead(self, fp, p0, dth_abs, ok):
        """shift the press point toward the next target without costing steps."""
        nxt = getattr(self, "next_pos", None)
        if nxt is None or self.stuck or self.idle:
            return p0
        k = max(1, math.ceil(max(cheb(fp, p0) / MAXD, dth_abs / MAXTH) - 1e-6))
        lim = k * MAXD - 1e-7
        dx, dy = nxt[0] - p0[0], nxt[1] - p0[1]
        if abs(dx) + abs(dy) < 1e-9:
            return p0

        def feas(l):
            p = (p0[0] + l * dx, p0[1] + l * dy)
            return cheb(fp, p) <= lim and ok(p)

        if feas(1.0):
            return nxt
        lo, hi = 0.0, 1.0
        for _ in range(18):
            m = (lo + hi) / 2
            if feas(m):
                lo = m
            else:
                hi = m
        return (p0[0] + lo * dx, p0[1] + lo * dy)

    def _in_bounds(self, p, margin):
        xlo, xhi, ylo, yhi = self._bounds(margin=margin)
        return xlo <= p[0] <= xhi and ylo <= p[1] <= yhi

    def _goto_button(self, b):
        vac = 1.0 if self.grasped else 0.0
        fp = (self.rx, self.ry)
        cur = getattr(self, "cur_target", None)
        if cur is None or cur[0] != b[0] or cur[3] != self.grasped:
            t = self._button_target(b, fp)
            if t is None:
                return self._act([0, 0, 0, 0], vac)
            cur = (b[0], t[3], t[2], self.grasped)
            self.cur_target = cur
        mode, th = cur[1], cur[2]
        if mode == "grip" and not self.grasped:
            tg = self._grip_target(b, fp, self.rth, 0.0, ths=[th])
            if tg is None:
                self.cur_target = None
                tg = self._grip_target(b, fp, self.rth, 0.0)
                if tg is None:
                    return self._act(self._move_action(b[1], b[2]), vac)
                th = tg[2]
                self.cur_target = (b[0], "grip", th, self.grasped)
            p = tg[1]
            cost, wp = self._nav(fp, p)
            if cheb(wp, p) < 1e-9:
                obs = self._obstacle()
                _, bx, by, r = b

                def okg(q, th=th):
                    return (self._grip_contact(q, th, bx, by) <= r - self.pm and self._in_bounds(q, 0.012)
                            and (obs is None or (not in_rect(q, obs) and not seg_hits_rect(fp, q, obs)))
                            and self._gripper_pose_ok(q, th))
                p = self._lookahead(fp, p, abs(wrap(th - self.rth)), okg)
                wp = p
            dth = wrap(th - self.rth)
            if abs(dth) > MAXTH:
                dth = math.copysign(MAXTH, dth)
            final = cheb(fp, p) <= MAXD + 1e-9 and abs(wrap(th - self.rth)) <= MAXTH + 1e-9
            darm = 0.1 if final else -0.1
            if self.stuck >= 2 and final:
                # extension blocked: fall back to base pressing
                self.cur_target = (b[0], "base", None, self.grasped)
            return self._act(self._move_action(wp[0], wp[1], dth, darm), vac)
        if mode == "stick" and self.grasped:
            th = cur[2] if cur[2] is not None else self.rth
            if cur[2] is None:
                self.cur_target = (cur[0], cur[1], th, cur[3])
            ts = self._stick_target(b, fp, self.rth, 0.0, thetas=[th], extend=False)
            if ts is None or self.stuck >= 3:
                ts = self._stick_target(b, fp, self.rth, 0.0)
                if ts is not None:
                    self.cur_target = (cur[0], cur[1], ts[2], cur[3])
            if ts is not None:
                p, th = ts[1], ts[2]
                W = self._world_off(th)
                _, bx, by, r = b
                saved_rel = self.rel
                self.rel = self._rel_at(th)
                bnd = self._bounds()
                self.rel = saved_rel

                def oks(q, th=th, W=W, bnd=bnd):
                    return (bnd[0] <= q[0] <= bnd[1] and bnd[2] <= q[1] <= bnd[3]
                            and self._stick_dist(b, q, W) <= r - self.pm
                            and self._stick_path_ok(fp, self.rth, q, th))
                p = self._lookahead(fp, p, abs(wrap(th - self.rth)), oks)
                dth = wrap(th - self.rth)
                if abs(dth) > MAXTH:
                    dth = math.copysign(MAXTH, dth)
                return self._act(self._move_action(p[0], p[1], dth, 0.0), vac)
        tb = self._base_target(b, fp)
        t = tb
        if t is None and self.grasped:
            t = self._stick_target(b, fp, self.rth, 0.0)
        if t is None:
            self.cur_target = None
            return self._act([0, 0, 0, 0], vac)
        cost, wp = self._nav(fp, t[1])
        if cheb(wp, t[1]) < 1e-9:
            obs = self._obstacle()
            _, bx, by, r = b
            lim_r = self.br + r - self.pm

            def okb(q):
                return (math.hypot(q[0] - bx, q[1] - by) <= lim_r and self._in_bounds(q, 0.012)
                        and (obs is None or (not in_rect(q, obs) and not seg_hits_rect(fp, q, obs))))
            wp = self._lookahead(fp, t[1], 0.0, okb)
        dth = 0.0
        darm = 0.0
        if not self.grasped:
            darm = -0.1
            nth = getattr(self, "next_th", None)
            if nth is not None:
                dth = wrap(nth - self.rth)
                if abs(dth) > MAXTH:
                    dth = math.copysign(MAXTH, dth)
        return self._act(self._move_action(wp[0], wp[1], dth, darm), vac)

    # ------------------------------------------------------------------
    def _grasp_flex(self, opt):
        gx, gy, thg, side, L = opt
        fp = (self.rx, self.ry)
        dth = wrap(thg - self.rth)
        dist = cheb(fp, (gx, gy))
        if self.stuck >= 3:
            self.fail_grasps += 1
            self.plan_key = None
        if dist <= MAXD + 1e-9 and abs(dth) <= MAXTH + 1e-9 and self.stuck == 0:
            self.pending = opt
            return self._act([gx - self.rx, gy - self.ry, dth, L - self.arm], 1.0)
        cost, wp = self._nav(fp, (gx, gy))
        dth_c = max(-MAXTH, min(MAXTH, dth))
        darm = max(-0.1, self.br - self.arm)
        if abs(darm) < 1e-5:
            darm = 0.0
        if dist <= 1e-4:
            self.rot_wait = getattr(self, "rot_wait", 0) + 1
        return self._act(self._move_action(wp[0], wp[1], dth_c, darm), 0.0)

    def _grasp(self, opt):
        gx, gy, th_g, side = opt
        y_touch = gy
        dth = wrap(th_g - self.rth)
        if self.stuck >= 2 and self.grasp_stage == 0 and cheb((self.rx, self.ry), (gx, gy)) <= 1e-3:
            rd = getattr(self, "rot_dir", None)
            self.rot_dir = -math.copysign(1.0, dth) if rd is None else -rd
            self.stuck = 0
        rd = getattr(self, "rot_dir", None)
        if rd is not None and abs(dth) > 1e-3:
            dist = (dth * rd) % (2 * math.pi)
            dth = rd * dist
        dth_c = max(-MAXTH, min(MAXTH, dth))
        fp = (self.rx, self.ry)
        if self.grasp_stage in (0, 1, 2):
            fast = self.fail_grasps == 0
            tgt = (gx, y_touch)
            aligned = abs(dth) < 1e-3
            if side == "below":
                below = self.ry <= gy + 1e-3
            elif side == "left":
                below = self.rx <= gx + 1e-3
            else:
                below = self.rx >= gx - 1e-3
            if aligned and below and self.stuck < 3:
                darm = min(0.1, self.arm_max - self.arm)
            else:
                darm = max(-0.1, self.br - self.arm)
            if abs(darm) < 1e-5:
                darm = 0.0
            dist = cheb(fp, tgt)
            if dist <= 1e-4 and aligned and abs(self.arm - self.arm_max) < 1e-4:
                # arrived (with vacuum already on in fast mode)
                if fast and self.vac > 0.5:
                    self.grasped = True
                    self.rel = self._pred_rel(self.rx, self.ry)
                    self.loc = self._loc_from(self.rx, self.ry, self.rth)
                    self.plan_key = None
                    return self.get_action_after_grasp()
                self.grasp_stage = 3
                self.vac_steps = 0
            else:
                if dist > 1e-4:
                    if not aligned and abs(dth) > 0.3:
                        t2 = (min(max(gx, 0.13), XMAX - 0.13), min(max(gy, 0.13), TABLE_Y - 0.13))
                        if cheb(fp, t2) > 1e-3:
                            tgt = t2
                    cost, wp = self._nav(fp, tgt)
                    a = self._move_action(wp[0], wp[1], dth_c, darm)
                else:
                    self.rot_wait = getattr(self, "rot_wait", 0) + 1
                    a = [0.0, 0.0, dth_c, darm]
                    if self.stuck >= 1:
                        a = [0.0, 0.0, dth_c, 0.0]
                # does this step finish the approach?
                finishing = (aligned and max(abs(a[0] - (tgt[0] - self.rx)), abs(a[1] - (tgt[1] - self.ry))) < 1e-4
                             and tgt == (gx, y_touch) and self.arm + a[3] >= self.arm_max - 1e-4)
                vac = 1.0 if (fast and (finishing or (dist <= 1e-4 and abs(self.arm - self.arm_max) < 1e-4))) else 0.0
                return self._act(a, vac)
        if self.grasp_stage == 3:
            self.vac_steps += 1
            if self.vac_steps == 1:
                return self._act([0, 0, 0, 0], 1.0)
            if self.vac_steps == 2:
                self.sb_before = self._stick_aabb()[2]
                return self._act([0, -0.01, 0, 0], 1.0)
            if abs(self._stick_aabb()[2] - self.sb_before) > 0.005:
                self.grasped = True
                self.rel = self._pred_rel(self.rx, self.ry)
                self.loc = self._loc_from(self.rx, self.ry, self.rth)
                self.plan_key = None
                return self.get_action_after_grasp()
            self.grasp_stage = 0
            self.fail_grasps += 1
            return self._act([0, 0, 0, 0], 0.0)
        return self._act([0, 0, 0, 0], 0.0)

    def _release(self):
        sth = wrap(self.stick[2])
        if abs(sth) > 0.01 and self.stuck < 4:
            d = max(-MAXTH, min(MAXTH, -sth))
            return self._act([0, 0, d, 0], 1.0)
        x0, x1, y0, y1 = self._stick_aabb()
        scx = (x0 + x1) / 2
        want = min(max(scx, 0.2), XMAX - 0.2)
        tx = self.rx + (want - scx)
        tx, ty = self._clampxy(tx, self.ry)
        if abs(tx - self.rx) > 1e-4:
            return self._act(self._move_action(tx, ty), 1.0)
        self.releases = getattr(self, "releases", 0) + 1
        self.grasped = False
        self.rel = None
        self.loc = None
        self.grasp_stage = 0
        self.plan_key = None
        return self._act([0, 0, 0, 0], 0.0)

    def get_action_after_grasp(self):
        key = (tuple(b[0] for b in self.buttons), self.grasped)
        self.plan_key = key
        self.plan = self._make_plan()
        for b in self.buttons:
            if b[0] == self.plan:
                return self._goto_button(b)
        return self._act([0, 0, 0, 0], 1.0)
