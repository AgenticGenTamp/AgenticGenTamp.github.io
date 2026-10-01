    def _base_targets(self, bx, by, r):
        rr = self.br + r - 0.012
        obs = self._obstacle()
        pts = []
        n = 16
        for k in range(n):
            a = 2 * math.pi * k / n
            for f in (1.0, 0.5):
                p = (bx + rr * f * math.cos(a), by + rr * f * math.sin(a))
                p = self._clampxy(*p, margin=0.012)
                if math.hypot(p[0] - bx, p[1] - by) > self.br + r - 0.006:
                    continue
                if obs is not None and in_rect(p, obs):
                    continue
                pts.append(p)
        return pts

    def _base_target(self, b, fp):
        """(steps, pos) for pressing with the robot base."""
        _, bx, by, r = b
        rr = self.br + r - 0.012
        dx, dy = fp[0] - bx, fp[1] - by
        d = math.hypot(dx, dy)
        if d < rr:
            c = fp
        else:
            c = (bx + dx / d * rr, by + dy / d * rr)
        c = self._clampxy(*c, margin=0.012)
        obs = self._obstacle()
        cands = []
        if math.hypot(c[0] - bx, c[1] - by) < self.br + r - 0.006 and (obs is None or not in_rect(c, obs)):
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

    def _grip_target(self, b, fp, th0, slack):
        """(steps, pos, th) for pressing with the extended gripper."""
        if self.grasped:
            return None
        _, bx, by, r = b
        D = self.arm_max + self.gw / 2 + r - 0.014
        obs = self._obstacle()
        ths = [math.atan2(by - fp[1], bx - fp[0]), th0]
        ths += [th0 + k * MAXTH for k in (-2, -1, 1, 2)]
        ths += [2 * math.pi * k / 12 for k in range(12)]
        best = None
        for th in ths:
            p = (bx - D * math.cos(th), by - D * math.sin(th))
            p = self._clampxy(*p, margin=0.012)
            if self._grip_contact(p, th, bx, by) > r - 0.008:
                continue
            if obs is not None and in_rect(p, obs):
                continue
            if not self._gripper_pose_ok(p, th):
                continue
            T = self._nav(fp, p)[0] / MAXD
            R = abs(wrap(th - th0)) / MAXTH - slack
            steps = max(T, R, 1.0)
            if best is None or steps < best[0]:
                best = (steps, p, th)
        return best

    def _stick_target(self, b, fp):
        if not self.grasped or self.rel is None:
            return None
        _, bx, by, r = b
        x0, x1, y0, y1 = self.rel
        tx = bx - (x0 + x1) / 2
        ty = min(max(fp[1], by - y1 + 0.02), by - y0 - 0.02)
        tx, ty = self._clampxy(tx, ty)
        X0, X1, Y0, Y1 = tx + x0, tx + x1, ty + y0, ty + y1
        cx = min(max(bx, X0), X1)
        cy = min(max(by, Y0), Y1)
        if math.hypot(cx - bx, cy - by) < r - 0.008:
            return cheb(fp, (tx, ty)) / MAXD, (tx, ty)
        return None

    def _button_target(self, b, fp, th0=None, slack=0.0):
        """returns (steps, pos, th_or_None, mode) or None"""
        if th0 is None:
            th0 = self.rth
        cands = []
        t = self._base_target(b, fp)
        if t is not None:
            cands.append((t[0], t[1], None, "base"))
        t = self._stick_target(b, fp)
        if t is not None:
            cands.append((t[0], t[1], None, "stick"))
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

    def _seq_cost(self, start, th0, seq, grasp_at, opt=None, trace=None):
        pos = start
        th = th0
        slack = 0.0
        cost = 0.0
        saved = (self.grasped, self.rel)
        try:
            for i, b in enumerate(seq):
                if i == grasp_at:
                    gx, gy, th_g, side = opt
                    T = self._nav(pos, (gx, gy))[0] / MAXD
                    R = abs(wrap(th_g - th)) / MAXTH - slack
                    cost += max(T, R) + 0.5
                    pos = (gx, gy)
                    th = th_g
                    slack = 0.0
                    self.grasped = True
                    self.rel = self._pred_rel(gx, gy)
                    if trace is not None:
                        trace.append(th_g)
                t = self._button_target(b, pos, th, slack)
                if t is None:
                    return 1e9
                cost += t[0]
                pos = t[1]
                if t[2] is None:
                    slack += t[0]
                else:
                    th = t[2]
                    slack = 0.0
                if trace is not None:
                    trace.append(t[2])
        finally:
            self.grasped, self.rel = saved
        return cost

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
        limit = 4 if can_grasp else 6
        if n <= limit:
            gopts = self._grasp_options(start) if can_grasp else []
            for perm in itertools.permutations(btns):
                c = self._seq_cost(start, th0, perm, None)
                if best is None or c < best[0]:
                    best = (c, perm, None, None)
                for g in range(n):
                    for opt in gopts:
                        c = self._seq_cost(start, th0, perm, g, opt)
                        if best is None or c < best[0]:
                            best = (c, perm, g, opt)
        if best is None or best[0] >= 1e9:
            best = self._greedy(start, btns, can_grasp)
        if best[0] >= 1e9 and self.grasped and getattr(self, "releases", 0) < 3:
            self.next_th = None
            return "release"
        _, perm, g, opt = best
        if g is not None and opt is None:
            opt = self._grasp_options(start)[0]
        trace = []
        self._seq_cost(start, th0, perm, g, opt, trace)
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

