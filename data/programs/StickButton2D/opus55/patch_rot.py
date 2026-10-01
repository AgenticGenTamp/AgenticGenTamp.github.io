s=open('approach.py').read()
def rep(old,new,cnt=1):
    global s
    assert s.count(old)>=1, old[:80]
    s=s.replace(old,new) if cnt==0 else s.replace(old,new,cnt)

# helpers after _pred_rel
rep('''    def _pred_rel(self, bx, by):''','''    def _loc_from(self, bx, by, th):
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

    def _pred_rel(self, bx, by):''')

# bounds: add stick bottom
rep('''            yhi = min(yhi, YMAX - y1 - 0.004)
''','''            yhi = min(yhi, YMAX - y1 - 0.004)
            ylo = max(ylo, YMIN - y0 + 0.004)
''')

# stick target
a=s.index("    def _stick_target(self, b, fp):")
b=s.index("    def _button_target(")
s=s[:a]+'''    def _stick_target(self, b, fp, th0=None, slack=0.0, thetas=None):
        if not self.grasped or self.loc is None:
            return None
        _, bx, by, r = b
        if th0 is None:
            th0 = self.rth
        if thetas is None:
            thetas = [th0 + k * MAXTH for k in range(-7, 8)]
        saved_rel = self.rel
        rho = self.stick[3] / 2 + r - 0.012
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
                qs = [P0[0] + f * (P1[0] - P0[0]) for f in ()]  # placeholder
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
                    if self._stick_dist(b, c, W) > r - 0.008:
                        continue
                    if not self._stick_path_ok(fp, th0, c, th):
                        continue
                    best = (steps, c, th)
        finally:
            self.rel = saved_rel
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

'''+s[b:]

# button_target: pass th/slack to stick target and keep theta
rep('''        t = self._stick_target(b, fp)
        if t is not None:
            cands.append((t[0], t[1], None, "stick"))''','''        t = self._stick_target(b, fp, th0, slack)
        if t is not None:
            th_s = None if abs(wrap(t[2] - th0)) < 1e-9 else t[2]
            cands.append((t[0], t[1], th_s, "stick"))''')

# seq cost
rep('''        saved = (self.grasped, self.rel)''','''        saved = (self.grasped, self.rel, self.loc)''')
rep('''            self.grasped, self.rel = saved''','''            self.grasped, self.rel, self.loc = saved''')
rep('''                    self.grasped = True
                    self.rel = self._pred_rel(gx, gy)
                    if trace''','''                    self.grasped = True
                    self.loc = self._loc_from(gx, gy, th_g)
                    self.rel = self._rel_at(th_g)
                    if trace''')
rep('''                if t[2] is None:
                    slack += t[0]
                else:
                    th = t[2]
                    slack = 0.0''','''                if t[2] is None:
                    slack = 0.0 if self.grasped else slack + t[0]
                else:
                    th = t[2]
                    slack = 0.0
                    if self.grasped:
                        self.rel = self._rel_at(th)''')

# verification in get_action
rep('''            rel = self._pred_rel(self.rx, self.ry)
            if self.rel is not None and max(abs(a - b) for a, b in zip(rel, self.rel)) > 0.004:
                self.fail_grasps += 1
                self.grasped = False
                self.rel = None
                self.grasp_stage = 0
                self.plan_key = None
            else:
                self.rel = rel''','''            loc = self._loc_from(self.rx, self.ry, self.rth)
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
                self.rel = self._pred_rel(self.rx, self.ry)''')
rep('''                    self.rel = self._pred_rel(self.rx, self.ry)
                    self.plan_key = None''','''                    self.rel = self._pred_rel(self.rx, self.ry)
                    self.loc = self._loc_from(self.rx, self.ry, self.rth)
                    self.plan_key = None''')
rep('''                self.rel = self._pred_rel(self.rx, self.ry)
                self.plan_key = None
                return self.get_action_after_grasp()
            self.grasp_stage = 0''','''                self.rel = self._pred_rel(self.rx, self.ry)
                self.loc = self._loc_from(self.rx, self.ry, self.rth)
                self.plan_key = None
                return self.get_action_after_grasp()
            self.grasp_stage = 0''')
# reset
rep('''        self.rel = None
        self.last_cmd = None''','''        self.rel = None
        self.loc = None
        self.last_cmd = None''')
# release: unrotate first
rep('''    def _release(self):
''','''    def _release(self):
        sth = wrap(self.stick[2])
        if abs(sth) > 0.01 and self.stuck < 4:
            d = max(-MAXTH, min(MAXTH, -sth))
            return self._act([0, 0, d, 0], 1.0)
''')
rep('''        self.grasped = False
        self.rel = None
        self.grasp_stage = 0
        self.plan_key = None
        return self._act([0, 0, 0, 0], 0.0)''','''        self.grasped = False
        self.rel = None
        self.loc = None
        self.grasp_stage = 0
        self.plan_key = None
        return self._act([0, 0, 0, 0], 0.0)''')
open('approach.py','w').write(s)
