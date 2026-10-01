s=open('approach.py').read()
def rep(old,new):
    global s
    assert old in s, old[:70]
    s=s.replace(old,new,1)

# flex grasp generator placed before _grasp_options
rep('''    def _grasp_options(self, pos):''','''    def _flex_grasp(self, side, pos, th):
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
        base_ang = math.atan2(-n[1], -n[0])
        hh = self.gh / 2
        hw = self.gw / 2
        for a in tilts:
            thg = base_ang + a
            dx, dy = math.cos(thg), math.sin(thg)
            gap = 0.004 if a == 0.0 else 0.003
            d_f = gap + hh * abs(math.sin(a))
            for L in (0.15, 0.2):
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
                    px, py = fx - reach * dx, fy - reach * dy
                    if not (xlo <= px <= xhi):
                        continue
                    if not (ylo <= py <= yhi):
                        py = min(max(py, ylo), yhi)
                        fy = py + reach * dy
                        if not (y0 + 0.03 <= fy <= y1 - 0.03):
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

    def _grasp_options(self, pos):''')

# seq cost: support side-string options
rep('''                if i == grasp_at:
                    gx, gy, th_g, side = opt''','''                if i == grasp_at:
                    extra = 0.5
                    if isinstance(opt, str):
                        o2 = self._flex_grasp(opt, pos, th)
                        if o2 is None:
                            return 1e9
                        if trace is not None:
                            self._first_opt = o2
                        gx, gy, th_g, side, _L = o2
                        extra = 0.0
                    else:
                        gx, gy, th_g, side = opt[:4]''')
rep('''                    cost += max(T, R) + 0.5
                    pos = (gx, gy)''','''                    cost += max(T, R) + extra
                    pos = (gx, gy)''')
# make_plan: grasp options are sides unless grasps failed
rep('''            gopts = self._grasp_options(start) if can_grasp else []''','''            if can_grasp and self.fail_grasps == 0 and abs(wrap(self.stick[2])) < 1e-3:
                gopts = ["below", "left", "right"]
            else:
                gopts = self._grasp_options(start) if can_grasp else []''')
rep('''        trace = []
        self._seq_cost(start, th0, perm, g, opt, trace)''','''        trace = []
        self._first_opt = None
        self._seq_cost(start, th0, perm, g, opt, trace)
        if isinstance(opt, str):
            if g == 0 and self._first_opt is not None:
                opt = self._first_opt
            elif g == 0:
                opt = self._grasp_options(start)[0]''')
open('approach.py','w').write(s)
