    def _push_options(self, d, T, arm, btn, goal):
        """Enumerate push options: list of (face, t, base_contact_pose, u, D, hth)."""
        r = d["br"]
        v = goal - btn
        D = float(np.linalg.norm(v))
        u = v / max(D, 1e-9)
        ang_u = math.atan2(u[1], u[0])
        hk = self.hook
        opts = []
        Tinv = inverse(T)
        for fi, (p0, p1, nrm) in enumerate(hk.faces):
            hth = wrap(ang_u - math.atan2(nrm[1], nrm[0]))
            R = rot(hth)
            for pl in self._face_pts[fi]:
                pw = btn - u * (r + 0.005)
                hpos = pw - R @ pl
                H = (float(hpos[0]), float(hpos[1]), hth)
                B = compose(H, Tinv)
                opts.append((fi, pl, B, u, D, hth))
        return opts

    def _init_face_pts(self, r):
        hk = self.hook
        self._face_pts = []
        for fi, (p0, p1, nrm) in enumerate(hk.faces):
            L = float(np.linalg.norm(p1 - p0))
            margin = 0.02
            if L < 2 * margin:
                ts = [L / 2]
            else:
                ts = np.linspace(margin, L - margin, max(2, int((L - 2 * margin) / 0.05) + 1))
            pts = []
            for t in ts:
                pl = p0 + (p1 - p0) * (t / L)
                c_local = pl + nrm * (r + 0.005)
                okf = True
                for fj, (q0, q1, _) in enumerate(hk.faces):
                    if fj != fi and seg_dist(c_local, q0, q1) < r + 0.012:
                        okf = False
                        break
                if not okf:
                    continue
                if hk.dist_local(np.array([c_local[0]]), np.array([c_local[1]]))[0] < r:
                    continue
                pts.append(pl)
            self._face_pts.append(pts)

    def _pose_valid(self, d, B, T, arm, btn, margin, check_btn=True):
        x, y, th = B
        rad = d["rad"]
        if not (rad + 0.005 < x < WORLD_X[1] - rad - 0.005):
            return False
        if not (rad + 0.005 < y < TABLE_Y - rad - 0.003):
            return False
        g = (x + arm * math.cos(th), y + arm * math.sin(th))
        if not (0.04 < g[0] < 3.46 and 0.04 < g[1] < 2.46):
            return False
        H = compose(B, T)
        wx, wy = self.hook.hull_world((np.array(H[0]), np.array(H[1]), np.array(H[2])))
        if np.min(wx) < 0.004 or np.max(wx) > WORLD_X[1] - 0.004:
            return False
        if np.min(wy) < 0.004 or np.max(wy) > WORLD_Y[1] - 0.004:
            return False
        if check_btn:
            r = d["br"]
            if math.hypot(g[0] - btn[0], g[1] - btn[1]) < r + 0.045:
                return False
            if self.hook.dist_world(H, np.array([btn[0]]), np.array([btn[1]]))[0] < r + margin:
                return False
        return True

    def _option_feasible(self, d, opt, T, arm, btn):
        fi, pl, B, u, D, hth = opt
        r = d["br"]
        # quick base checks first
        rad = d["rad"]
        ymax = TABLE_Y - rad - 0.003
        if B[1] > ymax or B[1] + u[1] * D > ymax:
            return None
        pre = (B[0] - u[0] * 0.09, B[1] - u[1] * 0.09, B[2])
        end = (B[0] + u[0] * D, B[1] + u[1] * D, B[2])
        if not self._pose_valid(d, B, T, arm, btn, 0, check_btn=False):
            return None
        if not self._pose_valid(d, end, T, arm, btn, 0, check_btn=False):
            return None
        if not self._pose_valid(d, pre, T, arm, btn, 0.04):
            return None
        # gripper must avoid button along the push
        for k in np.linspace(0, 1, 6):
            bb = (B[0] + u[0] * D * k, B[1] + u[1] * D * k, B[2])
            g = np.array([bb[0] + arm * math.cos(bb[2]), bb[1] + arm * math.sin(bb[2])])
            c = btn + u * D * k
            if np.hypot(*(g - c)) < r + 0.045:
                return None
        return pre, end

    def _leg_options(self, d, T, arm, btn, goal, B0, limit=None):
        out = []
        for opt in self._push_options(d, T, arm, btn, goal):
            res = self._option_feasible(d, opt, T, arm, btn)
            if res is None:
                continue
            pre, end = res
            c = math.hypot(pre[0] - B0[0], pre[1] - B0[1]) / GRID + abs(wrap(pre[2] - B0[2])) / DTH * 1.5
            out.append((c, opt, pre, end))
            if limit is not None and len(out) >= limit:
                break
        out.sort(key=lambda z: z[0])
        return out

    def _final_goals(self, d):
        tgt = d["tgt"]
        goals = [tgt]
        for k in range(8):
            a = k * np.pi / 4
            goals.append(tgt + 0.055 * np.array([math.cos(a), math.sin(a)]))
        return goals

    def _intermediates(self, d, btn):
        tgt = d["tgt"]
        out = []
        for rho in (0.12, 0.22, 0.32):
            for k in range(12):
                a = k * np.pi / 6
                w = btn + rho * np.array([math.cos(a), math.sin(a)])
                if not (0.15 < w[0] < 3.35 and TABLE_Y + 0.06 < w[1] < 2.4):
                    continue
                if np.linalg.norm(w - tgt) < 0.1:
                    continue
                out.append(w)
        out.sort(key=lambda w: np.linalg.norm(w - btn) + np.linalg.norm(tgt - w))
        return out

    def _plan_sequence(self, d, T, arm, B0, quick=False):
        """Return (cost, goals list, first-leg options) or None."""
        btn = d["btn"]
        best = None
        for goal in self._final_goals(d):
            legs = self._leg_options(d, T, arm, btn, goal, B0)
            if legs:
                c = legs[0][0] + np.linalg.norm(goal - d["tgt"]) * 20
                if best is None or c < best[0]:
                    best = (c, [goal], legs)
        if best is not None:
            return best
        for w in self._intermediates(d, btn):
            legs1 = self._leg_options(d, T, arm, btn, w, B0)
            if not legs1:
                continue
            for goal in self._final_goals(d)[:1]:
                end = legs1[0][3]
                legs2 = self._leg_options(d, T, arm, w, goal, end, limit=1)
                if legs2:
                    c = legs1[0][0] + 30 + legs2[0][0]
                    return (c, [w, goal], legs1)
        return None

    def _best_push_cost(self, d, T, B0, arm):
        res = self._plan_sequence(d, T, arm, B0)
        return None if res is None else res[0]

    def _plan_push(self, d):
        if self.fail_count > 4:
            return False
        T = self.T
        arm = self.arm
        B0 = d["robot"]
        res = self._plan_sequence(d, T, arm, B0)
        if res is None:
            return False
        _, goals, legs = res
        tried = 0
        for c, opt, pre, end in legs:
            path = self._astar(d, pre, T, arm)
            tried += 1
            if path is not None:
                self.plan = path
                self.push = opt
                self.goals = goals
                return True
            if tried >= 6:
                break
        return False

