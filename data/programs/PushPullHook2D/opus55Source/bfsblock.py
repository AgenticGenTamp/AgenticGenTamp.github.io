    # ============================================================ BFS push planning
    def _poses_valid(self, d, X, Y, TH, T, arm, btn=None, margin=0.0):
        rad = d["rad"]
        ok = (X > rad + 0.005) & (X < WORLD_X[1] - rad - 0.005)
        ok &= (Y > rad + 0.005) & (Y < TABLE_Y - rad - 0.003)
        c, s = np.cos(TH), np.sin(TH)
        tx, ty, tth = T
        HX = X + c * tx - s * ty
        HY = Y + s * tx + c * ty
        HT = TH + tth
        wx, wy = self.hook.hull_world((HX, HY, HT))
        ok &= (wx.min(-1) > 0.004) & (wx.max(-1) < WORLD_X[1] - 0.004)
        ok &= (wy.min(-1) > 0.004) & (wy.max(-1) < WORLD_Y[1] - 0.004)
        GXg = X + arm * c
        GYg = Y + arm * s
        ok &= (GXg > 0.04) & (GXg < 3.46) & (GYg > 0.04) & (GYg < 2.46)
        if btn is not None:
            r = d["br"]
            ok &= self.hook.dist_world((HX, HY, HT), btn[0], btn[1]) > r + margin
            ok &= np.hypot(GXg - btn[0], GYg - btn[1]) > r + 0.045
        return ok

    def _bfs_grid(self, d, B0, T, arm, btn, margin=0.012):
        rad = d["rad"]
        x0, y0, th0 = B0
        xlo, xhi = rad + 0.005, WORLD_X[1] - rad - 0.005
        ylo, yhi = rad + 0.005, TABLE_Y - rad - 0.003
        ix = np.arange(int(math.ceil((xlo - x0) / GRID)), int(math.floor((xhi - x0) / GRID)) + 1)
        iy = np.arange(int(math.ceil((ylo - y0) / GRID)), int(math.floor((yhi - y0) / GRID)) + 1)
        if len(ix) == 0 or len(iy) == 0 or ix[0] > 0 or iy[0] > 0 or ix[-1] < 0 or iy[-1] < 0:
            ix = np.arange(min(ix[0] if len(ix) else 0, 0), max(ix[-1] if len(ix) else 0, 0) + 1)
            iy = np.arange(min(iy[0] if len(iy) else 0, 0), max(iy[-1] if len(iy) else 0, 0) + 1)
        XS = x0 + ix * GRID
        YS = y0 + iy * GRID
        THS = th0 + np.arange(NTH) * DTH
        GX, GY, GT = np.meshgrid(XS, YS, THS, indexing="ij")
        valid = self._poses_valid(d, GX, GY, GT, T, arm, btn, margin)
        si, sj = -ix[0], -iy[0]
        valid[si, sj, 0] = True
        dist = np.full(valid.shape, -1, dtype=np.int32)
        dist[si, sj, 0] = 0
        frontier = np.zeros(valid.shape, dtype=bool)
        frontier[si, sj, 0] = True
        visited = frontier.copy()
        lvl = 0
        while frontier.any() and lvl < 400:
            lvl += 1
            f = frontier | np.roll(frontier, 1, 2) | np.roll(frontier, -1, 2)
            g = f.copy()
            g[1:] |= f[:-1]
            g[:-1] |= f[1:]
            h = g.copy()
            h[:, 1:] |= g[:, :-1]
            h[:, :-1] |= g[:, 1:]
            new = h & valid & ~visited
            dist[new] = lvl
            visited |= new
            frontier = new
        return dict(XS=XS, YS=YS, x0=x0, y0=y0, th0=th0, ix0=ix[0], iy0=iy[0],
                    dist=dist, valid=valid, start=(si, sj, 0))

    def _hop_nodes(self, G, P):
        """Grid nodes from which pose P is reachable in one step."""
        fx = (P[0] - G["x0"]) / GRID - G["ix0"]
        fy = (P[1] - G["y0"]) / GRID - G["iy0"]
        fk = wrap(P[2] - G["th0"]) / DTH
        nX, nY, _ = G["dist"].shape
        out = []
        for i in {int(math.floor(fx)), int(math.ceil(fx))}:
            for j in {int(math.floor(fy)), int(math.ceil(fy))}:
                for k in {int(math.floor(fk)), int(math.ceil(fk))}:
                    if 0 <= i < nX and 0 <= j < nY:
                        out.append((i, j, k % NTH))
        return out

    def _contact_options(self, d, T, arm, btn, goals):
        """Vectorized contact poses for all faces/points/goals.
        Returns list of tuples (fi, pl, Bc, u, Dn, goal)."""
        r = d["br"]
        tgt = d["tgt"]
        Tinv = inverse(T)
        out = []
        for goal in goals:
            v = goal - btn
            D = float(np.linalg.norm(v))
            if D < 1e-6:
                continue
            u = v / D
            Dn = max(0.0, D - (0.1 - float(np.linalg.norm(goal - tgt))) + 0.006) if len(goals) and np.linalg.norm(goal - tgt) < 0.1 else D
            ang_u = math.atan2(u[1], u[0])
            for fi, (p0, p1, nrm) in enumerate(self.hook.faces):
                pts = self._face_pts[fi]
                if not pts:
                    continue
                hth = wrap(ang_u - math.atan2(nrm[1], nrm[0]))
                R = rot(hth)
                pw = btn - u * (r - 0.006)
                for pl in pts:
                    hpos = pw - R @ pl
                    Bc = compose((float(hpos[0]), float(hpos[1]), hth), Tinv)
                    out.append((fi, pl, Bc, u, Dn, goal, D))
        if not out:
            return []
        X = np.array([o[2][0] for o in out])
        Y = np.array([o[2][1] for o in out])
        TH = np.array([o[2][2] for o in out])
        ok = self._poses_valid(d, X, Y, TH, T, arm)
        U = np.array([o[3] for o in out])
        DD = np.array([o[6] for o in out])
        ok &= self._poses_valid(d, X + U[:, 0] * DD, Y + U[:, 1] * DD, TH, T, arm)
        # gripper must avoid the button along the push
        GXg = X + arm * np.cos(TH)
        GYg = Y + arm * np.sin(TH)
        for k in np.linspace(0, 1, 6):
            cx = btn[0] + U[:, 0] * DD * k
            cy = btn[1] + U[:, 1] * DD * k
            ok &= np.hypot(GXg + U[:, 0] * DD * k - cx, GYg + U[:, 1] * DD * k - cy) > r + 0.045
        return [o for o, f in zip(out, ok) if f]

    def _eval_push(self, d, T, arm, B0, btn, goals):
        """Return best (cost, option, node, G) using BFS distances."""
        G = self._bfs_grid(d, B0, T, arm, btn)
        dist = G["dist"]
        best = None
        for o in self._contact_options(d, T, arm, btn, goals):
            fi, pl, Bc, u, Dn, goal, D = o
            push_steps = max(1, int(math.ceil(Dn / 0.044)))
            for n in self._hop_nodes(G, Bc):
                dn = dist[n]
                if dn < 0:
                    continue
                c = dn + push_steps
                if best is None or c < best[0]:
                    best = (c, o, n, G)
        return best

    def _extract(self, G, node):
        dist = G["dist"]
        nX, nY, _ = dist.shape
        path = [node]
        n = node
        while dist[n] > 0:
            i, j, k = n
            found = None
            for dk in (0, -1, 1):
                for di in (0, -1, 1):
                    for dj in (0, -1, 1):
                        ni, nj, nk = i + di, j + dj, (k + dk) % NTH
                        if 0 <= ni < nX and 0 <= nj < nY and dist[ni, nj, nk] == dist[n] - 1:
                            found = (ni, nj, nk)
                            break
                    if found:
                        break
                if found:
                    break
            if found is None:
                return None
            n = found
            path.append(n)
        path.reverse()
        out = []
        for i, j, k in path[1:]:
            out.append((float(G["XS"][i]), float(G["YS"][j]), float(wrap(G["th0"] + k * DTH))))
        return out

    def _best_push_cost(self, d, T, B0, arm):
        res = self._eval_push(d, T, arm, B0, d["btn"], self._final_goals(d))
        if res is not None:
            return res[0]
        res = self._plan_sequence(d, T, arm, B0)
        return None if res is None else res[0] + 10

    def _plan_push(self, d):
        if self.fail_count > 4:
            return False
        T = self.T
        arm = self.arm
        B0 = d["robot"]
        goals = self.goals if (self.goals is not None and len(self.goals) > 1) else None
        if goals is None:
            res = self._eval_push(d, T, arm, B0, d["btn"], self._final_goals(d))
            if res is not None:
                c, o, node, G = res
                path = self._extract(G, node)
                if path is not None:
                    self.plan = path
                    self.push = (o[0], o[1], o[2], o[3], o[6], None)
                    self.goals = [o[5]]
                    return True
            seq = self._plan_sequence(d, T, arm, B0)
            if seq is None:
                return False
            goals = seq[1]
        # multi-leg: plan first leg towards goals[0]
        res = self._eval_push(d, T, arm, B0, d["btn"], [goals[0]])
        if res is None:
            self.goals = None
            return False
        c, o, node, G = res
        path = self._extract(G, node)
        if path is None:
            return False
        self.plan = path
        self.push = (o[0], o[1], o[2], o[3], o[6], None)
        self.goals = list(goals)
        return True

