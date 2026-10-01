    def _route(self, cur, goal, world, held, closed):
        """Route the base around the table through ring waypoints (yaw-aware)."""
        ring = [(-0.72, -1.02), (0.0, -1.02), (0.72, -1.02), (0.72, 0.0),
                (0.72, 1.02), (0.0, 1.02), (-0.72, 1.02), (-0.72, 0.0)]
        y0, y1 = float(cur[2]), float(goal[2])
        pts = [tuple(cur[:2]), tuple(goal[:2])] + ring
        for p in (cur, goal):
            if abs(p[1]) < 0.75:
                pts.append((math.copysign(0.72, p[0]), float(p[1])))
            if abs(p[0]) < 0.45:
                pts.append((float(p[0]), math.copysign(1.02, p[1])))
        yaws = (y0, y1)
        # states: (point index, yaw index)
        S = [(i, j) for i in range(len(pts)) for j in range(2)]
        start = (0, 0); goal_s = (1, 1)

        def ecost(a, b):
            pa, pb = pts[a[0]], pts[b[0]]
            dyaw = abs(wrap(yaws[b[1]] - yaws[a[1]]))
            return max(abs(pb[0] - pa[0]), abs(pb[1] - pa[1]), dyaw) / STEP

        def edge_ok(a, b):
            pa, pb = pts[a[0]], pts[b[0]]
            ya, yb = yaws[a[1]], yaws[b[1]]
            dy = wrap(yb - ya)
            n = int(math.ceil(max(abs(pb[0] - pa[0]), abs(pb[1] - pa[1]), abs(dy)) / STEP - 1e-7))
            for k in range(1, n + 1):
                f = k / n
                if not base_ok((pa[0] + (pb[0] - pa[0]) * f, pa[1] + (pb[1] - pa[1]) * f, ya + dy * f)):
                    return False
            return True
        dist = {s: 1e9 for s in S}; prev = {}; dist[start] = 0.0; done = set()
        while True:
            u = None; du = 1e9
            for s in S:
                if s not in done and dist[s] < du:
                    u, du = s, dist[s]
            if u is None or u == goal_s:
                break
            done.add(u)
            for v in S:
                if v in done or v == u:
                    continue
                c = du + ecost(u, v) + 0.01
                if c < dist[v] and edge_ok(u, v):
                    dist[v] = c; prev[v] = u
        if dist[goal_s] >= 1e9:
            return None
        path = [goal_s]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path = path[::-1]
        total = max(dist[goal_s], 1e-6)
        dq = cfg_diff(cur, goal)
        best = None
        for mode in (0, 1):
            seq = [cur]
            acc = 0.0
            for i in range(1, len(path) - 1):
                acc += ecost(path[i - 1], path[i]) + 0.01
                f = acc / total
                c = cur + dq * f
                c[0], c[1] = pts[path[i][0]]
                c[2] = yaws[path[i][1]]
                if mode == 1:
                    c[3:] = Q0
                seq.append(c)
            seq.append(goal)
            tot = self._seq_ok(seq, world, held, closed)
            if tot is not None and (best is None or tot < best[1]):
                best = (seq[1:], tot)
        return best

    # ------------------------------------------------------------ options
    def _base_list(self, cur, x, y, extra):
        bases = [tuple(cur[:3])] + BASE_CANDS + block_bases(x, y, dense=extra)
        cand = []; seen = set()
        for base in bases:
            key = (round(base[0], 3), round(base[1], 3), round(wrap(base[2]), 3))
            if key in seen:
                continue
            seen.add(key)
            if not base_ok(base):
                continue
            sh = shoulder_xy(base)
            dist = math.hypot(x - sh[0], y - sh[1])
            if dist > 0.86 or dist < 0.2:
                continue
            lb = int(math.ceil(max(abs(base[0] - cur[0]), abs(base[1] - cur[1]),
                                   abs(wrap(base[2] - cur[2]))) / STEP - 1e-7))
            cand.append((lb, base))
        cand.sort(key=lambda t: t[0])
        return cand

    def _grasp_options(self, cur, name, b, world, extra=False, max_ik=None):
        x, y, z, yaw = b
        pos = np.array([x, y, z + GRASP_DZ])
        out = []; best = None; n_ik = 0
        max_ik = max_ik or (60 if extra else 20)
        for lb, base in self._base_list(cur, x, y, extra):
            if best is not None and lb > best + 1 and len(out) >= 4:
                break
            if n_ik >= max_ik:
                break
            same = np.allclose(base, cur[:3])
            inits = [cur[3:], Q0] if same else [Q0]
            for qi in inits:
                n_ik += 1
                c = self._ik_cfg(base, pos, qi, yaw=yaw)
                if c is None:
                    continue
                for k in range(4):
                    cc = c.copy(); cc[9] = wrap(cc[9] + k * math.pi / 2)
                    if not self._config_ok(cc, world, exclude=name, closed=False, margin=0.0):
                        continue
                    sc = max(nsteps(cur, cc), lb)
                    out.append((sc, cc))
                    best = sc if best is None else min(best, sc)
                break
        out.sort(key=lambda t: t[0])
        return out

    def _slot_candidates(self, placed):
        px, py = self.plate_c; hx, hy = self.plate_h
        lim = hx - 0.035 - PLATE_MARGIN
        g = np.linspace(-lim, lim, 9)
        out = []
        for sx in g:
            for sy in g:
                c = (px + sx, py + sy)
                ok = True
                for pb in placed:
                    if rect_overlap_2d(c, 0.0, (0.035, 0.035), pb[:2], pb[3], (0.035, 0.035), margin=0.004):
                        ok = False; break
                if ok:
                    out.append(c)
        return out

    def _room_left(self, placed, n):
        if n <= 0:
            return True
        cands = self._slot_candidates(placed)
        if not cands:
            return False
        if n == 1:
            return True
        for c in cands:
            if self._room_left(placed + [(c[0], c[1], 0.781, 0.0)], n - 1):
                return True
        return False

    def _place_options(self, gcfg, world, held, placed, nrem, extra=False, max_ik=None, nslots=4):
        """Place configs from config gcfg (holding). Returns sorted list of (steps, cfg)."""
        po, Rrel = held
        _, _, _, tool, _ = fk_full(gcfg[:3], gcfg[3:])
        slots = self._slot_candidates(placed)
        slots.sort(key=lambda c: (c[0] - tool[0]) ** 2 + (c[1] - tool[1]) ** 2)
        good = []
        for c in slots:
            if nrem > 0 and not self._room_left(placed + [(c[0], c[1], 0.781, 0.0)], nrem):
                continue
            good.append(c)
            if len(good) >= nslots:
                break
        out = []
        max_ik = max_ik or (40 if extra else 12)
        n_ik = 0
        for c in good:
            best = None
            for lb, base in self._base_list(gcfg, c[0], c[1], extra):
                if best is not None and lb > best:
                    break
                if n_ik >= max_ik * (1 + good.index(c)):
                    break
                for high in (False, True):
                    n_ik += 1
                    res = self._place_ik(base, gcfg, c, po, Rrel, high, world, held)
                    if res is None:
                        continue
                    s = max(nsteps(gcfg, res), lb)
                    out.append((s, res))
                    best = s if best is None else min(best, s)
                    break
        out.sort(key=lambda t: t[0])
        return out

    def _place_ik(self, base, gcfg, c, po, Rrel, high, world, held):
        _, _, _, _, Rg = fk_full(gcfg[:3], gcfg[3:])
        Rb = Rg @ Rrel
        byaw = math.atan2(Rb[1, 0], Rb[0, 0])
        tyaw = math.atan2(Rg[1, 1], Rg[0, 1])
        off = tyaw - byaw
        zc = (NEAR_TOOL_Z_HIGH - po[0]) if high else (TABLE_TOP + BLOCK_HZ + 0.008)
        same = np.allclose(base, gcfg[:3])
        # solve once, then use wrist roll for the 4 symmetric yaws
        k0 = round(wrap(tyaw - off) / (math.pi / 2))
        psi = wrap(k0 * math.pi / 2 + off)
        q, ok = ik_down(base, np.array([c[0], c[1], zc]), gcfg[3:] if same else Q0, yaw=psi, p_off=po)
        if not ok:
            return None
        best = None
        for k in range(4):
            # rotating the wrist roll by m*pi/2 rotates tool about vertical; offset point po moves
            qq = q.copy(); qq[6] = wrap(qq[6] + k * math.pi / 2)
            if k:
                qq, ok = ik_down(base, np.array([c[0], c[1], zc]), qq, yaw=wrap(psi - k * math.pi / 2), p_off=po, iters=30)
                if not ok:
                    continue
            cfg = np.concatenate([np.asarray(base, float), qq])
            if not self._config_ok(cfg, world, None, held, closed=True, margin=0.0):
                continue
            s = nsteps(gcfg, cfg)
            if best is None or s < best[0]:
                best = (s, cfg)
        return None if best is None else best[1]

    # ------------------------------------------------------------ top level
    def _home(self, cfg):
        return np.concatenate([[-0.75, 0.0, 0.0], Q0])

    def _make_plan(self, cfg, blocks, holding, held, gtf, gq):
        extra = self.fail >= 2 or self.no_plan > 0
        cheap = self.t_used > 35.0
        placed = [b for n, b in blocks.items() if n != held and self._on_plate(b)]
        remaining = [n for n, b in blocks.items() if n != held and not self._on_plate(b)]
        if holding and held is not None:
            heldinfo = (gtf, quat_to_R(gq))
            w2 = World({n: b for n, b in blocks.items() if n != held})
            opts = self._place_options(cfg, w2, heldinfo, placed, len(remaining), extra=extra)
            for s, pc in opts[:8]:
                bp = self._best_path(cfg, pc, w2, held=heldinfo, closed=True)
                if bp is not None:
                    wps, tot = bp
                    return [(w, 0.0) for w in wps[:-1]] + [(wps[-1], 1.0)]
            return self._fallback(cfg, w2, heldinfo, True)
        if not remaining:
            return []
        world = World(blocks)
        cands = []
        for n in remaining:
            for s, gc in self._grasp_options(cfg, n, blocks[n], world, extra=extra)[:4]:
                cands.append((s, n, gc))
        cands.sort(key=lambda t: t[0])
        # lookahead: add estimated place cost
        scored = []
        for s, n, gc in cands[:(4 if cheap else 10)]:
            bp = self._best_path(cfg, gc, world, closed=False)
            if bp is None:
                continue
            wps, tot = bp
            pc = 0
            if not cheap:
                b = blocks[n]
                # held offset estimate: block centre GRASP_DZ below tool, aligned
                po = np.array([GRASP_DZ, 0.0, 0.0])
                _, _, _, _, Rg = fk_full(gc[:3], gc[3:])
                Rrel = Rg.T @ Rz(b[3])
                po_opts = self._place_options(gc, World({k: v for k, v in blocks.items() if k != n}),
                                              (po, Rrel), placed, len(remaining) - 1, max_ik=6, nslots=2)
                pc = po_opts[0][0] if po_opts else 30
            scored.append((tot + pc, tot, n, wps))
        if not scored:
            return self._fallback(cfg, world, None, False)
        scored.sort(key=lambda t: (t[0], t[1]))
        _, tot, n, wps = scored[0]
        self.target = n
        return [(w, 0.0) for w in wps[:-1]] + [(wps[-1], -1.0)]

    def _fallback(self, cfg, world, held, closed):
        self.no_plan += 1
        home = self._home(cfg)
        if nsteps(cfg, home) > 0:
            bp = self._best_path(cfg, home, world, held=held, closed=closed)
            if bp is not None:
                return [(w, 0.0) for w in bp[0]]
        # random small perturbation of the arm
        rng = np.random.default_rng(self.no_plan)
        c = cfg.copy(); c[3:] += rng.uniform(-0.2, 0.2, 7)
        c[3:] = np.where(CONT, wrap(c[3:]), np.clip(c[3:], LO, HI))
        return [(c, 0.0)]

