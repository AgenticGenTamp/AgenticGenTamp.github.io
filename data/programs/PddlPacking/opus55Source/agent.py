import time


def block_info(flat, k):
    out = []
    for i in range(k):
        o = 19 + 20 + 11 * i
        out.append(dict(pos=flat[o:o + 3].copy(), quat=tuple(flat[o + 3:o + 7]), held=flat[o + 7] > 0.5))
    return out


def slot_positions(k):
    """Return list of (x, y, tol_x, tol_y, yaw_tol)."""
    if k == 1:
        return [(0.0, 0.0, 0.08, 0.08, 0.8)]
    if k <= 4:
        c = 0.0675
        return [(x, y, 0.02, 0.02, 0.4) for x in (-c, c) for y in (-c, c)]
    c = 0.09
    return [(x, y, 0.004, 0.004, 0.1) for x in (-c, 0.0, c) for y in (-c, 0.0, c)]


def bidx(i):
    return 19 + 20 + 11 * i


class _Ref:
    def __init__(self, pos, quat):
        self.pos, self.quat = pos, quat

    def __getitem__(self, i):
        if i == 3:
            return self.quat
        if isinstance(i, slice):
            return self.pos[i]
        return self.pos[i]


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.sims = {}
        self.plan = []
        self.exps = []

    def _sim(self, k):
        if k not in self.sims:
            self.sims[k] = Sim(k)
        return self.sims[k]

    def reset(self, state, info):
        import time
        self.t0 = time.time()
        flat, k = obs_to_flat(state)
        self.k = k
        self.sim = self._sim(k)
        self.planner = Planner(self.sim)
        self.plan = []
        self.exps = []
        self.slots = slot_positions(k)
        self.fails = 0
        self.slp_deadline = 25.0 + 2.0 * self.k
        self.slp_budget = 2.5
        self.total_budget = 40.0
        self.la_margin = 0
        self._alts, self._alts_keys = [], []
        self.next_place = None
        self.la_n = 4
        self.hard_extra = 8.0
        self.la_budget = 1.5
        self.la_deadline = 25.0
        self.deadline = 1e18

    def elapsed(self):
        import time
        return time.time() - self.t0

    # ------------------------------------------------------------------
    def _occupied_slots(self, blocks):
        occ = set()
        for b in blocks:
            if b['held']:
                continue
            for si, (sx, sy, _, _, _) in enumerate(self.slots):
                if abs(b['pos'][0] - sx) < 0.071 and abs(b['pos'][1] - sy) < 0.071 and b['pos'][2] < 0.79:
                    occ.add(si)
        return occ

    def _block_done(self, b):
        x, y, z = b['pos']
        if b['held'] or z > 0.79:
            return False
        if not (abs(x) < 0.135 and abs(y) < 0.135):
            return False
        yaw = quat_yaw(b['quat'])
        r = abs(np.cos(yaw)) * 0.035 + abs(np.sin(yaw)) * 0.035
        return abs(x) + r <= 0.135 and abs(y) + r <= 0.135

    def grasp_targets(self, b):
        byaw = quat_yaw(b['quat'])
        for m in range(4):
            yaw = wrap(byaw + m * np.pi / 2)
            yield np.asarray(b['pos']) + np.array([0, 0, 0.03]), top_quat(yaw)

    def _targets_solutions(self, qc, tasks, use_stations, best_cost):
        """Yield candidates (cost, key, q) in nondecreasing cost order (station IK lazily)."""
        import heapq
        s, P = self.sim, self.planner
        region = base_region(qc)
        heap = []
        cnt = 0
        if not use_stations:
            for key, task, ref in tasks:
                for q in P.ik_candidates(ref.pos, ref.quat, qc, False):
                    heap.append((nsteps(qc, q), cnt, key, q, None)); cnt += 1
            if self.elapsed() < self.slp_deadline:
                t_end = time.time() + self.slp_budget
                stop = min([c[0] for c in heap], default=1e9)
                for key, task, ref in tasks:
                    if time.time() > t_end:
                        break
                    r = slp_safe(s, P, qc, qc, task, region, stop_steps=stop)
                    if r is not None:
                        heap.append((nsteps(qc, r[0]) - 0.5, cnt, key, r[0], None)); cnt += 1
        else:
            for key, task, ref in tasks:
                for st, th0 in base_stations(ref.pos[0], ref.pos[1]):
                    seed = np.concatenate([st, qc[3:]])
                    lb = nsteps(qc, seed)
                    heap.append((lb, cnt, key, None, (ref, seed, th0))); cnt += 1
        heapq.heapify(heap)
        while heap:
            c, _, key, q, th = heapq.heappop(heap)
            if q is not None:
                yield (c, key, q)
                continue
            ref, seed, th0 = th
            q = P._ik_pushout(ref.pos, ref.quat, seed, side_normal(th0), False)
            if q is None:
                seed2 = np.concatenate([seed[:3], CARRY_ARM])
                q = P._ik_pushout(ref.pos, ref.quat, seed2, side_normal(th0), False)
            if q is not None:
                heapq.heappush(heap, (max(c, nsteps(qc, q)), cnt, key, q, None)); cnt += 1

    def _pre(self, q, dz):
        s = self.sim
        F = s.ff.fk(q)
        quat = mat_to_quat(F[:3, :3])
        return s.ik(F[:3, 3] + np.array([0, 0, dz]), quat, q, list(range(3, 10)))

    def plan_grasp(self, flat, cand_blocks, use_stations, best=None):
        s, P = self.sim, self.planner
        qc = flat[:10].copy()
        blocks = block_info(flat, self.k)
        s.set_state(flat)
        best_cost = 1e9 if best is None else len(best[0])
        tasks = []
        for bi in cand_blocks:
            b = blocks[bi]
            byaw = quat_yaw(b['quat'])
            for m in range(4):
                yaw = wrap(byaw + m * np.pi / 2)
                ref = (np.asarray(b['pos']) + np.array([0, 0, 0.03]), top_quat(yaw))
                tasks.append((bi, TaskGrasp(b['pos'], yaw), (ref[0][0], ref[0][1], ref[0][2], ref[1])))
        tasks = [(k_, t_, (np.array(r[:3]), r[3])) for k_, t_, r in tasks]
        tasks = [(k_, t_, _Ref(r[0], r[1])) for k_, t_, r in tasks]
        cands = self._targets_solutions(qc, tasks, use_stations, best_cost)
        mg = self.la_margin
        seen = set(self._alts_keys)
        for c in cands:
            if c[0] >= best_cost + mg:
                break
            if time.time() > self.deadline and (best is not None or mg > 0 or time.time() > self.deadline + self.hard_extra):
                break
            bi, q = c[1], c[2]
            akey = (bi, int(round(q[0] / 0.35)), int(round(q[1] / 0.35)))
            if mg > 0 and akey in seen:
                continue
            lim = best_cost + mg
            pre = self._pre(q, 0.12)

            def check(fin, bi=bi):
                return fin[bidx(bi) + 7] > 0.5

            res = P.best_path(flat, qc, q, -1.0, check, pre, lim)
            if res is not None:
                if akey not in seen:
                    seen.add(akey)
                    self._alts_keys.append(akey)
                self._alts.append((len(res[0]), bi, res))
                if len(res[0]) < best_cost:
                    best = res
                    best_cost = len(res[0])
        return best

    def plan_place(self, flat, bi, use_stations, best=None):
        s, P = self.sim, self.planner
        qc = flat[:10].copy()
        blocks = block_info(flat, self.k)
        G = (tuple(flat[12:15]), tuple(flat[15:19]))
        inv_g = p.invertTransform(*G)
        occ = self._occupied_slots(blocks)
        free = [i for i in range(len(self.slots)) if i not in occ]
        s.set_state(flat)
        best_cost = 1e9 if best is None else len(best[0])
        tasks = []
        for si in free:
            sx, sy, tx, ty, ytol = self.slots[si]
            for m in range(4):
                tpos, tq = p.multiplyTransforms((sx, sy, 0.80), yaw_quat(m * np.pi / 2), inv_g[0], inv_g[1])
                task = TaskPlace(G, sx, sy, m * np.pi / 2, (tx, ty), ytol, z=(0.786, 0.95))
                tasks.append((si, task, _Ref(np.array(tpos), tq)))
        cands = self._targets_solutions(qc, tasks, use_stations, best_cost)
        for c in cands:
            if c[0] >= best_cost:
                break
            if time.time() > self.deadline and (best is not None or time.time() > self.deadline + self.hard_extra):
                break
            si, q = c[1], c[2]
            pre = self._pre(q, 0.1)
            sx, sy, tx, ty, _ = self.slots[si]

            def check(fin, sx=sx, sy=sy, tx=tx, ty=ty):
                if fin[11] > 0.5:
                    return False
                o = bidx(bi)
                return abs(fin[o] - sx) <= tx + 1e-3 and abs(fin[o + 1] - sy) <= ty + 1e-3 and fin[o + 2] < 0.79

            res = P.best_path(flat, qc, q, 1.0, check, pre, best_cost)
            if res is not None and len(res[0]) < best_cost:
                best = res
                best_cost = len(res[0])
        return best

    def replan(self, flat):
        k = self.k
        blocks = block_info(flat, k)
        n_todo = sum(1 for b in blocks if not self._block_done(b))
        rem = max(0.0, self.total_budget - self.elapsed())
        macro_b = max(0.3, min(3.0, rem / (n_todo + 1)))
        self.deadline = time.time() + macro_b
        self.slp_budget = 0.5 * macro_b
        held = [i for i, b in enumerate(blocks) if b['held']]
        if held:
            if self.next_place is not None and np.max(np.abs(self.next_place[0] - flat)) < 1e-4:
                res = self.next_place[1]
                self.next_place = None
                if res is not None:
                    return res
            self.next_place = None
            return self._place_full(flat, held[0])
        todo = [i for i, b in enumerate(blocks) if not self._block_done(b)]
        if not todo:
            return None
        tp = self.sim.ff.fk(flat[:10])[:3, 3] if hasattr(self.sim, 'ff') else np.zeros(3)
        todo.sort(key=lambda i: np.linalg.norm(blocks[i]['pos'][:2] - tp[:2]))
        use_la = self.elapsed() < self.la_deadline
        self.la_margin = 0
        self._alts, self._alts_keys = [], []
        res = self.plan_grasp(flat, todo[:4], False)
        if res is None or len(res[0]) > 4:
            res = self.plan_grasp(flat, todo, True, res)
        if use_la and res is not None:
            gbi = self._alts[-1][1] if self._alts else None
            for ln, bi_, r in self._alts:
                if r is res:
                    gbi = bi_
            self.deadline = time.time() + self.la_budget
            pl = self._place_full(res[1], gbi)
            if pl is None:
                return res
            best_tot, best_pair = len(res[0]) + len(pl[0]), (res, pl)
            if len(pl[0]) > 2 or len(res[0]) > 3:
                self.la_margin = min(6, best_tot - 2)
                self.deadline = time.time() + self.la_budget
                self.plan_grasp(flat, todo, True, res)
                self.la_margin = 0
                alts = sorted(self._alts, key=lambda t: t[0])
                n = 0
                for ln, bi_, r in alts:
                    if r is res or ln + 1 >= best_tot:
                        continue
                    if n >= self.la_n:
                        break
                    n += 1
                    self.deadline = time.time() + 0.5 * self.la_budget
                    pl2 = self._place_full(r[1], bi_)
                    if pl2 is not None and ln + len(pl2[0]) < best_tot:
                        best_tot, best_pair = ln + len(pl2[0]), (r, pl2)
            res = best_pair[0]
            self.next_place = (np.array(res[1]), best_pair[1])
        return res

    def _retreat(self, flat):
        """Fallback: move to a safe conf away from the table (keeps holding state)."""
        P = self.planner
        qc = flat[:10].copy()
        bx, by = qc[0], qc[1]
        cands = []
        for hx, hy, hth in ((-0.8, 0.0, 0.0), (0.8, 0.0, np.pi), (0.0, -1.15, np.pi / 2), (0.0, 1.2, -np.pi / 2),
                            (-0.85, -1.15, 0.0), (-0.85, 1.15, 0.0), (0.85, -1.15, np.pi), (0.85, 1.15, np.pi)):
            q = np.concatenate([[hx, hy, hth], CARRY_ARM])
            cands.append((np.hypot(hx - bx, hy - by), q))
        cands.sort(key=lambda c: c[0])
        held = flat[11] > 0.5
        for _, q in cands:
            gl = 0.0
            res = P.best_path(flat, qc, q, gl, lambda fin: (fin[11] > 0.5) == held, None, 1e9, 20)
            if res is not None:
                return res
        return None

    def _place_full(self, flat, bi):
        res = self.plan_place(flat, bi, False)
        if res is None or len(res[0]) > 4:
            res = self.plan_place(flat, bi, True, res)
        return res

    def get_action(self, state):
        flat, k = obs_to_flat(state)
        if self.plan and self.exps:
            if np.max(np.abs(flat - self.exps[0])) > 1e-3:
                self.plan = []
                self.mismatch = getattr(self, 'mismatch', 0) + 1
            else:
                self.exps.pop(0)
        if not self.plan:
            self.exps = []
            res = self.replan(flat)
            if res is None:
                res = self._retreat(flat)
            if res is None:
                self.fails += 1
                a = np.zeros(11, dtype=np.float32)
                rng = np.random.default_rng(self.fails)
                a[:10] = rng.uniform(-0.1, 0.1, 10)
                return a
            acts, fin, term = res
            exps = []
            self.sim.set_state(flat)
            for a in acts:
                o = self.sim.env.step(np.asarray(a, np.float32))[0]
                exps.append(np.array(o, float))
            if np.max(np.abs(exps[-1] - fin)) > 1e-3:
                self.mismatch = getattr(self, 'mismatch', 0) + 100
            self.plan = list(acts)
            self.exps = exps
        a = self.plan.pop(0)
        return np.clip(np.asarray(a, np.float32), self.action_space.low, self.action_space.high)
