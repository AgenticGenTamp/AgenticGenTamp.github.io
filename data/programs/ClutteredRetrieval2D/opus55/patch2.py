src = open('approach.py').read()
i = src.index('    def plan_place(self')
j = src.index('    # ------------------------------------------------------------------ dump')
src = src[:i] + '''    def plan_place(self, time_budget=4.0):
        obs, _ = self.make_obstacles(exclude=(self.held,))
        model = Model(obs, self.held_local)
        P = self.place_configs()
        cands = list(P[~model.hits(P)])
        if not cands:
            model = Model(obs, self.held_local, margin=0.001)
            cands = list(P[~model.hits(P)])
        cands.sort(key=lambda q: np.linalg.norm(q[:2] - self.q[:2]) + 0.1 * abs(wrap(q[2] - self.q[2])))
        t0 = time.time()
        for q in cands[:8]:
            rem = time_budget - (time.time() - t0)
            if rem <= 0:
                break
            path = rrt_connect(model, self.q, q, self.rng, time_limit=min(rem, 1.5))
            if path is not None:
                return shortcut(model, path, self.rng)
        return None

    def place_blockers(self, grasp_cands):
        """For target grasp candidates, return (good_cands, best_blocker_list)."""
        obs, onames = self.make_obstacles(exclude=('target_block',))
        good = []
        best = None
        for q in grasp_cands:
            hl = self.local_of('target_block', q)
            model = Model(obs, hl)
            P = self.place_configs(hl, q[3])
            if (~model.hits(P)).any():
                good.append(q)
                continue
            wb = model.wall_bad(P)
            H = model.per_obj(P)
            for k in range(len(P)):
                if wb[k]:
                    continue
                bl = [onames[i] for i in np.nonzero(H[k])[0]]
                if best is None or len(bl) < len(best):
                    best = bl
        return good, (best or [])

'''+ src[j:]
i = src.index('    def _plan(self):')
j = src.index('    def _go_grasp(self')
src = src[:i] + '''    def _plan(self):
        if self.held is not None:
            if self.held == 'target_block':
                path = self.plan_place()
                if path is not None:
                    self.enqueue_path(path, 1.0)
                    return
            path = self.plan_dump()
            if path is not None:
                self.enqueue_path(path, 1.0)
            self._release()
            return
        cands = self.grasp_candidates('target_block')
        good, pblock = self.place_blockers(cands)
        if good:
            path = self.plan_grasp('target_block', time_budget=2.5, cands=good)
            if path is not None:
                self._go_grasp(path, 'target_block')
                return
        order = []
        if not good and pblock:
            order += sorted(pblock, key=lambda n: np.linalg.norm(self.center(n) - self.q[:2]))
        for n in self.removal_order():
            if n not in order:
                order.append(n)
        for name in order[:6]:
            path = self.plan_grasp(name, time_budget=1.0)
            if path is not None:
                self._go_grasp(path, name)
                return
        # fallback: try grasping target anyway
        path = self.plan_grasp('target_block', time_budget=1.5)
        if path is not None:
            self._go_grasp(path, 'target_block')

''' + src[j:]
open('approach.py','w').write(src)
