import re
src = open('approach.py').read()
# plan_place parametrize
i = src.index('    def plan_place(self'); j = src.index('    def escape_blockers')
src = src[:i] + '''    def plan_place(self, time_budget=4.0, q0=None, hl=None):
        time_budget *= self.tscale()
        q0 = self.q if q0 is None else q0
        hl = self.held_local if hl is None else hl
        obs, _ = self.make_obstacles(exclude=('target_block',))
        model = Model(obs, hl)
        P = self.place_configs(hl, q0[3])
        cands = list(P[~model.hits(P)])
        if not cands:
            model = Model(obs, hl, margin=0.001)
            cands = list(P[~model.hits(P)])
        cands.sort(key=lambda q: np.linalg.norm(q[:2] - q0[:2]) + 0.1 * abs(wrap(q[2] - q0[2])))
        t0 = time.time()
        for q in cands[:8]:
            rem = time_budget - (time.time() - t0)
            if rem <= 0:
                break
            path = rrt_connect(model, q0, q, self.rng, time_limit=min(rem, 1.5))
            if path is not None:
                return shortcut(model, path, self.rng)
        return None

''' + src[j:]
# escape blockers parametrize
src = src.replace('''    def escape_blockers(self):
        obs, onames = self.make_obstacles(exclude=(self.held,))
        model = Model(obs, self.held_local)''', '''    def escape_blockers(self, name=None, q0=None, hl=None):
        name = self.held if name is None else name
        q0 = self.q if q0 is None else q0
        hl = self.held_local if hl is None else hl
        obs, onames = self.make_obstacles(exclude=(name,))
        model = Model(obs, hl)''')
src = src.replace('''        Q = self.q[None] + D
        Q[:, 3] = np.clip''', '''        Q = q0[None] + D
        Q[:, 3] = np.clip''')
# plan_dump parametrize
src = src.replace('''    def plan_dump(self, time_budget=3.0):
        time_budget *= self.tscale()
        obs, _ = self.make_obstacles(exclude=(self.held,))
        model = Model(obs, self.held_local, margin=0.01)''', '''    def plan_dump(self, time_budget=3.0, q0=None, hl=None, name=None):
        time_budget *= self.tscale()
        q0 = self.q if q0 is None else q0
        hl = self.held_local if hl is None else hl
        name = self.held if name is None else name
        obs, _ = self.make_obstacles(exclude=(name,))
        model = Model(obs, hl, margin=0.01)''')
i = src.index('    def plan_dump('); j = src.index('    # ------------------------------------------------------------------ main planner')
body = src[i:j]
body = body.replace('self.held_local', 'hl').replace('self.q[', 'q0[').replace('self.q,', 'q0,')
src = src[:i] + body + src[j:]
# plan_grasp -> generator of paths
i = src.index('    def plan_grasp('); j = src.index('    def enqueue_path')
src = src[:i] + '''    def iter_grasp_paths(self, name, time_budget=2.0, cands=None, maxc=6):
        time_budget *= self.tscale()
        obs_all, _ = self.make_obstacles(exclude=(self.held,) if self.held else ())
        model = Model(obs_all, None)
        if cands is None:
            cands = self.grasp_candidates(name)
        if not cands:
            return
        rok = {id(q): self.retreat_ok(name, q) for q in cands}
        cands = sorted(cands, key=lambda q: (not rok[id(q)], np.linalg.norm(q[:2] - self.q[:2])))
        t0 = time.time()
        for q in cands[:maxc]:
            rem = time_budget - (time.time() - t0)
            if rem <= 0:
                break
            path = rrt_connect(model, self.q, q, self.rng, time_limit=min(rem, 0.8))
            if path is not None:
                yield shortcut(model, path, self.rng)

    def plan_grasp(self, name, time_budget=2.0, cands=None):
        for p in self.iter_grasp_paths(name, time_budget, cands):
            return p
        return None

''' + src[j:]
# _plan rewrite
i = src.index('    def _plan(self):'); j = src.index('    def _go_grasp(self')
src = src[:i] + '''    def _plan(self):
        if self.held is not None:
            carry = self.cached_carry
            self.cached_carry = None
            if self.held == 'target_block':
                path = carry if carry is not None else self.plan_place()
                if path is not None:
                    self.enqueue_path(path, 1.0)
                    return
                self.priority_remove = self.escape_blockers()
                if DEBUG: print('escape blockers', self.priority_remove)
                self._release()
                return
            path = carry if carry is not None else self.plan_dump()
            if path is not None:
                self.enqueue_path(path, 1.0)
            else:
                self.fail_count[self.held] = self.fail_count.get(self.held, 0) + 1
                self.priority_remove = self.escape_blockers()
                if DEBUG: print('dump failed; escape blockers', self.priority_remove)
            self._release()
            return
        order = [n for n in self.priority_remove if n in self.rects]
        self.priority_remove = []
        if not order:
            cands = self.grasp_candidates('target_block')
            good, pblock = self.place_blockers(cands)
            esc = []
            if good:
                ntry = 0
                for path in self.iter_grasp_paths('target_block', 3.0, good, maxc=6):
                    qg = path[-1]
                    hl = self.local_of('target_block', qg)
                    carry = self.plan_place(2.0, qg, hl)
                    if carry is not None:
                        self._go_grasp(path, 'target_block', carry)
                        return
                    for n in self.escape_blockers('target_block', qg, hl):
                        if n not in esc:
                            esc.append(n)
                    ntry += 1
                    if ntry >= 2:
                        break
                if DEBUG: print('target carry failed; escape blockers', esc)
            order = list(esc)
            if not good and pblock:
                order += sorted(pblock, key=lambda n: np.linalg.norm(self.center(n) - self.q[:2]))
            for n in self.removal_order():
                if n not in order:
                    order.append(n)
        order = [n for n in order if self.fail_count.get(n, 0) < 2] + \\
            [n for n in order if self.fail_count.get(n, 0) >= 2]
        t0 = time.time()
        for name in order[:6]:
            if time.time() - t0 > 6.0 * self.tscale():
                break
            for path in self.iter_grasp_paths(name, 1.0, maxc=3):
                qg = path[-1]
                hl = self.local_of(name, qg)
                carry = self.plan_dump(1.0, qg, hl, name)
                if carry is not None:
                    self._go_grasp(path, name, carry)
                    return
                break
            self.fail_count[name] = self.fail_count.get(name, 0) + 0.5
        # fallback: try grasping target anyway
        path = self.plan_grasp('target_block', time_budget=1.5)
        if path is not None:
            self._go_grasp(path, 'target_block')
            return
        for name in order:
            path = self.plan_grasp(name, time_budget=0.5)
            if path is not None:
                self._go_grasp(path, name)
                return

''' + src[j:]
src = src.replace('''    def _go_grasp(self, path, name):''', '''    def _go_grasp(self, path, name, carry=None):
        self.cached_carry = carry''')
src = src.replace('''        self.priority_remove = []
''', '''        self.priority_remove = []
        self.cached_carry = None
''', 1)
open('approach.py','w').write(src)
