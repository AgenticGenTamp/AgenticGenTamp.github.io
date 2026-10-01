src = open('approach.py').read()
src = src.replace('''            if self.held == 'target_block':
                path = self.plan_place()
                if path is not None:
                    self.enqueue_path(path, 1.0)
                    return
''', '''            if self.held == 'target_block':
                path = self.plan_place()
                if path is not None:
                    self.enqueue_path(path, 1.0)
                    return
                self.priority_remove = self.escape_blockers()
                self._release()
                return
''')
src = src.replace('''        order = []
        if not good and pblock:''', '''        order = [n for n in self.priority_remove if n in self.rects]
        self.priority_remove = []
        if not good and pblock:''')
src = src.replace('''        self.expected = None
''', '''        self.expected = None
        self.priority_remove = []
''', 1)
src = src.replace('''    def place_blockers(self''', '''    def escape_blockers(self):
        obs, onames = self.make_obstacles(exclude=(self.held,))
        model = Model(obs, self.held_local)
        M = 400
        D = np.column_stack([self.rng.uniform(-0.25, 0.25, M), self.rng.uniform(-0.25, 0.25, M),
                             self.rng.uniform(-0.6, 0.6, M), self.rng.uniform(-0.1, 0.1, M)])
        Q = self.q[None] + D
        Q[:, 3] = np.clip(Q[:, 3], ARM_MIN, ARM_MAX)
        H = model.per_obj(Q)
        cnt = H.sum(axis=0)
        order = [onames[i] for i in np.argsort(-cnt) if cnt[i] > 0 and onames[i] != 'target_block']
        return order[:4]

    def place_blockers(self''')
open('approach.py','w').write(src)
