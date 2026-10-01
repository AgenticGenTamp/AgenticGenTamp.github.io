src = open('approach.py').read()
old = "        cands.sort(key=lambda q: np.linalg.norm(q[:2] - self.q[:2]))\n"
assert src.count(old) == 1
src = src.replace(old, '''        rok = {id(q): self.retreat_ok(name, q) for q in cands}
        cands.sort(key=lambda q: (not rok[id(q)], np.linalg.norm(q[:2] - self.q[:2])))
''')
src = src.replace('''    def plan_grasp(self''', '''    def retreat_ok(self, name, q):
        obs, _ = self.make_obstacles(exclude=(name,))
        hl = self.local_of(name, q)
        model = Model(obs, hl, margin=0.002)
        c, s = math.cos(q[2]), math.sin(q[2])
        ks = np.linspace(0.02, 0.2, 10)
        Q = np.repeat(q[None], len(ks), axis=0)
        Q[:, 0] -= ks * c
        Q[:, 1] -= ks * s
        return not model.hits(Q).any()

    def plan_grasp(self''')
open('approach.py','w').write(src)
