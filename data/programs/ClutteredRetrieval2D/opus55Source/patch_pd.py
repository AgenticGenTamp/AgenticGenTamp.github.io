s=open('approach.py').read()
old="""        trees.reverse()
    return None
"""
new="""        trees.reverse()
    for t in trees:
        if t['start']:
            LAST_FAIL[0] = (t['N'][:t['n']].copy(), list(t['par']))
    return None
"""
assert old in s; s=s.replace(old,new)
s=s.replace("def rrt_connect(", "LAST_FAIL = [None]\n\n\ndef rrt_connect(",1)
s=s.replace("""    def _drop_cfgs(self, chk, q0, rects, held, target_mode=False):""","""    def _drop_cfgs(self, chk, q0, rects, held, target_mode=False, samples=None):""")
old="""        cands = []
        for _ in range(700):
            q = np.array([self.rng.uniform(0.1, 2.4), self.rng.uniform(0.1, 2.4),
                          self.rng.uniform(-np.pi, np.pi), ARM_MIN])
            if not chk.free(q, margin=0.01):
                continue"""
new="""        cands = []
        for k in range(700 if samples is None else len(samples)):
            if samples is None:
                q = np.array([self.rng.uniform(0.1, 2.4), self.rng.uniform(0.1, 2.4),
                              self.rng.uniform(-np.pi, np.pi), ARM_MIN])
                if not chk.free(q, margin=0.01):
                    continue
            else:
                q = samples[k]"""
assert old in s; s=s.replace(old,new)
old="""        cands.sort(key=lambda t: t[0])
        return [q for _, q in cands[:20]]"""
new="""        if samples is not None:
            return cands
        cands.sort(key=lambda t: t[0])
        return [q for _, q in cands[:20]]

    def _partial_drop(self, chk, q0, rects, held):
        \"\"\"After a failed drop plan, move the held obstruction to the best spot
        reachable within the explored RRT tree.\"\"\"
        if LAST_FAIL[0] is None or not chk.free(q0):
            return None
        N, par = LAST_FAIL[0]
        LAST_FAIL[0] = None
        if len(N) < 2 or np.abs(cfg_diff(N[0], q0)).max() > 1e-6:
            return None
        idx = np.arange(len(N))
        if len(idx) > 300:
            idx = np.concatenate([[0], self.rng.choice(np.arange(1, len(N)), 299, replace=False)])
        sc = self._drop_cfgs(chk, q0, rects, held, samples=[N[i] for i in idx])
        sc = [(c - nsteps(q0, N[i]) * 0.7, i) for (c, _), i in zip(sc, idx)]
        base = sc[0][0]
        best = min(sc, key=lambda t: t[0])
        if best[0] > base - 10:
            return None
        out = []
        i = best[1]
        while i != -1:
            out.append(N[i].copy())
            i = par[i]
        return densify(out[::-1])"""
assert old in s; s=s.replace(old,new)
old="""            if path is None:
                if DEBUG:
                    print('DBG held-fail', held"""
new="""            if path is None:
                path = self._partial_drop(chk, q0, rects, held)
                if path is not None:
                    if DEBUG:
                        print('DBG partial-drop', held, len(path))
                    self.plan = self._mk_plan(path, 1.0, 0.0)
                    return True
            if path is None:
                if DEBUG:
                    print('DBG held-fail', held"""
assert old in s; s=s.replace(old,new)
open('approach.py','w').write(s)
