src = open('approach.py').read()
# 1. per-object blockers helper in Model
src = src.replace('''    def free(self, q):''', '''    def per_obj(self, Q, m=None):
        """(M,N) bool: which obstacles each config hits."""
        Q = np.atleast_2d(Q)
        m = self.m if m is None else m
        ob = self.obs
        if ob.n == 0:
            return np.zeros((len(Q), 0), bool)
        d = Q[:, None, :2] - ob.P[None, :, 0, :]
        lu = np.einsum('mnd,nd->mn', d, ob.u)
        lv = np.einsum('mnd,nd->mn', d, ob.v)
        cu = np.clip(lu, 0, ob.w[None])
        cv = np.clip(lv, 0, ob.h[None])
        hit = ((lu - cu) ** 2 + (lv - cv) ** 2) < (BASE_R + m) ** 2
        for P in self.body_polys(Q):
            hit |= ob.poly_hit(P, m, per_obj=True)
        return hit

    def wall_bad(self, Q):
        Q = np.atleast_2d(Q)
        m = self.m
        bad = (Q[:, 0] < BASE_R + m) | (Q[:, 0] > WORLD - BASE_R - m) | \\
              (Q[:, 1] < BASE_R + m) | (Q[:, 1] > WORLD - BASE_R - m)
        for P in self.body_polys(Q):
            bad |= (P < m).any(axis=(1, 2)) | (P > WORLD - m).any(axis=(1, 2))
        return bad

    def free(self, q):''')
# 2. record grasp via helper
src = src.replace('''    def _record_grasp(self):
        name = self.grasp_target
        C = self.corners(name)
        q = self.q
        c, s = math.cos(q[2]), math.sin(q[2])
        d = C - q[None, :2]
        lx = c * d[:, 0] + s * d[:, 1] - q[3]
        ly = -s * d[:, 0] + c * d[:, 1]
        self.held = name
        self.held_local = np.stack([lx, ly], axis=1)
        self.held_q = q.copy()''', '''    def local_of(self, name, q):
        C = self.corners(name)
        c, s = math.cos(q[2]), math.sin(q[2])
        d = C - q[None, :2]
        lx = c * d[:, 0] + s * d[:, 1] - q[3]
        ly = -s * d[:, 0] + c * d[:, 1]
        return np.stack([lx, ly], axis=1)

    def _record_grasp(self):
        name = self.grasp_target
        self.held = name
        self.held_local = self.local_of(name, self.q)
        self.held_q = self.q.copy()''')
# 3. place configs param
src = src.replace('''    def place_configs(self):
        R = self.corners('target_region')
        rc = R.mean(axis=0)
        rth = self.rects['target_region'][2]
        hl = self.held_local''', '''    def place_configs(self, hl=None, arm0=None):
        R = self.corners('target_region')
        rc = R.mean(axis=0)
        rth = self.rects['target_region'][2]
        if hl is None:
            hl = self.held_local
        if arm0 is None:
            arm0 = self.q[3]''')
src = src.replace("            for arm in (self.q[3], ARM_MAX, ARM_MIN, 0.15):",
                  "            for arm in (arm0, ARM_MAX, ARM_MIN, 0.125, 0.15, 0.175):")
src = src.replace("        return cfgs\n\n    def plan_place", "        return np.array(cfgs)\n\n    def plan_place")
# 4. plan_grasp with candidate filter
src = src.replace('''    def plan_grasp(self, name, time_budget=2.0):''', '''    def grasp_candidates(self, name):
        obs_all, _ = self.make_obstacles(exclude=(self.held,) if self.held else ())
        model = Model(obs_all, None)
        others, _ = self.make_obstacles(exclude=(name,))
        cands = []
        for q in self.grasp_configs(name):
            if not model.free(q):
                continue
            gp = model.body_polys(q[None])[0]
            if others.n and others.poly_hit(gp, 0.022)[0]:
                continue
            cands.append(q)
        return cands

    def plan_grasp(self, name, time_budget=2.0, cands=None):''')
src = src.replace('''        cands = []
        for q in self.grasp_configs(name):
            if not model.free(q):
                continue
            gp = Model(others, None, margin=0.022).body_polys(q[None])[0]
            if others.n and others.poly_hit(gp, 0.022)[0]:
                continue
            cands.append(q)
        if not cands:''', '''        if cands is None:
            cands = self.grasp_candidates(name)
        if not cands:''')
open('approach.py','w').write(src)
