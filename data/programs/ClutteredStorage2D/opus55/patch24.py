src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""            _, p, th, D = opts[0]
            self.goal = (float(p[0]), float(p[1]))
            self.goal_th = th
            self.D = D
            self.phase = 'nav'
            self.path = None
        if self.phase == 'nav':
            if self.arm > self.base_r + 1e-3:
                return self._act(darm=-0.1)
            a = self._goto(self.goal, Rn, (), th_goal=self.goal_th, rot_R=self.base_r + 0.03)
            if a is None:
                self._abort()
                return self._act()""","""            _, p, th, D = opts[0]
            self.goal = (float(p[0]), float(p[1]))
            self.goal_th = th
            self.D = D
            self.opts = opts
            self.opt_i = 0
            self.phase = 'nav'
            self.path = None
        if self.phase == 'nav':
            if self.arm > self.base_r + 1e-3:
                return self._act(darm=-0.1)
            a = self._goto(self.goal, Rn, (), th_goal=self.goal_th, rot_R=self.base_r + 0.03)
            tries = 0
            while a is None and self.opt_i + 1 < len(self.opts) and tries < 6:
                # current standoff unreachable: try the next grasp option
                self.opt_i += 1
                tries += 1
                _, p, th, D = self.opts[self.opt_i]
                self.goal = (float(p[0]), float(p[1]))
                self.goal_th = th
                self.D = D
                self.path = None
                a = self._goto(self.goal, Rn, (), th_goal=self.goal_th, rot_R=self.base_r + 0.03)
            if a is None:
                self._abort()
                return self._act()""")
open('approach.py','w').write(src)
