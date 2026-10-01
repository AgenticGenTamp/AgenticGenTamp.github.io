src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""            a = self._goto(self.goal, Rn, (), th_goal=self.goal_th, rot_R=self.base_r + 0.03)
            if a is None:
                self._abort()
                return self._act()
            if isinstance(a, str):
                self.phase = 'extend'
                self.ext_steps = 0
            else:
                return a
        if self.phase == 'extend':""","""            a = self._goto(self.goal, Rn, (), th_goal=self.goal_th, rot_R=self.base_r + 0.03)
            if a is None:
                self._abort()
                return self._act()
            if isinstance(a, str):
                self.phase = 'extend'
                self.ext_steps = 0
            else:
                # final step: extend the arm concurrently
                gx, gy = self.goal
                if (self.path is not None and len(self.path) == 1
                        and abs(self.rx + float(a[0]) - gx) < 1e-5 and abs(self.ry + float(a[1]) - gy) < 1e-5
                        and abs(wrap(self.rth + float(a[2]) - self.goal_th)) < 1e-5):
                    h = np.array([math.cos(self.goal_th), math.sin(self.goal_th)])
                    gap = float(((b['poly'] - np.array([gx, gy])) @ h).min()) - (self.arm + 0.01)
                    if gap > 0.02:
                        a = a.copy()
                        a[3] = min(0.1, gap - 0.008)
                        a[4] = 1.0
                        self.phase = 'extend'
                        self.ext_steps = 1
                return a
        if self.phase == 'extend':
            if math.hypot(self.rx - self.goal[0], self.ry - self.goal[1]) > 1e-4:
                self.phase = 'nav'
                self.path = None
                return self._act(darm=-(self.arm - self.base_r), vac=0.0)""")
open('approach.py','w').write(src)
