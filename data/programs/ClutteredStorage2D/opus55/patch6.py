src = open('approach.py').read()
def rep(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
rep("""    # ------------------------------------------------------------ push task""", """    def _straight_up_ready(self, th_goal):
        if self.path is None:
            return False
        gx, gy = self.goal
        return (abs(self.rx - gx) < 0.002 and abs(wrap(th_goal - self.rth)) < 1e-4
                and self.ry <= gy + 1e-6
                and all(abs(p[0] - gx) < 0.002 for p in self.path))

    # ------------------------------------------------------------ push task""")
rep("""            a = self._goto(self.goal, R, (), th_goal=math.pi / 2, rot_R=R - 0.02)
            if a is None:
                self._abort()
                return self._act()
            if isinstance(a, str):
                self.phase = 'extend'
            else:
                return a""", """            a = 'done' if self._straight_up_ready(math.pi / 2) else \\
                self._goto(self.goal, R, (), th_goal=math.pi / 2, rot_R=R - 0.02)
            if a is None:
                self._abort()
                return self._act()
            if isinstance(a, str):
                self.phase = 'extend'
            else:
                return a""")
rep("""            if gap > 0.02:
                return self._act(darm=min(0.1, gap - 0.008), vac=1.0)""", """            if gap > 0.02:
                need = gap - 0.008
                da = min(0.1, need, self.arm_max - self.arm)
                room = SHELF_BOTTOM - self.base_r - 0.004 - self.ry
                dy = max(0.0, min(0.05, room, need - da))
                return self._act(dx=self.goal[0] - self.rx, dy=dy, darm=da, vac=1.0)""")
rep("""            a = self._goto(self.goal, self.Rc, (name,), th_goal=self.goal_th,
                           rot_R=self.Rc, vac=1.0)""", """            a = 'done' if self._straight_up_ready(self.goal_th) else \\
                self._goto(self.goal, self.Rc, (name,), th_goal=self.goal_th,
                           rot_R=self.Rc, vac=1.0)""")
rep("""            if d > 1e-3 and self.arm < self.arm_max - 1e-4 and not rejected:
                s = math.sin(self.rth)
                da = min(0.1, d / max(s, 0.5))
                dy = 0.0
                room = SHELF_BOTTOM - self.base_r - 0.004 - self.ry
                if d - da * s > 1e-3 and room > 0:
                    dy = min(0.05, room, d - da * s)
                return self._act(dy=dy, darm=da, vac=1.0)""", """            if d > 1e-3 and not rejected:
                s = math.sin(self.rth)
                da = max(0.0, min(0.1, d / max(s, 0.5), self.arm_max - self.arm))
                room = SHELF_BOTTOM - self.base_r - 0.004 - self.ry
                dy = max(0.0, min(0.05, room, d - da * s))
                if da > 1e-5 or dy > 1e-5:
                    return self._act(dy=dy, darm=da, vac=1.0)""")
open('approach.py','w').write(src)
