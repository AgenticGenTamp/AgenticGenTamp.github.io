s=open('approach.py').read()
def rep(old,new):
    global s
    assert old in s, old[:70]
    s=s.replace(old,new,1)
rep('''        if isinstance(self.plan, tuple) and not self.grasped:
            return self._grasp(self.plan[1])''','''        if isinstance(self.plan, tuple) and not self.grasped:
            if len(self.plan[1]) == 5:
                return self._grasp_flex(self.plan[1])
            return self._grasp(self.plan[1])''')
# pending check before grasp verification
rep('''        if self.grasped:
            loc = self._loc_from(self.rx, self.ry, self.rth)''','''        pg = getattr(self, "pending", None)
        if pg is not None:
            self.pending = None
            gx, gy, thg, side, L = pg
            if (not self.grasped and self.vac > 0.5 and abs(self.rx - gx) < 1e-4 and abs(self.ry - gy) < 1e-4
                    and abs(wrap(self.rth - thg)) < 1e-4 and abs(self.arm - L) < 1e-4):
                self.grasped = True
                self.rel = self._pred_rel(self.rx, self.ry)
                self.loc = self._loc_from(self.rx, self.ry, self.rth)
                self.plan_key = None
                self.plan = None
            elif not self.grasped:
                # final approach step was rejected
                self.fail_grasps += 1
                self.plan_key = None
        if self.grasped:
            loc = self._loc_from(self.rx, self.ry, self.rth)''')
rep('''    def _grasp(self, opt):''','''    def _grasp_flex(self, opt):
        gx, gy, thg, side, L = opt
        fp = (self.rx, self.ry)
        dth = wrap(thg - self.rth)
        dist = cheb(fp, (gx, gy))
        if self.stuck >= 3:
            self.fail_grasps += 1
            self.plan_key = None
        if dist <= MAXD + 1e-9 and abs(dth) <= MAXTH + 1e-9 and self.stuck == 0:
            self.pending = opt
            return self._act([gx - self.rx, gy - self.ry, dth, L - self.arm], 1.0)
        cost, wp = self._nav(fp, (gx, gy))
        dth_c = max(-MAXTH, min(MAXTH, dth))
        darm = max(-0.1, self.br - self.arm)
        if abs(darm) < 1e-5:
            darm = 0.0
        if dist <= 1e-4:
            self.rot_wait = getattr(self, "rot_wait", 0) + 1
        return self._act(self._move_action(wp[0], wp[1], dth_c, darm), 0.0)

    def _grasp(self, opt):''')
rep('''        self.fail_grasps = 0
''','''        self.fail_grasps = 0
        self.pending = None
''')
open('approach.py','w').write(s)
