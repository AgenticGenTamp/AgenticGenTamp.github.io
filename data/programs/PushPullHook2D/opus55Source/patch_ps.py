s=open('approach.py').read()
old="""        dth = wrap(Bt[2] - th)
        dth = float(np.clip(dth, -0.1, 0.1))
        dx = float(np.clip(Bt[0] - x, -0.05, 0.05))
        dy = float(np.clip(Bt[1] - y, -0.05, 0.05))
        return self._act(dx, dy, dth, 0.0, 1.0)"""
new="""        dth = wrap(Bt[2] - th)
        dth = float(np.clip(dth, -0.1, 0.1))
        dx = float(np.clip(Bt[0] - x, -0.05, 0.05))
        dy = float(np.clip(Bt[1] - y, -0.05, 0.05))
        for (ax, ay, at) in ((dx, dy, dth), (dx, dy, 0.0), (dx, dy, dth * 0.5),
                             (0.0, 0.0, dth), (dx * 0.5, dy * 0.5, dth)):
            P = (x + ax, y + ay, wrap(th + at))
            if self._pose_valid(d, P, T, self.arm, btn, 0, check_btn=False):
                return self._act(ax, ay, at, 0.0, 1.0)
        self.stuck = 0
        self.mode = "plan_push"
        self.fail_count += 1
        return None"""
assert old in s
s=s.replace(old,new)
open('approach.py','w').write(s)
