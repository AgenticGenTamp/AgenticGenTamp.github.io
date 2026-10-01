src=open('approach.py').read()
old="""        a = self._policy()
        self.last_nonzero = bool(np.any(np.abs(a[:4]) > 1e-6))
        return a"""
new="""        a = self._policy()
        self.last_nonzero = bool(np.any(np.abs(a[:4]) > 1e-6))
        # watchdog: many consecutive idle steps -> random nudge to break deadlocks
        self.idle = 0 if (self.last_nonzero or abs(float(a[4]) - self.vac) > 1e-6) else self.idle + 1
        if self.idle > 10:
            ang = self.rng.uniform(-math.pi, math.pi)
            a = self._act(dx=0.05 * math.cos(ang), dy=0.05 * math.sin(ang), darm=-0.1, vac=0.0)
            self.last_nonzero = True
            self.task = None
            self.phase = None
        return a"""
assert src.count(old)==1
src=src.replace(old,new)
old2="""        self.cap_zero = {}"""
src=src.replace(old2, old2+"""
        self.idle = 0
        self.rng = np.random.default_rng(0)""")
open('approach.py','w').write(src)
