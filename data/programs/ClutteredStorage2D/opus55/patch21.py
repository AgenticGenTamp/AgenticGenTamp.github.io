src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""                if da > 1e-5 or dy > 1e-5:
                    return self._act(dy=dy, darm=da, vac=1.0)""","""                if da > 1e-5 or dy > 1e-5:
                    if da * s + dy >= d - 1e-4 and bot + da * s + dy > SHELF_BOTTOM + 0.004:
                        # final step: move then release in the same step
                        self.task = None
                        self.phase = 'retreat'
                        self.retreat_n = 0
                        return self._act(dy=dy, darm=da, vac=0.0)
                    return self._act(dy=dy, darm=da, vac=1.0)""")
rep("""            if d > 1e-3 and self.arm < self.arm_max - 1e-4:
                return self._act(darm=min(0.1, d), vac=1.0)""","""            if d > 1e-3 and self.arm < self.arm_max - 1e-4:
                if d <= 0.1 and self.arm + d <= self.arm_max:
                    self.task = None
                    self.phase = 'retreat'
                    self.retreat_n = 0
                    return self._act(darm=d, vac=0.0)
                return self._act(darm=min(0.1, d), vac=1.0)""")
open('approach.py','w').write(src)
