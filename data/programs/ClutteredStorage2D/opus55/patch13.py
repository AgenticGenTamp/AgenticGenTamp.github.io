src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""        deficit = n_out - total
        if deficit <= 0:
            return None
        best = None""","""        deficit = n_out - total
        self.plan_total = total
        if deficit <= 0:
            return None
        best = None""")
rep("""        pp = self._plan_push()
        if pp is not None:""","""        pp = self._plan_push()
        if pp is not None and (self.plan_total == 0 or self.ry > LAZY_PUSH_Y):""")
rep("""STACK_DY = 0.048""","""STACK_DY = 0.048
LAZY_PUSH_Y = 1.8""")
open('approach.py','w').write(src)
