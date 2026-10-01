src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""                    cols, cw = self._columns()
                    if not any(lo <= hi for lo, hi in
                               (self._bx_interval(cx_, cw, ox_rel) for cx_ in cols)):
                        continue""","""                    key = round(ox_rel, 3)
                    if key not in feas_cache:
                        cols, cw = self._columns()
                        fz, rm = False, False
                        for cx_ in cols:
                            lo, hi = self._bx_interval(cx_, cw, ox_rel)
                            if lo > hi:
                                continue
                            fz = True
                            bx_ = min(max(cx_, lo), hi)
                            if self._ins_target_top(bx_, bx_ - ox_rel) - BH >= SHELF_BOTTOM + 0.004:
                                rm = True
                        feas_cache[key] = (fz, rm)
                    fz, rm = feas_cache[key]
                    if not fz:
                        continue""")
rep("""                    opts.append((cost, p, math.atan2(-n[1], -n[0]), D))
        opts.sort(key=lambda o: o[0])
        return opts""","""                    (opts if rm else opts_noroom).append((cost, p, math.atan2(-n[1], -n[0]), D))
        if not opts:
            opts = opts_noroom
        opts.sort(key=lambda o: o[0])
        return opts""")
src = src.replace("""    def _grasp_options(self, name):
""", """    def _grasp_options(self, name):
        feas_cache = {}
        opts_noroom = []
""")
open('approach.py','w').write(src)
