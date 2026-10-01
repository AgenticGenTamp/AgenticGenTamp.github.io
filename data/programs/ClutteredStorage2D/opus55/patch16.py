src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
s=src.index("    def _plan_push(self):")
e=src.index("    def _choose_task(self):")
new='''    def _plan_push(self):
        """Return (block, target_bottom) of the next in-shelf block to lift, or None.
        Also sets self.plan_total (current capacity) and self.lazy_cols (columns that
        can be filled now without losing needed potential capacity)."""
        cols, cw = self._columns()
        n_out = sum(1 for b in self.blocks.values() if not self._inside(b))
        self.plan_total = 0
        self.lazy_cols = set(range(len(cols)))
        if n_out == 0:
            return None
        info = []
        total = 0
        tot_pot = 0
        for j, cx in enumerate(cols):
            if all(lo > hi for lo, hi in (self._bx_interval(cx, cw, ox, 0.0)
                                          for ox in (0.0, 0.07, -0.07, 0.11, -0.11))):
                continue
            ceil = self._col_ceiling(cx, cw / 2 - 0.005)
            cap = self._cap(ceil)
            zero = j in self.cap_zero and abs(self.cap_zero[j] - ceil) < 1e-3
            if zero:
                cap = 0
            members = [(n, b) for n, b in self.blocks.items() if self._inside(b)
                       and abs(b['center'][0] - cx) <= cw / 2]
            pot, push = cap, None
            if members:
                n, b = min(members, key=lambda nb: nb[1]['poly'][:, 1].min())
                if self.failed.get(n, 0) < 3:
                    bot, top = float(b['poly'][:, 1].min()), float(b['poly'][:, 1].max())
                    above = self._col_ceiling(b['center'][0], 0.145, exclude=(n,))
                    max_bot = above - 0.008 - (top - bot)
                    if max_bot >= bot + 0.02:
                        pc = self._cap(max_bot - 0.004)
                        if zero and max_bot < bot + STACK_DY:
                            pc = 0
                        if pc > cap:
                            pot = pc
                            push = (n, bot, max_bot)
            total += cap
            tot_pot += pot
            info.append((j, cx, cap, pot, zero, push))
        deficit = n_out - total
        self.plan_total = total
        if deficit <= 0:
            return None
        self.lazy_cols = {j for j, cx, cap, pot, zero, push in info
                          if cap > 0 and tot_pot - pot + cap >= n_out}
        best = None
        for j, cx, cap, pot, zero, push in info:
            if push is None:
                continue
            n, bot, max_bot = push
            k = min(pot, cap + deficit)
            R = SHELF_BOTTOM + 0.004 + k * STACK_DY + 0.004
            if zero:
                R = max(R, bot + STACK_DY)
            R = min(R, max_bot)
            gain = k - cap
            cost = -gain * 100 + abs(cx - self.rx) / 0.05
            if best is None or cost < best[0]:
                best = (cost, n, R)
        if best is None:
            self.lazy_cols = set(range(len(cols)))
            return None
        return best[1], best[2]

'''
src=src[:s]+new+src[e:]
rep("""        if pp is not None and (self.plan_total == 0 or self.ry > LAZY_PUSH_Y):""",
"""        self.allowed_cols = self.lazy_cols if pp is not None else None
        if pp is not None and (not self.lazy_cols or self.ry > LAZY_PUSH_Y):""")
rep("""            for j, cx in enumerate(cols):
                if j in self.bad_cols.get(name, ()):
                    continue""","""            for j, cx in enumerate(cols):
                if j in self.bad_cols.get(name, ()):
                    continue
                if self.allowed_cols is not None and j not in self.allowed_cols:
                    continue""")
rep("""                        for cx_ in cols:
                            lo, hi = self._bx_interval(cx_, cw, ox_rel)""","""                        for j_, cx_ in enumerate(cols):
                            if self.allowed_cols is not None and j_ not in self.allowed_cols:
                                continue
                            lo, hi = self._bx_interval(cx_, cw, ox_rel)""")
rep("""        self.cap_zero = {}""","""        self.cap_zero = {}
        self.allowed_cols = None
        self.lazy_cols = set()""")
open('approach.py','w').write(src)
