src = open('approach.py').read()
def rep(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
start = src.index("    def _choose_task(self):")
end = src.index("        outs = [(n, b) for n, b in self.blocks.items() if not self._inside(b)]", start)
new = '''    def _cap(self, ceil):
        return max(0, int(math.floor((ceil - (SHELF_BOTTOM + 0.004)) / STACK_DY)))

    def _plan_push(self):
        """Return (block, target_bottom) of the next in-shelf block to lift, or None."""
        cols, cw = self._columns()
        n_out = sum(1 for b in self.blocks.values() if not self._inside(b))
        if n_out == 0:
            return None
        info = []
        total = 0
        for cx in cols:
            ceil = self._col_ceiling(cx, cw / 2 - 0.005)
            cap = self._cap(ceil)
            total += cap
            members = [(n, b) for n, b in self.blocks.items() if self._inside(b)
                       and abs(b['center'][0] - cx) <= cw / 2]
            if not members:
                continue
            n, b = min(members, key=lambda nb: nb[1]['poly'][:, 1].min())
            if self.failed.get(n, 0) >= 3:
                continue
            bot, top = b['poly'][:, 1].min(), b['poly'][:, 1].max()
            above = self._col_ceiling(b['center'][0], 0.145, exclude=(n,))
            max_lift = (above - 0.008) - top
            if max_lift < 0.02:
                continue
            cap_after = self._cap(bot + max_lift)
            if cap_after > cap:
                info.append((n, b, cx, cap, cap_after, bot))
        deficit = n_out - total
        if deficit <= 0 or not info:
            return None
        def cost(it):
            gain = min(deficit, it[4] - it[3])
            return -gain * 100 + abs(it[2] - self.rx) / 0.05
        n, b, cx, cap, cap_after, bot = min(info, key=cost)
        k = cap + min(deficit, cap_after - cap)
        target_bottom = SHELF_BOTTOM + 0.004 + k * STACK_DY + 0.004
        return n, target_bottom

    def _choose_task(self):
        pp = self._plan_push()
        if pp is not None:
            self.push_target_bottom = pp[1]
            return ('push', pp[0])
'''
src = src[:start] + new + src[end:]
rep("""            target = min(ceil - 0.008, top + self._need_bottom() - b['poly'][:, 1].min())""",
    """            target = min(ceil - 0.008, top + self.push_target_bottom - b['poly'][:, 1].min())""")
rep("""DEBUG = False""", """DEBUG = False
STACK_DY = 0.048""")
rep("""                if t_top - BH < SHELF_BOTTOM + 0.012:""", """                if t_top - BH < SHELF_BOTTOM + 0.004:""")
open('approach.py','w').write(src)
