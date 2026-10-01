src=open('approach.py').read()
s=src.index("    def _plan_push(self):")
e=src.index("    def _choose_task(self):")
new='''    def _plan_push(self):
        """Return (block, target_bottom) of the next in-shelf block to lift, or None."""
        cols, cw = self._columns()
        n_out = sum(1 for b in self.blocks.values() if not self._inside(b))
        if n_out == 0:
            return None
        info = []
        total = 0
        for j, cx in enumerate(cols):
            lo, hi = self._bx_interval(cx, cw, 0.0, 0.0)
            if lo > hi:
                continue
            ceil = self._col_ceiling(cx, cw / 2 - 0.005)
            cap = self._cap(ceil)
            zero = j in self.cap_zero and abs(self.cap_zero[j] - ceil) < 1e-3
            if zero:
                cap = 0
            total += cap
            members = sorted(((n, b) for n, b in self.blocks.items() if self._inside(b)
                              and abs(b['center'][0] - cx) <= cw / 2),
                             key=lambda nb: nb[1]['poly'][:, 1].min())
            if members:
                info.append((j, cx, cap, zero, members))
        deficit = n_out - total
        if deficit <= 0:
            return None
        best = None
        for j, cx, cap, zero, members in info:
            bots = [float(b['poly'][:, 1].min()) for _, b in members]
            hs = [float(b['poly'][:, 1].max()) - bt for (_, b), bt in zip(members, bots)]
            if any(self.failed.get(n, 0) >= 3 for n, _ in members):
                continue
            chosen = None
            for k in range(cap + deficit, cap, -1):
                R = SHELF_BOTTOM + 0.004 + k * STACK_DY + 0.004
                if zero:
                    R = max(R, bots[0] + STACK_DY)
                Rs = []
                ok = True
                i = 0
                while i < len(members) and bots[i] < R - 0.001:
                    Rs.append(R)
                    R = R + hs[i] + 0.012
                    i += 1
                lim = bots[i] if i < len(members) else SHELF_TOP
                if Rs and Rs[-1] + hs[i - 1] + 0.008 > lim:
                    ok = False
                if ok and Rs:
                    chosen = (k, i, Rs)
                    break
            if chosen is None:
                continue
            k, i, Rs = chosen
            gain = k - cap
            cost = -gain * 100 + i * 30 + abs(cx - self.rx) / 0.05
            if best is None or cost < best[0]:
                best = (cost, members[i - 1][0], Rs[-1])
        if best is None:
            return None
        return best[1], best[2]

'''
src=src[:s]+new+src[e:]
open('approach.py','w').write(src)
