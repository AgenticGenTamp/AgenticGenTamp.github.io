src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""            cap = self._cap(ceil)
            if j in self.cap_zero and abs(self.cap_zero[j] - ceil) < 1e-3:
                cap = 0""","""            cap = self._cap(ceil)
            zero = j in self.cap_zero and abs(self.cap_zero[j] - ceil) < 1e-3
            if zero:
                cap = 0""")
rep("""            if cap_after > cap:
                info.append((n, b, cx, cap, cap_after, bot))""","""            if zero and bot + STACK_DY > bot + max_lift:
                continue
            if cap_after > cap:
                info.append((n, b, cx, cap, cap_after, bot, zero))""")
rep("""        n, b, cx, cap, cap_after, bot = min(info, key=cost)
        k = cap + min(deficit, cap_after - cap)
        target_bottom = SHELF_BOTTOM + 0.004 + k * STACK_DY + 0.004""","""        n, b, cx, cap, cap_after, bot, zero = min(info, key=cost)
        k = cap + min(deficit, cap_after - cap)
        target_bottom = SHELF_BOTTOM + 0.004 + k * STACK_DY + 0.004
        if zero:
            target_bottom = max(target_bottom, bot + STACK_DY)""")
open('approach.py','w').write(src)
