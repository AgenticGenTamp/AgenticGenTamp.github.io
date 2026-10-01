src = open('approach.py').read()
def rep(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
rep("""            cx = b['center'][0]
            self.goal = (cx, SHELF_BOTTOM - R - 0.01)""", """            cx = b['center'][0]
            lo = max(self.sx1 + 0.074, b['poly'][:, 0].min() - 0.03, self.base_r + 0.002)
            hi = min(self.sx1 + self.sw1 - 0.074, b['poly'][:, 0].max() + 0.03, 5.0 - self.base_r - 0.002)
            cx = min(max(cx, lo), hi) if lo <= hi else min(max(cx, self.base_r + 0.002), 5.0 - self.base_r - 0.002)
            self.goal = (cx, SHELF_BOTTOM - R - 0.01)""")
# helper for block-center x interval
rep("""    def _ins_target_top(self, bx, gx, exclude=()):""", """    def _bx_interval(self, cx, cw, ox_rel, gdx=0.0):
        \"\"\"feasible block-center x interval in column centered cx when block center is
        ox_rel right of robot center and gripper is gdx right of robot center.\"\"\"
        ox = self.sx1 + 0.004
        oX = self.sx1 + self.sw1 - 0.004
        slack = max(0.0, cw / 2 - BW / 2 - 0.006)
        lo = max(cx - slack, ox + BW / 2 + 0.002, self.base_r + 0.002 + ox_rel,
                 ox + 0.072 - gdx + ox_rel)
        hi = min(cx + slack, oX - BW / 2 - 0.002, 5.0 - self.base_r - 0.002 + ox_rel,
                 oX - 0.072 - gdx + ox_rel)
        return lo, hi

    def _ins_target_top(self, bx, gx, exclude=()):""")
rep("""                slack = max(0.0, cw / 2 - BW / 2 - 0.006)
                # robot x such that block center is bx, gripper center = rx + gdx
                lo = max(cx - slack, ox + BW / 2 + 0.002)
                hi = min(cx + slack, oX - BW / 2 - 0.002)
                # gripper within opening: rx + gdx +- 0.072 within [ox, oX]; rx = bx - off0
                glo = ox + 0.072 - gdx + off[0]
                ghi = oX - 0.072 - gdx + off[0]
                lo, hi = max(lo, glo), min(hi, ghi)
                if lo > hi:""", """                lo, hi = self._bx_interval(cx, cw, off[0], gdx)
                if lo > hi:""")
# grasp option insertion feasibility
rep("""                    sgx = min(max(p[0], self.sx1 + 0.15), self.sx1 + self.sw1 - 0.15)""", """                    hd = -n
                    phi = math.pi / 2 - math.atan2(hd[1], hd[0])
                    ux = math.cos(phi) * u[0] - math.sin(phi) * u[1]
                    ox_rel = -off * ux
                    cols, cw = self._columns()
                    if not any(lo <= hi for lo, hi in
                               (self._bx_interval(cx_, cw, ox_rel) for cx_ in cols)):
                        continue
                    sgx = min(max(p[0], self.sx1 + 0.15), self.sx1 + self.sw1 - 0.15)""")
open('approach.py','w').write(src)
