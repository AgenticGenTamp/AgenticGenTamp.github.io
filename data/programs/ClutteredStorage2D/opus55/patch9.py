src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""            if best is None:
                self._abort()
                return self._act(vac=0.0)
            _, self.cx, gx, gy, self.col_j, t_top = best""","""            if best is None:
                cols_, cw_ = self._columns()
                for j_, cx_ in enumerate(cols_):
                    self.cap_zero[j_] = self._col_ceiling(cx_, cw_ / 2 - 0.005)
                self._abort()
                return self._act(vac=0.0)
            _, self.cx, gx, gy, self.col_j, t_top = best""")
rep("""        self.bad_cols = {}""","""        self.bad_cols = {}
        self.cap_zero = {}""")
rep("""        for cx in cols:
            ceil = self._col_ceiling(cx, cw / 2 - 0.005)
            cap = self._cap(ceil)""","""        for j, cx in enumerate(cols):
            ceil = self._col_ceiling(cx, cw / 2 - 0.005)
            cap = self._cap(ceil)
            if j in self.cap_zero and abs(self.cap_zero[j] - ceil) < 1e-3:
                cap = 0""")
open('approach.py','w').write(src)
