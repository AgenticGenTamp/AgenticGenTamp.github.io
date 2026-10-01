src=open('approach.py').read()
old="""            lo, hi = self._bx_interval(cx, cw, 0.0, 0.0)
            if lo > hi:
                continue
            ceil = self._col_ceiling(cx, cw / 2 - 0.005)"""
new="""            if all(lo > hi for lo, hi in (self._bx_interval(cx, cw, ox, 0.0)
                                          for ox in (0.0, 0.07, -0.07, 0.11, -0.11))):
                continue
            ceil = self._col_ceiling(cx, cw / 2 - 0.005)"""
assert src.count(old)==1
open('approach.py','w').write(src.replace(old,new))
