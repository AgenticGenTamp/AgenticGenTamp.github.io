src=open('approach.py').read()
old="""                lo, hi = self._bx_interval(cx, cw, off[0], gdx)
                if lo > hi:"""
new="""                lo, hi = self._bx_interval(cx, cw, off[0], gdx)
                if DEBUG:
                    bx_ = min(max(cx, lo), hi)
                    print('col', j, round(lo, 3), round(hi, 3), 'off', off.round(3), 'ttop', round(self._ins_target_top(bx_, bx_ - off[0] + gdx), 3))
                if lo > hi:"""
assert src.count(old)==1
open('approach.py','w').write(src.replace(old,new))
