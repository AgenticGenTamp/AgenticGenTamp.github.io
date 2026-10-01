src = open('approach.py').read()
def rep(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
rep("""                if best is None or score < best[0]:
                    best = (score, bx, gx, gy, j)""", """                if best is None or score < best[0]:
                    best = (score, bx, gx, gy, j, t_top)""")
rep("""            _, self.cx, gx, gy, self.col_j = best
            self.goal = (float(np.clip(gx, self.Rc, 5 - self.Rc)), gy)""", """            _, self.cx, gx, gy, self.col_j, t_top = best
            gx = float(np.clip(gx, self.Rc, 5 - self.Rc))
            # lowest robot y from which the arm alone can insert (block top offset off[1]+BH/2)
            gy_low = t_top - (off[1] + BH / 2) - (self.arm_max - self.base_r - 0.03)
            gy_try = min(gy, max(gy_low, self.ry + abs(gx - self.rx)))
            if gy_try < gy - 0.02:
                ok = all(self._clearance(gx, yy, (name,)) >= self.Rc
                         for yy in np.linspace(gy_try, gy, 12))
                # corridor above robot for arm+block must be free of floor blocks
                if ok:
                    corr = np.array([[gx - 0.16 + min(0, off[0]), gy_try],
                                     [gx + 0.16 + max(0, off[0]), gy_try],
                                     [gx + 0.16 + max(0, off[0]), SHELF_BOTTOM],
                                     [gx - 0.16 + min(0, off[0]), SHELF_BOTTOM]])
                    for m, ob in self.blocks.items():
                        if m != name and not self._inside(ob) and polys_overlap(corr, ob['poly']):
                            ok = False
                            break
                if ok:
                    gy = float(gy_try)
            self.goal = (gx, gy)""")
open('approach.py','w').write(src)
