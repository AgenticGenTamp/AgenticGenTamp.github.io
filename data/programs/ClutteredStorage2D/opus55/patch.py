src = open('approach.py').read()
start = src.index("            cols, cw = self._columns()\n            best = None\n            for cx in cols:")
end = src.index("            self.task = None\n            self.phase = 'retreat'\n            self.retreat_n = 0\n            return self._act(vac=0.0)\n        return self._act()", start)
new = '''            cols, cw = self._columns()
            ox = self.sx1 + 0.004
            oX = self.sx1 + self.sw1 - 0.004
            gdx = 0.5 * math.cos(th_t)  # gripper center x offset from robot (approx)
            best = None
            for j, cx in enumerate(cols):
                if j in self.bad_cols.get(name, ()):
                    continue
                slack = max(0.0, cw / 2 - BW / 2 - 0.006)
                # robot x such that block center is bx, gripper center = rx + gdx
                lo = max(cx - slack, ox + BW / 2 + 0.002)
                hi = min(cx + slack, oX - BW / 2 - 0.002)
                # gripper within opening: rx + gdx +- 0.072 within [ox, oX]; rx = bx - off0
                glo = ox + 0.072 - gdx + off[0]
                ghi = oX - 0.072 - gdx + off[0]
                lo, hi = max(lo, glo), min(hi, ghi)
                if lo > hi:
                    continue
                bx = min(max(cx, lo), hi)
                gx = bx - off[0]
                t_top = self._ins_target_top(bx, gx + gdx)
                if t_top - BH < SHELF_BOTTOM + 0.012:
                    continue
                space = t_top - SHELF_BOTTOM
                gy = SHELF_BOTTOM - self.Rc - 0.01
                score = -space * 10 + math.hypot(gx - self.rx, gy - self.ry) * 0.1
                if best is None or score < best[0]:
                    best = (score, bx, gx, gy, j)
            if best is None:
                self._abort()
                return self._act(vac=0.0)
            _, self.cx, gx, gy, self.col_j = best
            self.goal = (float(np.clip(gx, self.Rc, 5 - self.Rc)), gy)
            self.goal_th = th_t
            self.path = None
            self.phase = 'carry'
        if self.phase == 'carry':
            a = self._goto(self.goal, self.Rc, (name,), th_goal=self.goal_th,
                           rot_R=self.Rc, vac=1.0)
            if a is None:
                self._abort()
                return self._act(vac=0.0)
            if not isinstance(a, str):
                return a
            self.phase = 'align'
        if self.phase == 'align':
            ex = self.cx - b['center'][0]
            eth = wrap(self.goal_th - self.rth)
            if abs(ex) > 1e-4 or abs(eth) > 1e-5:
                return self._act(dx=ex, dth=eth, vac=1.0)
            self.phase = 'insert'
            self.ins_last = None
        if self.phase == 'backout':
            if self.arm > self.base_r + 1e-3 or self.ry > SHELF_BOTTOM - self.Rc - 0.01:
                dy = -min(0.05, max(0.0, self.ry - (SHELF_BOTTOM - self.Rc - 0.01)))
                return self._act(dy=dy, darm=-0.1, vac=1.0)
            self.phase = 'carry_plan'
            return self._act(vac=1.0)
        if self.phase == 'insert':
            top = b['poly'][:, 1].max()
            bot = b['poly'][:, 1].min()
            gxc = self.rx + self.arm * math.cos(self.rth)
            target = self._ins_target_top(b['center'][0], gxc, exclude=(name,))
            d = target - top
            rejected = self.ins_last is not None and abs(top - self.ins_last) < 1e-7
            self.ins_last = top
            if d > 1e-3 and self.arm < self.arm_max - 1e-4 and not rejected:
                s = math.sin(self.rth)
                da = min(0.1, d / max(s, 0.5))
                dy = 0.0
                room = SHELF_BOTTOM - self.base_r - 0.004 - self.ry
                if d - da * s > 1e-3 and room > 0:
                    dy = min(0.05, room, d - da * s)
                return self._act(dy=dy, darm=da, vac=1.0)
            if bot <= SHELF_BOTTOM + 0.002:
                # could not insert: back out holding the block, try another column
                self.bad_cols.setdefault(name, set()).add(self.col_j)
                self.phase = 'backout'
                self.stuck = 0
                return self._act(darm=-0.1, vac=1.0)
'''
src = src[:start] + new + src[end:]
src = src.replace('''    def _choose_task(self):''', '''    def _ins_target_top(self, bx, gx, exclude=()):
        """highest allowed top y for a horizontal block centered at bx pushed by gripper at gx."""
        cb = self._col_ceiling(bx, BW / 2 + 0.004, exclude)
        cg = self._col_ceiling(gx, 0.075, exclude)
        return min(cb - 0.006, cg - 0.008 + BH)

    def _choose_task(self):''')
src = src.replace('''        self.failed = {}
''', '''        self.failed = {}
        self.bad_cols = {}
''')
src = src.replace("+ 2 * math.ceil((D - 0.23) / 0.1) + abs(off)", "+ 2 * math.ceil((D - 0.23) / 0.1) + abs(off) * 50")
open('approach.py','w').write(src)
