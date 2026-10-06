s = open('approach.py').read()
old = """        nX, nY, nT = node_ok.shape
        # start
        x0, y0, th0 = d["robot"]
        si = int(round((x0 - gx) / GRID)) - ix[0]
        sj = int(round((y0 - gy) / GRID)) - iy[0]
        sk = int(round(wrap(th0 - gth) / DTH)) % NTH
        si = min(max(si, 0), nX - 1)
        sj = min(max(sj, 0), nY - 1)
        gi, gj = -ix[0], -iy[0]
        if not (0 <= gi < nX and 0 <= gj < nY) or not node_ok[gi, gj, 0]:
            return None
        start = (si, sj, sk)
        goaln = (gi, gj, 0)
"""
new = """        nX, nY, nT = node_ok.shape
        # goal: grid node nearest to the exact goal pose (grid aligned w/ start xy)
        x0, y0, th0 = d["robot"]
        si, sj = -ix[0], -iy[0]
        if not (0 <= si < nX and 0 <= sj < nY):
            return None
        gi = int(round((gx - x0) / GRID)) - ix[0]
        gj = int(round((gy - y0) / GRID)) - iy[0]
        if not (0 <= gi < nX and 0 <= gj < nY) or not node_ok[gi, gj, 0]:
            return None
        # start theta: snap onto goal-aligned theta grid, checking the rotation
        kf = wrap(th0 - gth) / DTH
        first = None
        for sk in sorted({int(math.floor(kf)) % NTH, int(math.ceil(kf)) % NTH},
                         key=lambda k: abs(wrap(gth + k * DTH - th0))):
            thk = wrap(gth + sk * DTH)
            ths = th0 + np.linspace(0, 1, 6) * wrap(thk - th0)
            good = all(self._pose_valid(d, (x0, y0, float(tt)), T, arm, btn, margin * 0.5)
                       for tt in ths[1:])
            if good and node_ok[si, sj, sk]:
                first = sk
                break
        if first is None:
            return None
        sk = first
        start = (si, sj, sk)
        goaln = (gi, gj, 0)
"""
assert old in s; s = s.replace(old, new)
old2 = """        out = []
        for i, j, k in path:
            out.append((float(XS[i]), float(YS[j]), float(wrap(gth + k * DTH))))
        out[-1] = (gx, gy, gth)
        return out"""
new2 = """        out = []
        for i, j, k in path:
            out.append((float(XS[i]), float(YS[j]), float(wrap(gth + k * DTH))))
        out.append((gx, gy, gth))
        return out"""
assert old2 in s; s = s.replace(old2, new2)
# XS/YS aligned with start
old3 = """        gx, gy, gth = goal
        xlo, xhi = rad + 0.005, WORLD_X[1] - rad - 0.005
        ylo, yhi = rad + 0.005, TABLE_Y - rad - 0.003
        ix = np.arange(int(math.ceil((xlo - gx) / GRID)), int(math.floor((xhi - gx) / GRID)) + 1)
        iy = np.arange(int(math.ceil((ylo - gy) / GRID)), int(math.floor((yhi - gy) / GRID)) + 1)
        XS = gx + ix * GRID
        YS = gy + iy * GRID"""
new3 = """        gx, gy, gth = goal
        sx0, sy0 = d["robot"][0], d["robot"][1]
        xlo, xhi = rad + 0.005, WORLD_X[1] - rad - 0.005
        ylo, yhi = rad + 0.005, TABLE_Y - rad - 0.003
        ix = np.arange(int(math.ceil((xlo - sx0) / GRID)), int(math.floor((xhi - sx0) / GRID)) + 1)
        iy = np.arange(int(math.ceil((ylo - sy0) / GRID)), int(math.floor((yhi - sy0) / GRID)) + 1)
        if len(ix) == 0 or len(iy) == 0 or ix[0] > 0 or iy[0] > 0 or ix[-1] < 0 or iy[-1] < 0:
            return None
        XS = sx0 + ix * GRID
        YS = sy0 + iy * GRID"""
assert old3 in s; s = s.replace(old3, new3)
# regrasp after many failures
old4 = """    def _plan_push(self, d):
        T = self.T"""
new4 = """    def _plan_push(self, d):
        if self.fail_count > 4:
            return False
        T = self.T"""
assert old4 in s; s = s.replace(old4, new4)
old5 = """            if not ok:
                self.fail_count += 1
                if self.fail_count > 3:"""
new5 = """            if not ok:
                self.fail_count += 1
                if self.fail_count > 5:"""
assert old5 in s; s = s.replace(old5, new5)
open('approach.py','w').write(s)
