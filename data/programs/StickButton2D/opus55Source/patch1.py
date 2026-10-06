src=open('approach.py').read()
def rep(old,new,cnt=1):
    global src
    assert src.count(old)>=1, old[:80]
    src=src.replace(old,new) if cnt==0 else src.replace(old,new,cnt)

# --- grasp waypoints with variants ---
start=src.index('    def grasp_waypoints(self, p):')
end=src.index('    # ================= holding the stick')
src=src[:start]+'''    def grasp_pose(self, v, p, ty=None):
        """Return (W, G, theta) for grasp variant v, or None if infeasible."""
        sb = self.sy
        if v == "G":
            scx = self.sx + STICK_W / 2
            if not (HX_LO + 0.002 <= scx <= HX_HI - 0.002) or sb - GRASP_D < HY_LO:
                return None
            return (scx, sb - 0.14), (scx, sb - GRASP_D), np.pi / 2
        if ty is None:
            ty = float(min(max(p[1], sb + 0.01), PY_HI))
        if v == "GL":  # robot right of the stick, facing left
            tx = self.sx + STICK_W + GRASP_D
            if tx + 0.03 > PX_HI:
                return None
            return (tx + 0.03, ty), (tx, ty), np.pi
        tx = self.sx - GRASP_D  # "GR": robot left of the stick, facing right
        if tx - 0.03 < PX_LO:
            return None
        return (tx - 0.03, ty), (tx, ty), 0.0

    def grasp_waypoints(self, p, v="G", ty=None):
        """(waypoints excluding ones equal to p, theta) or (None, None)."""
        gp = self.grasp_pose(v, p, ty)
        if gp is None:
            return None, None
        W, G, th = gp
        # already on the final approach segment W->G?
        wx, wy = G[0] - W[0], G[1] - W[1]
        L2 = wx * wx + wy * wy
        ux = ((p[0] - W[0]) * wx + (p[1] - W[1]) * wy) / L2
        px_ = W[0] + ux * wx - p[0]
        py_ = W[1] + ux * wy - p[1]
        if -1e-6 <= ux <= 1 + 1e-6 and abs(px_) < 1e-7 and abs(py_) < 1e-7:
            if abs(p[0] - G[0]) < 1e-7 and abs(p[1] - G[1]) < 1e-7:
                return [], th
            return [G], th
        sb = self.sy
        low = max(PY_LO, sb - 0.26)
        options = [[W], [(p[0], low), (W[0], low), W]]
        for side in (-1, 1):
            ex = (self.sx - 0.135) if side < 0 else (self.sx + STICK_W + 0.135)
            if PX_LO <= ex <= PX_HI:
                options.append([(ex, p[1]), (ex, low), (W[0], low), W])
        for opt in options:
            q = p
            ok = True
            for w in opt:
                if self.collides(path_points(q, w)):
                    ok = False
                    break
                q = w
            if ok:
                out = list(opt) + [G]
                while out and abs(out[0][0] - p[0]) < 1e-7 and abs(out[0][1] - p[1]) < 1e-7:
                    out.pop(0)
                return out, th
        return None, None

'''+src[end:]

# --- theta grid relative to stick-vertical orientation ---
rep('''        T = np.concatenate([[c[2]], TH_GRID]) if rotate else np.array([c[2]])
        relx, rely, phi = h''','''        relx, rely, phi = h
        T = np.concatenate([[c[2]], TH_GRID - phi]) if rotate else np.array([c[2]])''')

# --- sim ---
rep('''            if item == "G":
                if holding:
                    continue
                p = (c[0], c[1])
                scx = world.sx + STICK_W / 2
                if not (HX_LO + 0.002 <= scx <= HX_HI - 0.002) or world.sy - GRASP_D < HY_LO:
                    return np.inf, None
                wps = world.grasp_waypoints(p)
                for k, w in enumerate(wps):''','''            if isinstance(item, str):
                if holding:
                    continue
                p = (c[0], c[1])
                wps, gth = world.grasp_waypoints(p, item)
                if wps is None:
                    return np.inf, None
                for k, w in enumerate(wps):''')
rep('''                        t = max(t, t_rot0)
                    p = w
                t += 1
                holding = True
                h = (GRASP_D, 0.0, 0.0)
                c = (p[0], p[1], np.pi / 2)
                out.append("G")''','''                        t = max(t, t_rot0[item])
                    p = w
                if len(wps) <= 1:
                    t = max(t, t_rot0[item])
                t += 1
                holding = True
                bcx, bcy = world.sx + STICK_W / 2 - p[0], world.sy - p[1]
                cg, sg = np.cos(gth), np.sin(gth)
                h = (cg * bcx + sg * bcy, -sg * bcx + cg * bcy, wrap(np.pi / 2 - gth))
                c = (p[0], p[1], gth)
                out.append(item)''')
# --- make_plan ---
rep('''        t_rot = int(np.ceil(abs(wrap(np.pi / 2 - r["th"])) / MAXTH - 1e-9))''',
'''        t_rot = {v: int(np.ceil(abs(wrap(th - r["th"])) / MAXTH - 1e-9))
                 for v, th in (("G", np.pi / 2), ("GL", np.pi), ("GR", 0.0))}''')
rep('''        prefixes = [[]] if holding else ([["G"]] + [[i, "G"] for i in items] + [[]])''',
'''        GV = ("G", "GL", "GR")
        if holding:
            prefixes = [[]]
        else:
            prefixes = [[v] for v in GV] + [[]] + [[i, v] for v in GV for i in items]''')
rep('''            best_seq = (["G"] if not holding else []) + items''','''            best_seq = (["G"] if not holding else []) + items''')
rep('''        if not holding and "G" not in cur:
            cur = cur + ["G"]''','''        if not holding and not any(isinstance(x, str) for x in cur):
            cur = cur + ["G"]''')
rep('''            mv = rng.integers(3)
            i, j = sorted(rng.choice(n, 2, replace=False))
            if mv == 0:''','''            mv = rng.integers(4)
            i, j = sorted(rng.choice(n, 2, replace=False))
            if mv == 3:
                gi = [k for k, x in enumerate(new) if isinstance(x, str)]
                if not gi:
                    continue
                new[gi[0]] = GV[rng.integers(3)]
            elif mv == 0:''')
# --- follow ---
rep('''            if item == "G":
                if self.holding or all(b["pressed"] for b in btns):
                    self.plan.pop(0)
                    continue
                return self._grasp_step(r, s, rejected)''','''            if isinstance(item, str):
                if self.holding or all(b["pressed"] for b in btns):
                    self.plan.pop(0)
                    continue
                return self._grasp_step(r, s, rejected, item)''')
rep('''            dth = 0.0 if rejected else wrap(np.pi / 2 - r["th"])
            dx, dy = tgt[0] - p[0], tgt[1] - p[1]''','''            gv = [x for x in self.plan if isinstance(x, str)]
            gth = {"G": np.pi / 2, "GL": np.pi, "GR": 0.0}[gv[0]] if gv else np.pi / 2
            dth = 0.0 if rejected else wrap(gth - r["th"])
            dx, dy = tgt[0] - p[0], tgt[1] - p[1]''')
rep('''        if not self.holding and "G" not in self.plan:
            self.plan.append("G")''','''        if not self.holding and not any(isinstance(x, str) for x in self.plan):
            self.plan.append("G")''')
# --- grasp step ---
start=src.index('    # ---------- grasping from below ----------')
src=src[:start]+'''    # ---------- grasping ----------
    def _grasp_step(self, r, s, rejected, v):
        if self.grasp_cmd:
            return self._act(0.0, 0.0, 0.0, 1.0)
        p = (r["x"], r["y"])
        world = World([], [], (s["x"], s["y"]))
        if self.grasp_key != (v, len(self.plan)):
            self.grasp_key = (v, len(self.plan))
            gp = world.grasp_pose(v, p)
            self.grasp_ty = None if gp is None else gp[1][1]
        wps, gth = world.grasp_waypoints(p, v, self.grasp_ty)
        if wps is None:
            # infeasible variant: fall back to grasping from below / replanning
            self.plan.pop(0)
            if v != "G":
                self.plan.insert(0, "G")
            return self._follow(r, s, self._last_btns, False, 1)
        dth = wrap(gth - r["th"])
        W = world.grasp_pose(v, p, self.grasp_ty)[0]
        if not wps:
            if abs(dth) < 1e-7:
                self.grasp_cmd = True
                return self._act(0.0, 0.0, 0.0, 1.0)
            return self._act(W[0] - p[0], W[1] - p[1], 0.0, 0.0)
        tgt = wps[0]
        if len(wps) == 1 and abs(dth) > 1e-7:
            if rejected:
                return self._act(W[0] - p[0], W[1] - p[1], 0.0, 0.0)
            return self._act(0.0, 0.0, dth, 0.0)
        dth_cmd = 0.0 if rejected else dth
        dx, dy = tgt[0] - p[0], tgt[1] - p[1]
        if rejected and abs(self.prev_action[2]) < 1e-9:
            dx, dy = self.rng.uniform(-STEP, STEP, size=2)
        return self._act(dx, dy, dth_cmd, 0.0)
'''
# --- reset: relocation only as fallback ---
rep('''        if scx < HX_LO + 0.002 or scx > HX_HI - 0.002:
            self.phase = "side_approach"''','''        self.grasp_key = None
        self.grasp_ty = None
        self._last_btns = btns
        if False:
            self.phase = "side_approach"''')
rep('''        if self.plan is None:
            self._make_plan(r, s, btns)
        return self._follow(r, s, btns, rejected)''','''        self._last_btns = btns
        if self.plan is None:
            self._make_plan(r, s, btns)
            if not np.isfinite(self.plan_cost) and not self.holding and not self.relocated:
                scx = s["x"] + STICK_W / 2
                self.relocated = True
                self.phase = "side_approach"
                self.side_d = 1.0 if scx < XMAX / 2 else -1.0
                a = self._side_step(r, s)
                if a is not None:
                    return a
        return self._follow(r, s, btns, rejected)''')
rep('''        self.grasp_key = None''','''        self.grasp_key = None
        self.relocated = False''')
open('approach.py','w').write(src)
