"""Approach for Obstruction2DEnv: clear obstructions off the target surface,
then pick the target block and place it on the surface.

Robot is kept pointing straight down (theta=-pi/2) with arm fully extended.
Motions between key poses are planned with a small collision-checked
path search (plateau paths with diagonal rise/fall)."""
import math
import numpy as np

FLOOR = 0.1
X_MIN, X_MAX = 0.1001, 1.5179   # robot base x limits
Y_MAX = 0.8995                  # robot base y limit
WORLD_X = 1.618
GAP = 0.0015                    # gripper gap above object when grasping
PLACE_GAP = 0.004               # object gap above floor when releasing
SPEED = 0.05
ARM_SPEED = 0.1
BLOCK_Z = 0.019                 # block bottom height above surface when placing (tol 0.025)
STAGE_GAIN = 1.0


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _seg_steps(a, b):
    v = max(abs(a[0] - b[0]) / SPEED, abs(a[1] - b[1]) / SPEED)
    if len(a) > 2 and len(b) > 2:
        v = max(v, abs(a[2] - b[2]) / ARM_SPEED)
    return math.ceil(v - 1e-6)


def _nsteps(path):
    return sum(_seg_steps(path[i], path[i + 1]) for i in range(len(path) - 1))


class GeneratedApproach:
    FLOOR_BOX = dict(name="__floor__", x=-2.0, y=-2.0, w=6.0, h=2.0 + FLOOR)
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)

    # ------------------------------------------------------------------ parse
    def _parse(self, state):
        robot = None
        rects = {}
        block = surface = None
        for name in state.get_object_names():
            o = state.get_object_from_name(name) if isinstance(name, str) else name
            tn = o.type.name
            if tn == "crv_robot":
                robot = o
            elif tn == "target_block":
                block = o
            elif tn == "target_surface":
                surface = o
            elif tn in ("rectangle",):
                rects[o.name] = o
        g = lambda o, f: float(state.get(o, f))
        self.robot = dict(x=g(robot, "x"), y=g(robot, "y"), theta=g(robot, "theta"),
                          arm=g(robot, "arm_joint"), vac=g(robot, "vacuum"),
                          arm_max=g(robot, "arm_length"),
                          gw=g(robot, "gripper_width"), gh=g(robot, "gripper_height"),
                          r=g(robot, "base_radius"))

        def box(o):
            return dict(name=o.name, x=g(o, "x"), y=g(o, "y"), w=g(o, "width"), h=g(o, "height"))
        self.block = box(block)
        self.surface = box(surface)
        self.obst = {n: box(o) for n, o in rects.items()}

    def _all_movable(self):
        d = dict(self.obst)
        d[self.block["name"]] = self.block
        return d

    def _grip_off(self):
        return self.robot["arm_max"] + self.robot["gw"] / 2.0

    # --------------------------------------------------------- collision check
    def _collides(self, pts, obstacles, carried, margin_map, default_margin, check_bounds=True):
        """pts: (N,2) base positions. carried: (dx0, dx1, dy0, dy1) rel to base or None.
        obstacles: list of boxes. Returns True if any sample collides."""
        X = pts[:, 0]
        Y = pts[:, 1]
        r = self.robot["r"]
        half = self.robot["gh"] / 2.0
        amax = self.robot["arm_max"]
        if pts.shape[1] > 2:
            Ar = pts[:, 2]
        else:
            Ar = np.full(len(X), amax)
        go = Ar + self.robot["gw"] / 2.0
        if check_bounds and (np.any(X < X_MIN - 1e-6) or np.any(X > X_MAX + 1e-6)
                             or np.any(Y > Y_MAX + 1e-6)):
            return True
        # boxes attached to robot: gripper/arm, carried
        att = [(X - half, X + half, Y - go, Y)]
        if carried is not None:
            sh = amax - Ar
            att.append((X + carried[0], X + carried[1], Y + carried[2] + sh, Y + carried[3] + sh))
        for o in list(obstacles) + [self.FLOOR_BOX]:
            m = margin_map.get(o["name"], default_margin)
            ox0, ox1 = o["x"] - m, o["x"] + o["w"] + m
            oy0, oy1 = o["y"] - m, o["y"] + o["h"] + m
            # circle vs rect
            cx = np.clip(X, ox0, ox1)
            cy = np.clip(Y, oy0, oy1)
            if np.any((X - cx) ** 2 + (Y - cy) ** 2 < r * r):
                return True
            for (x0, x1, y0, y1) in att:
                if np.any((x0 < ox1) & (x1 > ox0) & (y0 < oy1) & (y1 > oy0)):
                    return True
        return False

    def _path_free(self, path, obstacles, carried, margin_map, m):
        for i in range(len(path) - 1):
            a, b = np.array(path[i], dtype=float), np.array(path[i + 1], dtype=float)
            L = max(np.linalg.norm(b[:2] - a[:2]), abs(b[2] - a[2]) if len(a) > 2 else 0.0, 1e-9)
            n = int(L / 0.003) + 2
            t = np.linspace(0, 1, n)[:, None]
            pts = a[None] * (1 - t) + b[None] * t
            if self._collides(pts, obstacles, carried, margin_map, m):
                return False
        return True

    def _plan_path(self, A, B, obstacles, carried, special_margin=None):
        """Find short collision-free polyline from A to B (points are (x, y, arm))."""
        amax = self.robot["arm_max"]
        amin = self.robot["r"]
        A = (A[0], A[1], A[2] if len(A) > 2 else amax)
        B = (B[0], B[1], B[2] if len(B) > 2 else amax)
        m = 0.004
        margin_map = {}
        # obstacles in contact at endpoints get zero margin
        for o in list(obstacles) + [self.FLOOR_BOX]:
            for P in (A, B):
                if self._collides(np.array([P]), [o], carried, {}, m):
                    margin_map[o["name"]] = 0.0
        if special_margin:
            margin_map.update(special_margin)
        cands = [[A, B]]
        sgn = 1.0 if B[0] >= A[0] else -1.0
        for ap in (amax, amin):
            h0 = max(A[1], B[1]) - (amax - ap)
            hs = np.arange(h0, Y_MAX + 1e-9, 0.01)
            hs = np.append(hs, Y_MAX)
            h_min = None
            for h in hs:
                if self._path_free([(A[0], h, ap), (B[0], h, ap)], obstacles, carried, margin_map, m):
                    h_min = h
                    break
            if h_min is None:
                continue
            hs = hs[hs >= h_min - 1e-9]
            for h in hs:
                r1 = max(abs(h - A[1]), SPEED if abs(ap - A[2]) > 1e-9 else 0.0)
                r2 = max(abs(h - B[1]), SPEED if abs(ap - B[2]) > 1e-9 else 0.0)
                for k1 in (1.0, 0.5, 0.0):
                    for k2 in (1.0, 0.5, 0.0):
                        x1 = A[0] + sgn * k1 * r1
                        x2 = B[0] - sgn * k2 * r2
                        if sgn * (x2 - x1) < 0:
                            continue
                        p = [A, (x1, h, ap), (x2, h, ap), B]
                        cands.append(p)
                for xv in np.linspace(A[0], B[0], 9)[1:-1]:
                    cands.append([A, (xv, h, ap), B])
        scored = sorted(((_nsteps(p), len(p), i) for i, p in enumerate(cands)))
        checked = 0
        for s, _, i in scored:
            p = cands[i]
            checked += 1
            if checked > 3000:
                break
            if self._path_free(p, obstacles, carried, margin_map, m):
                return p
        # fallback: straight up, over, down
        return [A, (A[0], Y_MAX, amax), (B[0], Y_MAX, amax), B]

    # --------------------------------------------------------------- planning
    def reset(self, state, info):
        self.plan = []
        self.stall = 0
        self.last_pos = None
        self.target_name = None
        self.block_dest = None
        self.staged = False
        self.attempts = {}
        self.carry = None
        self.stuck_carry = 0

    def _union(self, objs):
        x0 = min(o["x"] for o in objs)
        y0 = min(o["y"] for o in objs)
        x1 = max(o["x"] + o["w"] for o in objs)
        y1 = max(o["y"] + o["h"] for o in objs)
        return dict(name=objs[0]["name"], x=x0, y=y0, w=x1 - x0, h=y1 - y0)

    def _cograsp(self, tgt, gx, others):
        """Objects that will be sucked up together with tgt when grasping at gx."""
        half = self.robot["gh"] / 2.0
        btop = tgt["y"] + tgt["h"]
        return [o for o in others if o["name"] != self.block["name"]
                and btop - 0.016 <= o["y"] + o["h"] <= btop + 1e-4
                and o["x"] < gx + half and o["x"] + o["w"] > gx - half]

    def _grasp_x(self, b, others, rng=None, pref=None, allow_group=False):
        half = self.robot["gh"] / 2.0
        lo, hi = b["x"] + 0.012 - half, b["x"] + b["w"] - 0.012 + half
        if rng is not None:
            lo, hi = max(lo, rng[0]), min(hi, rng[1])
        lo, hi = max(lo, X_MIN), min(hi, X_MAX)
        if hi < lo:
            lo = hi = min(max(b["x"] + b["w"] / 2, X_MIN), X_MAX)
        by = b["y"] + b["h"] + self._grip_off() + GAP
        c = b["x"] + b["w"] / 2 if pref is None else pref
        ilo, ihi = b["x"] + half + 0.003, b["x"] + b["w"] - half - 0.003

        def key(x):
            over = 0.0 if ilo <= x <= ihi else min(abs(x - ilo), abs(x - ihi))
            return (round(over, 4) > 0, abs(x - c))
        cands = sorted(np.linspace(lo, hi, 41), key=key)
        R = self.robot["r"] + 0.005
        btop = b["y"] + b["h"]
        self.grasp_conflict = None
        self.grasp_group = []
        bname = self.block["name"]
        for x in cands:
            if x < X_MIN or x > X_MAX:
                continue
            ok = True
            group = []
            for o in others:
                otop = o["y"] + o["h"]
                if otop > btop - 0.02 and o["x"] < x + half + 0.004 \
                        and o["x"] + o["w"] > x - half - 0.004:
                    if allow_group and o["name"] != bname and btop - 0.009 <= otop <= btop + 1e-4 \
                            and o["x"] < x + half - 0.004 and o["x"] + o["w"] > x - half + 0.004:
                        group.append(o)
                        continue
                    ok = False
                    if self.grasp_conflict is None:
                        self.grasp_conflict = o
                    break
            if not ok:
                continue
            for o in others:
                x0, x1 = o["x"], o["x"] + o["w"]
                d = 0.0 if x0 <= x <= x1 else min(abs(x - x0), abs(x - x1))
                if d >= R:
                    continue
                if o["y"] + o["h"] > by - math.sqrt(R * R - d * d):
                    ok = False
                    if self.grasp_conflict is None:
                        self.grasp_conflict = o
                    break
            if ok:
                self.grasp_group = group
                return x
        return None

    def _free_spot(self, b, others, forbid, gx_rel, gpos, next_pos):
        """Find (left x, bottom y) for box b on floor avoiding others and forbid."""
        w = b["w"]
        m = 0.03
        ivs = [(o["x"] - m, o["x"] + o["w"] + m) for o in others] + forbid
        best = None
        off = gpos[1] - b["y"]
        for bx in np.arange(X_MIN, X_MAX, 0.005):
            x0 = bx - gx_rel
            if x0 < 0.0 or x0 + w > WORLD_X:
                continue
            if any(x0 < i1 and x0 + w > i0 for i0, i1 in ivs):
                continue
            if self._column_hit(dict(b, x=x0, y=FLOOR + PLACE_GAP), extra=0.02):
                continue
            P = (bx, FLOOR + PLACE_GAP + off)
            cost = _cheb(gpos, P) + _cheb(P, next_pos)
            if best is None or cost < best[0]:
                best = (cost, x0)
        if best is not None:
            return best[1], FLOOR
        # fallback: stack on top of some other object away from surface
        best = None
        for o in others:
            top = o["y"] + o["h"]
            x0 = o["x"] + o["w"] / 2 - w / 2
            if self._column_hit(dict(b, x=x0, y=top + PLACE_GAP), extra=0.02):
                continue
            if best is None or top < best[0]:
                best = (top, x0)
        if best is not None:
            return best[1], best[0]
        return None

    def _column_info(self, dest_x, grel=None):
        b = self.block
        gx_rel = self.block_grel if grel is None else grel
        gy_rel = b["h"] + self._grip_off() + GAP
        bx = dest_x + gx_rel
        py = FLOOR + BLOCK_Z + gy_rel
        pts = np.stack([np.full(40, bx), np.linspace(py, Y_MAX, 40)], axis=1)
        carried = (-gx_rel, b["w"] - gx_rel, -gy_rel, b["h"] - gy_rel)
        return pts, carried

    def _column_hit(self, o, extra=0.0, dest_x=None, grel=None):
        if dest_x is None:
            dest_x = self.block_dest
        pts, carried = self._column_info(dest_x, grel)
        return self._collides(pts, [o], carried, {"__floor__": -1.0}, 0.002 + extra,
                              check_bounds=False)

    def _choose_block_dest(self):
        s = self.surface
        b = self.block
        lo = s["x"] + 0.003
        hi = s["x"] + s["w"] - b["w"] - 0.003
        center = s["x"] + (s["w"] - b["w"]) / 2
        if hi <= lo:
            lo = hi = center
        best = None
        for dx in np.linspace(lo, hi, 25):
            half = self.robot["gh"] / 2.0
            glo = max(0.012 - half, X_MIN - dx, X_MIN - b["x"])
            ghi = min(b["w"] - 0.012 + half, X_MAX - dx, X_MAX - b["x"])
            if ghi >= glo:
                # prefer no gripper overhang
                ilo, ihi = max(glo, half + 0.003), min(ghi, b["w"] - half - 0.003)
            if ghi < glo:
                continue
            if ilo <= ihi:
                pref = min(max(b["w"] / 2, ilo), ihi)
            else:
                pref = min(max(b["w"] / 2, glo), ghi)
            for grel in (pref, glo, ghi, 0.5 * (pref + glo), 0.5 * (pref + ghi)):
                nb = 0
                for o in self.obst.values():
                    if self._column_hit(o, dest_x=dx, grel=grel):
                        reach = o["x"] + 0.012 - half <= X_MAX and o["x"] + o["w"] - 0.012 + half >= X_MIN
                        nb += 1 if reach else 100
                cost = nb * 100 + abs(dx - center) * 0.1 + abs(grel - pref) * 0.5
                if best is None or cost < best[0]:
                    best = (cost, dx, grel)
        if best is None:
            return center, b["w"] / 2
        return best[1], best[2]

    def _make_plan(self):
        """Returns list of segments: each (waypoint list, vac, kind)."""
        s = self.surface
        s0, s1 = s["x"], s["x"] + s["w"]
        movable = self._all_movable()
        if self.block_dest is None:
            self.block_dest, self.block_grel = self._choose_block_dest()
        blockers = [o for o in self.obst.values() if self._column_hit(o)]
        rx, ry = self.robot["x"], self.robot["y"]
        if not blockers:
            bb = self.block
            others = [b for n, b in movable.items() if n != bb["name"]]
            rng = (bb["x"] + X_MIN - self.block_dest, bb["x"] + X_MAX - self.block_dest)
            gxb = self._grasp_x(bb, others, rng=rng, pref=bb["x"] + self.block_grel)
            if gxb is not None and abs(gxb - bb["x"] - self.block_grel) > 1e-4:
                grel = gxb - bb["x"]
                bl2 = [o for o in self.obst.values() if self._column_hit(o, grel=grel)]
                if bl2:
                    self.block_grel = grel
                    blockers = bl2
        stage = False
        if blockers:
            tgt = min(blockers, key=lambda b: abs(b["x"] + b["w"] / 2 - rx))
            if not self.staged:
                oc = tgt["x"] + tgt["w"] / 2
                bc = self.block["x"] + self.block_grel
                gain = abs(rx - oc) + abs(oc - bc) - abs(rx - bc) - abs(bc - self.block_dest)
                if gain > STAGE_GAIN:
                    stage = True
                    tgt = self.block
        else:
            tgt = self.block
        gx = None
        tried = []
        for _ in range(10):
            tried.append(tgt)
            others = [b for n, b in movable.items() if n != tgt["name"]]
            if tgt["name"] == self.block["name"]:
                bb = self.block
                rng = (bb["x"] + X_MIN - self.block_dest, bb["x"] + X_MAX - self.block_dest)
                gx = self._grasp_x(tgt, others, rng=rng, pref=bb["x"] + self.block_grel)
            else:
                gx = self._grasp_x(tgt, others)
            if gx is not None or self.grasp_conflict is None:
                break
            tgt = self.grasp_conflict
        if gx is None:
            for t in tried:
                if t["name"] == self.block["name"]:
                    continue
                oth = [b for n, b in movable.items() if n != t["name"]]
                g = self._grasp_x(t, oth, allow_group=True)
                if g is not None:
                    tgt, gx = t, g
                    break
        n_att = self.attempts.get(tgt["name"], 0)
        self.attempts[tgt["name"]] = n_att + 1
        if n_att >= 3:
            # repeated failure: move the tallest nearby object instead
            near = [o for o in movable.values() if o["name"] != tgt["name"]
                    and o["name"] != self.block["name"]
                    and abs((o["x"] + o["w"] / 2) - (tgt["x"] + tgt["w"] / 2)) < 0.3
                    and self.attempts.get(o["name"], 0) < 3]
            if near:
                tgt = max(near, key=lambda o: o["y"] + o["h"])
                others = [b for n, b in movable.items() if n != tgt["name"]]
                gx = self._grasp_x(tgt, others)
        if gx is None:
            gx = min(max(tgt["x"] + tgt["w"] / 2, X_MIN), X_MAX)
        others = [b for n, b in movable.items() if n != tgt["name"]]
        self.target_name = tgt["name"]
        is_block = tgt["name"] == self.block["name"]
        gy = tgt["y"] + tgt["h"] + self._grip_off() + GAP
        gpos = (gx, gy)
        group = [] if is_block else self._cograsp(tgt, gx, others)
        gnames = [tgt["name"]] + [o["name"] for o in group]
        others = [b for b in others if b["name"] not in gnames]
        tgt0 = tgt
        tgt = self._union([tgt0] + group)
        off = gy - tgt["y"]
        gx_rel = gx - tgt["x"]
        if is_block and stage and tgt["name"] == self.block["name"]:
            self.staged = True
            nb = min(blockers, key=lambda b: abs(b["x"] + b["w"] / 2 - self.block_dest))
            next_pos = (nb["x"] + nb["w"] / 2, nb["y"] + nb["h"] + self._grip_off() + GAP)
            spot = self._free_spot(tgt, others, [], gx_rel, gpos, next_pos)
            if spot is None:
                spot = (tgt["x"], min(0.6, tgt["y"] + 0.3))
            is_block = False
            dest_x, dest_y = spot
        elif is_block:
            dest_x = self.block_dest
            dest_y = FLOOR + BLOCK_Z - PLACE_GAP
        else:
            b = self.block
            bgx = b["x"] + b["w"] / 2
            next_pos = (bgx, b["y"] + b["h"] + self._grip_off() + GAP)
            rem = [o for o in blockers if o["name"] != tgt["name"]]
            if rem:
                nb = min(rem, key=lambda o: abs(o["x"] + o["w"] / 2 - gx))
                next_pos = (nb["x"] + nb["w"] / 2, nb["y"] + nb["h"] + self._grip_off() + GAP)
            spot = self._free_spot(tgt, others, [], gx_rel, gpos, next_pos)
            if spot is None:
                spot = (tgt["x"], min(0.6, tgt["y"] + 0.3))
            dest_x, dest_y = spot
        px = min(max(dest_x + gx_rel, X_MIN), X_MAX)
        py = dest_y + off + PLACE_GAP
        # approach path (nothing carried)
        p1 = self._plan_path((rx, ry, self.robot["arm"]), gpos, others + [tgt0] + group, None,
                             special_margin={n: 0.0 for n in gnames})
        carried = (tgt["x"] - gx, tgt["x"] + tgt["w"] - gx, tgt["y"] - gy, tgt["y"] + tgt["h"] - gy)
        p2 = self._plan_path(gpos, (px, py), others, carried)
        segs = []
        for w in p1[1:]:
            segs.append(dict(x=w[0], y=w[1], a=w[2], vac=0, kind="move"))
        segs.append(dict(x=gx, y=gy, vac=1, kind="grasp"))
        for w in p2[1:]:
            segs.append(dict(x=w[0], y=w[1], a=w[2], vac=1, kind="carry"))
        # release on the final step of the carry
        segs[-1]["release_last"] = not is_block
        self.carry = dict(name=tgt["name"], names=gnames, px=px, py=py, release=not is_block,
                          rel=(tgt["x"] - gx, tgt["y"] - gy))
        return segs

    def _replan_carry(self):
        """Replan carry path from current pose if still holding the object."""
        c = self.carry
        if c is None or self.robot["vac"] < 0.5:
            return None
        movable = self._all_movable()
        if c["name"] not in movable:
            return None
        o = movable[c["name"]]
        rx, ry = self.robot["x"], self.robot["y"]
        sh = self.robot["arm_max"] - self.robot["arm"]
        if abs(o["x"] - rx - c["rel"][0]) > 2e-3 or abs(o["y"] - ry - sh - c["rel"][1]) > 2e-3:
            return None
        names = c.get("names", [c["name"]])
        others = [b for n, b in movable.items() if n not in names]
        o = self._union([movable[n] for n in names if n in movable])
        carried = (o["x"] - rx, o["x"] + o["w"] - rx, o["y"] - ry - sh, o["y"] + o["h"] - ry - sh)
        p = self._plan_path((rx, ry, self.robot["arm"]), (c["px"], c["py"]), others, carried)
        segs = [dict(x=w[0], y=w[1], a=w[2], vac=1, kind="carry") for w in p[1:]]
        if not segs:
            return None
        segs[-1]["release_last"] = c["release"]
        return segs

    # ---------------------------------------------------------------- control
    def get_action(self, state):
        self._parse(state)
        r = self.robot
        if not self.plan:
            self.plan = self._make_plan()
            self.stall = 0
        for _ in range(20):
            wp = self.plan[0]
            if wp["kind"] == "grasp":
                reached = r["vac"] > 0.5
            else:
                reached = abs(r["x"] - wp["x"]) < 1e-4 and abs(r["y"] - wp["y"]) < 1e-4 \
                    and abs(r["arm"] - wp.get("a", r["arm_max"])) < 1e-4
                if wp.get("release_last"):
                    reached = reached and r["vac"] < 0.5
            if reached:
                self.plan.pop(0)
                self.stall = 0
                if not self.plan:
                    self.plan = self._make_plan()
                continue
            break
        wp = self.plan[0]
        pos = (round(r["x"], 5), round(r["y"], 5), round(r["arm"], 5), round(r["vac"]))
        if self.last_pos == pos:
            self.stall += 1
        else:
            self.stall = 0
        self.last_pos = pos
        dth = -math.pi / 2 - r["theta"]
        darm = r["arm_max"] - r["arm"]
        if wp["kind"] == "grasp":
            return self._act(0, 0, darm, 1, dth)
        if wp["kind"] == "move" and r["vac"] > 0.5:
            # make sure nothing is held before moving freely
            return self._act(0, 0, 0, 0, 0)
        if self.stall >= 2:
            # stuck: nudge up and replan
            self.stall = 0
            self.plan = []
            if wp["kind"] == "carry":
                self.stuck_carry += 1
                if self.stuck_carry <= 4:
                    segs = self._replan_carry()
                    if segs:
                        self.plan = [dict(x=r["x"], y=min(Y_MAX, r["y"] + 0.02), a=r["arm"], vac=1,
                                          kind="carry")] + segs
            if wp["kind"] == "carry" and wp is not None:
                return self._act(0, 0.02, 0, 1 if r["vac"] > 0.5 else 0, dth)
            return self._act(0, 0.02, 0, 0, dth)
        dx = wp["x"] - r["x"]
        dy = wp["y"] - r["y"]
        da = wp.get("a", r["arm_max"]) - r["arm"]
        mx = max(abs(dx) / SPEED, abs(dy) / SPEED, abs(da) / ARM_SPEED)
        sc = min(1.0, 1.0 / mx) if mx > 0 else 1.0
        darm = da * sc
        vac = wp["vac"]
        if wp.get("release_last") and sc >= 1.0:
            vac = 0
        return self._act(dx * sc, dy * sc, darm, vac, dth)

    def _act(self, dx, dy, darm, vac, dth=0.0):
        a = np.array([dx, dy, dth, darm, float(vac)], dtype=np.float32)
        return np.clip(a, self.low, self.high).astype(np.float32)
