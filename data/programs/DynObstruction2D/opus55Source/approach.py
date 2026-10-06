"""Push-based approach for DynObstruction2DEnv.

Strategy: lower an open, vertical gripper beside the target block (on the side
away from the target surface) so that a finger is just above the table, then
push the block horizontally along the table onto the surface.  Obstructions in
the way are bulldozed ahead of the target block; obstructions too close behind
the target are pushed along with it (we lower the pusher behind the whole
cluster).  The plan is recomputed whenever contact is lost or progress stalls.
"""
import numpy as np



def _get_pickplace():
    return _PickPlacePrim

TABLE_TOP = 0.1
WORLD_MAX_X = 1.0 + np.sqrt(5)
MAX_DX = 0.05
MAX_DTH = np.pi / 16
MAX_DARM = 0.1
MAX_DGRIP = 0.02
FINGER_LEN = 0.225
Y_HIGH = 1.25
TIP_CLEAR = 0.015  # fingertip height above table while pushing
SIDE_MARGIN = 0.025  # gap between pusher and object when lowering


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _objs_of(state, type_name):
    out = []
    for name in sorted(state.get_object_names(), key=str):
        obj = state.get_object_from_name(name)
        if obj.type.name == type_name:
            out.append(obj)
    return out


def _rect_info(state, obj):
    d = {k: float(state.get(obj, k)) for k in ["x", "y", "theta", "width", "height"]}
    c, s = np.cos(d["theta"]), np.sin(d["theta"])
    hw, hh = d["width"] / 2, d["height"] / 2
    xs, ys = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            xs.append(d["x"] + sx * hw * c - sy * hh * s)
            ys.append(d["y"] + sx * hw * s + sy * hh * c)
    d["l"], d["r"] = min(xs), max(xs)
    d["bot"], d["top"] = min(ys), max(ys)
    d["name"] = obj.name
    return d


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives
        self.reset(None, None)

    def reset(self, state, info):
        self.phase = "up"
        self.plan = None
        self.stall = 0
        self.idle = 0
        self.unjam_steps = 0
        self.prev_bx = None
        self.steps = 0
        self.unjam_count = {}
        self.unjam = None
        self.pick_count = {}
        self.clear_count = {}
        self.picker = None

    # ------------------------------------------------------------------
    def _parse(self, state):
        robot = _objs_of(state, "kin_robot")[0]
        r = {k: float(state.get(robot, k)) for k in
             ["x", "y", "theta", "arm_joint", "finger_gap", "base_radius",
              "arm_length", "finger_height", "finger_width",
              "gripper_base_width", "gripper_base_height"]}
        b = _rect_info(state, _objs_of(state, "target_block")[0])
        surf = _objs_of(state, "target_surface")[0]
        s = {k: float(state.get(surf, k)) for k in ["x", "y", "width", "height"]}
        obs = [_rect_info(state, o) for o in _objs_of(state, "dyn_rectangle")]
        return r, b, s, obs

    def _action(self, r, tx=None, ty=None, tth=None, tarm=None, tgap=None):
        a = np.zeros(5, dtype=np.float64)
        if tx is not None:
            a[0] = np.clip(tx - r["x"], -MAX_DX, MAX_DX)
        if ty is not None:
            a[1] = np.clip(ty - r["y"], -MAX_DX, MAX_DX)
        if tth is not None:
            a[2] = np.clip(_wrap(tth - r["theta"]), -MAX_DTH, MAX_DTH)
        if tarm is not None:
            a[3] = np.clip(tarm - r["arm_joint"], -MAX_DARM, MAX_DARM)
        if tgap is not None:
            a[4] = np.clip(tgap - r["finger_gap"], -MAX_DGRIP, MAX_DGRIP)
        lo = self.action_space.low * 0.999
        hi = self.action_space.high * 0.999
        return np.clip(a, lo, hi).astype(np.float64)

    @staticmethod
    def _safe_x(r):
        m = r["base_radius"] + r["arm_joint"] + 0.25
        return float(np.clip(r["x"], m, WORLD_MAX_X - m))

    # ------------------------------------------------------------------
    def _half_ext(self, r):
        return r["gripper_base_height"] / 2 + r["finger_height"] / 2

    def _make_plan(self, r, b, s, obs, subj=None, d=None, goal=None, margin=None):
        if margin is None:
            p1 = self._make_plan(r, b, s, obs, subj, d, goal, margin=0.025)
            p2 = self._make_plan(r, b, s, obs, subj, d, goal, margin=0.012)
            if (not p1["wall"] or p2["wall"]) and len(p1["chain"]) <= len(p2["chain"]):
                return p1
            return p2
        """Choose push direction, pusher x, and target x for the subject
        (the target block by default)."""
        is_target = subj is None
        if is_target:
            slack = s["width"] - b["width"]
            d = 1.0 if b["x"] < s["x"] else -1.0
            goal = s["x"] - d * 0.15 * slack
            obs = list(obs)
        else:
            obs = [o for o in obs if o["name"] != subj["name"]] + [b]
            b = subj
        he = self._half_ext(r)
        rad = r["base_radius"]
        y_push = TABLE_TOP + TIP_CLEAR + FINGER_LEN + r["arm_length"]
        # walk outwards from the block through the cluster behind it
        cur = b
        chain = [b]
        others = [o for o in obs]
        while True:
            if d > 0:
                bx = cur["l"] - he - margin
                lo_f, hi_f = bx - he - 0.005, bx + he
            else:
                bx = cur["r"] + he + margin
                lo_f, hi_f = bx - he, bx + he + 0.005
            wall_ok = (bx >= rad + 0.005) and (bx <= WORLD_MAX_X - rad - 0.005) \
                and lo_f >= 0.0 and hi_f <= WORLD_MAX_X
            blocker = None
            for o in others:
                if o in chain:
                    continue
                # is the object behind cur (on the pusher side)?
                behind = (o["x"] < cur["x"]) if d > 0 else (o["x"] > cur["x"])
                if not behind:
                    continue
                # footprint of fingers / gripper (low part)
                if o["r"] > lo_f and o["l"] < hi_f:
                    blocker = o
                    break
                # base circle (centre at bx, y_push)
                dxc = max(o["l"] - bx, bx - o["r"], 0.0)
                dyc = max(y_push - o["top"], 0.0)
                if dxc * dxc + dyc * dyc < (rad + 0.01) ** 2:
                    blocker = o
                    break
            if blocker is not None:
                # pick the closest blocker behind
                cands = [o for o in others if o not in chain and
                         ((o["x"] < cur["x"]) if d > 0 else (o["x"] > cur["x"])) and
                         o["r"] > lo_f - rad and o["l"] < hi_f + rad]
                if d > 0:
                    blocker = max(cands, key=lambda o: o["r"])
                else:
                    blocker = min(cands, key=lambda o: o["l"])
                chain.append(blocker)
                cur = blocker
                continue
            if not wall_ok:
                bx = float(np.clip(bx, rad + 0.005, WORLD_MAX_X - rad - 0.005))
                return dict(d=d, bx=bx, goal=goal, y_push=y_push, chain=chain,
                            wall=True, subj=b["name"], is_target=is_target)
            return dict(d=d, bx=bx, goal=goal, y_push=y_push, chain=chain,
                        wall=False, subj=b["name"], is_target=is_target)

    # ------------------------------------------------------------------
    def get_action(self, state):
        self.steps += 1
        r, b, s, obs = self._parse(state)
        rad = r["base_radius"]
        gap_open = r["gripper_base_height"]
        arm_max = r["arm_length"]
        down = -np.pi / 2

        if self.phase == "up":
            ok_y = r["y"] >= Y_HIGH - 0.02
            if ok_y:
                self.phase = "rotate"
            else:
                tx = self._safe_x(r) if r["y"] > 0.9 else r["x"]
                return self._action(r, tx=tx, ty=Y_HIGH, tarm=rad, tgap=gap_open)
        if self.phase == "rotate":
            if (abs(_wrap(down - r["theta"])) < 1e-3 and r["arm_joint"] <= rad + 1e-3
                    and abs(r["x"] - self._safe_x(r)) < 1e-3):
                self._decide(state, r, b, s, obs)
                if self.phase != "travel":
                    return self.get_action(state)
            else:
                return self._action(r, tx=self._safe_x(r), ty=Y_HIGH, tth=down,
                                    tarm=rad, tgap=gap_open)
        if self.phase == "pick":
            a = self.picker.step(state)
            if a is not None:
                return a
            self.phase = "up"
            return self.get_action(state)
        if self.phase == "unjam":
            self.unjam_steps = getattr(self, "unjam_steps", 0) + 1
            a = self.unjam.step(state) if self.unjam_steps <= 160 else None
            if a is not None:
                return a
            self.unjam_steps = 0
            self.phase = "up"
            return self.get_action(state)
        p = self.plan
        if self.phase == "travel":
            if abs(r["x"] - p["bx"]) < 2e-3:
                self.phase = "extend"
            else:
                return self._action(r, tx=p["bx"], ty=Y_HIGH, tth=down,
                                    tarm=rad, tgap=gap_open)
        if self.phase == "extend":
            if r["arm_joint"] >= arm_max - 1e-3:
                self.phase = "descend"
            else:
                return self._action(r, tx=p["bx"], ty=Y_HIGH, tth=down,
                                    tarm=arm_max, tgap=gap_open)
        if self.phase == "descend":
            if abs(r["y"] - p["y_push"]) < 2e-3:
                self.phase = "push"
                self.gap0 = None
                self.stall = 0
                self.prev_bx = None
            else:
                return self._action(r, tx=p["bx"], ty=p["y_push"], tth=down,
                                    tarm=arm_max, tgap=gap_open)
        if self.phase == "push":
            d = p["d"]
            he = self._half_ext(r)
            allo = obs + [b]
            tb = b
            b = [o for o in allo if o["name"] == p["subj"]][0]
            remaining = (p["goal"] - b["x"]) * d
            face = r["x"] + d * he
            # total free gap between pusher face and block
            names = {o["name"] for o in p["chain"]}
            chain_objs = [o for o in allo if o["name"] in names and o["name"] != b["name"]]
            if d > 0:
                gaps = b["l"] - face - sum(o["r"] - o["l"] for o in chain_objs)
            else:
                gaps = face - b["r"] - sum(o["r"] - o["l"] for o in chain_objs)
            gaps = max(gaps, 0.0)
            # lost contact (pusher passed the block) or overshoot -> replan
            if self.gap0 is None:
                self.gap0 = gaps
            self.gap0 = min(self.gap0, gaps)
            if gaps > self.gap0 + 0.08 or (d > 0 and face > b["r"]) or (d < 0 and face < b["l"]):
                self.phase = "up"
                return self.get_action_retry(state)
            if p["is_target"] and remaining < -0.25 * (s["width"] - b["width"]):
                self.phase = "up"
                return self.get_action_retry(state)
            if not p["is_target"] and remaining < 0.005:
                self.phase = "up"
                return self.get_action_retry(state)
            if self.prev_bx is not None and abs(b["x"] - self.prev_bx) < 1e-3 and remaining > 0.005 and gaps < 0.01:
                self.stall += 1
            else:
                self.stall = 0
            self.prev_bx = b["x"]
            if self.stall > 6:
                self.phase = "up"
                return self.get_action_retry(state)
            step = float(np.clip(remaining + gaps, 0.0, MAX_DX))
            self.idle = self.idle + 1 if step < 1e-3 else 0
            if self.idle > 5:
                self.idle = 0
                self.phase = "up"
                return self.get_action_retry(state)
            return self._action(r, tx=r["x"] + d * step, ty=p["y_push"], tth=down,
                                tarm=arm_max, tgap=gap_open)
        return self._action(r)

    # ------------------------------------------------------------------
    NARROW = 0.25
    FCLR = 0.19 + 0.02  # half footprint of the open gripper plus margin

    def _room_ok(self, plan, b, s, obs):
        d = plan["d"]
        ahead = [o for o in obs if (o["x"] - b["x"]) * d > 0]
        wsum = sum(o["r"] - o["l"] for o in ahead)
        sl, sr = s["x"] - s["width"] / 2, s["x"] + s["width"] / 2
        if d > 0:
            room = WORLD_MAX_X - (sl + 0.25 * (s["width"] - b["width"]) + b["width"])
        else:
            room = (sr - 0.25 * (s["width"] - b["width"])) - b["width"]
        return wsum <= room - 0.01, ahead

    def _spot_free(self, x, w, objs, exclude):
        half = max(w / 2, self.FCLR - 0.02) + 0.02
        if x - half < 0.0 or x + half > WORLD_MAX_X or x < 0.25 or x > WORLD_MAX_X - 0.25:
            return False
        for o in objs:
            if o["name"] == exclude:
                continue
            if o["r"] > x - half - 0.01 and o["l"] < x + half + 0.01:
                return False
        return True

    def _find_spot(self, w, objs, exclude, avoid_lo, avoid_hi):
        best, best_score = None, -1e9
        for x in np.arange(0.26, WORLD_MAX_X - 0.26, 0.01):
            if not self._spot_free(x, w, objs, exclude):
                continue
            if x + w / 2 > avoid_lo and x - w / 2 < avoid_hi:
                continue
            dist = min(abs(x - avoid_lo), abs(x - avoid_hi))
            score = dist
            if score > best_score:
                best, best_score = x, score
        return best

    def _pick_ok(self, state, name):
        try:
            return _get_pickplace().feasible(state, name)
        except Exception:
            return False

    def _start_pick(self, name, tx):
        k = self.pick_count.get(name, 0)
        if k >= 2:
            return False
        self.pick_count[name] = k + 1
        self.picker = _get_pickplace()(self.action_space, name, float(tx))
        self.phase = "pick"
        return True

    def _clear_spot(self, r, b, s, obs):
        half = max(self.FCLR, b["width"] / 2 + 0.08)
        lo_s, hi_s = s["x"] - half, s["x"] + half
        over = [o for o in obs if o["r"] > lo_s and o["l"] < hi_s]
        over.sort(key=lambda o: -(min(o["r"], hi_s) - max(o["l"], lo_s)))
        allobjs = obs + [b]
        for o in over:
            k = self.clear_count.get(o["name"], 0)
            if k >= 3:
                continue
            dirs = [1.0, -1.0] if o["x"] > s["x"] else [-1.0, 1.0]
            for dd in dirs:
                if dd > 0:
                    goal = hi_s + 0.02 + (o["x"] - o["l"])
                    room = WORLD_MAX_X - (goal + (o["r"] - o["x"]))
                else:
                    goal = lo_s - 0.02 - (o["r"] - o["x"])
                    room = goal - (o["x"] - o["l"])
                ahead = [q for q in allobjs if q["name"] != o["name"]
                         and (q["x"] - o["x"]) * dd > 0]
                need = sum(q["r"] - q["l"] for q in ahead)
                if need > room - 0.01:
                    continue
                plan = self._make_plan(r, b, s, obs, subj=o, d=dd, goal=goal)
                if plan["wall"]:
                    continue
                self.clear_count[o["name"]] = k + 1
                self.plan = plan
                self.phase = "travel"
                return True
        return False

    def _decide(self, state, r, b, s, obs):
        plan = self._make_plan(r, b, s, obs)
        self.plan = plan
        self.phase = "travel"
        room_ok, ahead = self._room_ok(plan, b, s, obs)
        if not plan["wall"] and room_ok:
            return
        allobjs = obs + [b]
        t_narrow = (b["r"] - b["l"]) <= self.NARROW and abs(np.sin(2 * b["theta"])) < 0.1
        # 1) narrow target: place it directly on the surface if the spot is free
        if t_narrow and self._pick_ok(state, b["name"]):
            if self._spot_free(s["x"], b["width"], allobjs, b["name"]):
                if self._start_pick(b["name"], s["x"]):
                    return
            # relocate narrow obstructions occupying the surface spot
            for o in sorted(obs, key=lambda o: abs(o["x"] - s["x"])):
                if not (o["r"] > s["x"] - self.FCLR and o["l"] < s["x"] + self.FCLR):
                    continue
                if (o["r"] - o["l"]) <= self.NARROW and abs(np.sin(2 * o["theta"])) < 0.1 \
                        and self._pick_ok(state, o["name"]):
                    spot = self._find_spot(o["width"], allobjs, o["name"],
                                           s["x"] - 0.3, s["x"] + 0.3)
                    if spot is not None and self._start_pick(o["name"], spot):
                        return
            # push obstructions off the placement spot
            if self._clear_spot(r, b, s, obs):
                return
        # 2) not enough room ahead: relocate narrow obstructions ahead
        if not room_ok:
            d = plan["d"]
            lo = min(b["l"] - 0.45, s["x"] - s["width"] / 2) if d > 0 else b["l"] - 0.05
            hi = b["r"] + 0.05 if d > 0 else max(b["r"] + 0.45, s["x"] + s["width"] / 2)
            if d > 0:
                lo, hi = b["l"] - 0.45, WORLD_MAX_X
            else:
                lo, hi = 0.0, b["r"] + 0.45
            for o in sorted(ahead, key=lambda o: abs(o["x"] - s["x"])):
                if (o["r"] - o["l"]) <= self.NARROW and abs(np.sin(2 * o["theta"])) < 0.1 \
                        and self._pick_ok(state, o["name"]):
                    spot = self._find_spot(o["width"], allobjs, o["name"], lo, hi)
                    if spot is not None and self._start_pick(o["name"], spot):
                        return
            # narrow obstruction stuck at a wall: unjam it first (then pick later)
            for o in ahead:
                if (o["r"] - o["l"]) <= self.NARROW and min(o["l"], WORLD_MAX_X - o["r"]) < 0.4:
                    k = self.unjam_count.get(o["name"], 0)
                    if k < 2:
                        self.unjam_count[o["name"]] = k + 1
                        self.unjam = WallUnjam(self.action_space, o["name"])
                        self.phase = "unjam"
                        return
        # 3) wall-blocked: unjam the object against the wall
        if plan["wall"]:
            name = plan["chain"][-1]["name"]
            k = self.unjam_count.get(name, 0)
            if k < 3:
                self.unjam_count[name] = k + 1
                self.unjam = WallUnjam(self.action_space, name)
                self.phase = "unjam"
                return
        # otherwise: best-effort push

    def get_action_retry(self, state):
        r, b, s, obs = self._parse(state)
        return self._action(r, ty=Y_HIGH, tarm=r["base_radius"],
                            tgap=r["gripper_base_height"])


# ======================================================================
# Wall-unjam primitive
# ======================================================================


WU_WALL_L, WU_WALL_R = 0.0, 1.0 + np.sqrt(5)
WU_Y_HIGH, WU_L_MAX, WU_ARM_MIN = 1.45, 0.48, 0.24
WU_TIP_FINAL = 0.12        # lowest fingertip height at the bottom of the descent
WU_SINK_TOL = 0.06         # abort if the block sinks this far into the table
WU_V_DESC, WU_V_PUSH = 0.0499, 0.02
WU_PUSH_STEPS, WU_G_TARGET, WU_G_NEED = 30, 0.56, 0.46


def _wu_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wu_rect_xext(o):
    c, s = np.cos(o["theta"]), np.sin(o["theta"])
    hw, hh = o["width"] / 2, o["height"] / 2
    xs = [o["x"] + sx * hw * c - sy * hh * s for sx in (-1, 1) for sy in (-1, 1)]
    return min(xs), max(xs)


def _wu_bottom_y(o):
    c, s = np.cos(o["theta"]), np.sin(o["theta"])
    return o["y"] - abs(o["width"] / 2 * s) - abs(o["height"] / 2 * c)


def _wu_rest_y(o):
    return 0.1 + (o["height"] / 2 if abs(np.cos(o["theta"])) > 0.7 else o["width"] / 2)


def _wu_gripper_parts(x, y, th, L, gap):
    """Corners of gripper base and both fingers (geometry measured from pymunk)."""
    u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    G = np.array([x, y]) + L * u
    rect = lambda c, hu, hv: np.array([c + a * hu * u + b * hv * v for a in (-1, 1) for b in (-1, 1)])
    return [rect(G, 0.03, 0.16), rect(G + 0.125 * u + gap / 2 * v, 0.1, 0.03),
            rect(G + 0.125 * u - gap / 2 * v, 0.1, 0.03)]


class WallUnjam:
    def __init__(self, action_space, block_name, configs=None):
        self.lo, self.hi = action_space.low * 0.999, action_space.high * 0.999
        self.name = block_name
        self.configs = configs
        self.phase, self.k, self.ci = "init", 0, 0
        self.done, self.success = False, False

    # ---- helpers -----------------------------------------------------
    def _parse(self, state):
        r = b = None
        for nm in state.get_object_names():
            o = state.get_object_from_name(nm)
            if o.type.name == "kin_robot":
                r = {f: float(state.get(o, f)) for f in ("x", "y", "theta", "arm_joint", "finger_gap")}
            elif nm == self.name:
                b = {f: float(state.get(o, f)) for f in ("x", "y", "theta", "width", "height")}
        return r, b

    def _act(self, r, tx=None, ty=None, tth=None, tarm=None, tgap=None):
        a = np.zeros(5)
        if tx is not None: a[0] = tx - r["x"]
        if ty is not None: a[1] = ty - r["y"]
        if tth is not None: a[2] = _wu_wrap(tth - r["theta"])
        if tarm is not None: a[3] = tarm - r["arm_joint"]
        if tgap is not None: a[4] = tgap - r["finger_gap"]
        return np.clip(a, self.lo, self.hi).astype(np.float64)

    def wall_gap(self, b):
        l, rr = _wu_rect_xext(b)
        return min(l - WU_WALL_L, WU_WALL_R - rr)

    def _ok(self, b):
        t = b["theta"] % (np.pi / 2)
        flat = min(t, np.pi / 2 - t) < 0.05 and abs(b["y"] - _wu_rest_y(b)) < 0.012
        return flat and self.wall_gap(b) >= WU_G_NEED

    def _setup(self, b):
        l, rr = _wu_rect_xext(b)
        self.d = -1.0 if (WU_WALL_R - rr) < (l - WU_WALL_L) else 1.0  # direction away from wall
        if self.configs is None:   # (finger_gap, tilt toward wall)
            self.configs = [(0.12, 0.4), (0.32, 0.0), (0.12, 0.5), (0.12, 0.3), (0.12, 0.6), (0.32, 0.2)]
        self._load_config()

    def _load_config(self):
        self.gap, a = self.configs[self.ci]
        self.th = -np.pi / 2 - self.d * a
        xs = np.concatenate([p[:, 0] for p in _wu_gripper_parts(0.0, 1.0, self.th, WU_L_MAX, self.gap)])
        if self.d < 0:
            self.bx = min(WU_WALL_R - 0.012 - xs.max(), WU_WALL_R - 0.245)
        else:
            self.bx = max(WU_WALL_L + 0.012 - xs.min(), WU_WALL_L + 0.245)
        self.xs = float(np.clip(self.bx + self.d * 0.3, 0.6, WU_WALL_R - 0.6))
        low = min(p[:, 1].min() for p in _wu_gripper_parts(0.0, 0.0, self.th, WU_L_MAX, self.gap)[1:])
        self.y_final = WU_TIP_FINAL - low

    def _lim(self, n=60):
        """Per-phase step limit (the robot may be blocked by contacts)."""
        if self.phase != getattr(self, "_lp", None):
            self._lp, self.pk = self.phase, 0
        self.pk += 1
        return self.pk <= n

    # ---- main --------------------------------------------------------
    def step(self, state):
        """Return next action, or None when finished (see self.success)."""
        r, b = self._parse(state)
        if self.phase == "init":
            self._setup(b); self.phase = "lift"
        if self.phase == "lift":
            if self._lim() and (r["y"] < WU_Y_HIGH - 0.01 or abs(r["arm_joint"] - WU_ARM_MIN) > 2e-3):
                return self._act(r, ty=WU_Y_HIGH, tarm=WU_ARM_MIN)
            self.phase = "rotate"
        if self.phase == "rotate":
            if self._lim() and (abs(r["x"] - self.xs) > 2e-3 or abs(_wu_wrap(self.th - r["theta"])) > 1e-3
                                or abs(r["finger_gap"] - self.gap) > 2e-3):
                # first move (arm retracted) to a safe x away from the wall, then rotate there
                far = abs(r["x"] - self.xs) > 0.05
                return self._act(r, tx=self.xs, ty=WU_Y_HIGH, tth=None if far else self.th,
                                 tarm=WU_ARM_MIN, tgap=self.gap)
            self.phase = "travel"
        if self.phase == "travel":
            if self._lim() and abs(r["x"] - self.bx) > 2e-3:
                return self._act(r, tx=self.bx, ty=WU_Y_HIGH, tth=self.th, tarm=WU_ARM_MIN, tgap=self.gap)
            self.phase = "extend"
        if self.phase == "extend":
            if self._lim() and abs(r["arm_joint"] - WU_L_MAX) > 2e-3:
                return self._act(r, tx=self.bx, ty=WU_Y_HIGH, tth=self.th, tarm=WU_L_MAX, tgap=self.gap)
            self.phase = "descend"
        if self.phase == "descend":
            if b["y"] < _wu_rest_y(b) - WU_SINK_TOL:
                self.phase = "retreat"      # block being crushed into table: abort
            elif self._lim() and abs(r["y"] - self.y_final) > 1e-3:
                return self._act(r, tx=self.bx, ty=max(self.y_final, r["y"] - WU_V_DESC),
                                 tth=self.th, tarm=WU_L_MAX, tgap=self.gap)
            else:
                self.phase, self.k = "push", 0
        if self.phase == "push":
            if self.k < WU_PUSH_STEPS and self.wall_gap(b) < WU_G_TARGET and _wu_bottom_y(b) > 0.1 - WU_SINK_TOL:
                self.k += 1
                return self._act(r, tx=r["x"] + self.d * WU_V_PUSH, ty=self.y_final,
                                 tth=self.th, tarm=WU_L_MAX, tgap=self.gap)
            self.phase, self.k = "retreat", 0
        if self.phase == "retreat":
            if r["y"] < WU_Y_HIGH - 0.01 and self.k < 80:
                self.k += 1
                return self._act(r, ty=WU_Y_HIGH)
            self.phase, self.k = "settle", 0
        if self.phase == "settle":
            if self.k < 10:
                self.k += 1
                return self._act(r)
            if self._ok(b):
                self.done, self.success = True, True
                return None
            self.ci += 1
            if self.ci >= len(self.configs) or abs(b["y"]) > 3:
                self.done = True
                return None
            self.d = None
            l, rr = _wu_rect_xext(b)
            self.d = -1.0 if (WU_WALL_R - rr) < (l - WU_WALL_L) else 1.0
            self._load_config()
            self.phase = "lift"
            return self.step(state)
        return self._act(r)


# ===================== Pick-and-place primitive =====================
"""Top-down pick-and-place primitive for DynObstruction2DEnv (step-wise, WallUnjam style).

PickPlace(action_space, obj_name, target_x).step(state) -> action or None (see .success).
Grasp: theta=-pi/2, arm fully extended (L=0.48), gripper plate centre G placed
PP_TOP_OFF=0.06 above the object top (object top then sits between the plate
bottom (G-0.03) and the grasp-detection limit (G-0.10)); fingers open to the
narrowest gap that straddles the object, then close at max speed until held.
"""



def _pp_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _pp_gripper_parts(x, y, th, L, gap):
    """Corners of gripper plate and both fingers (same geometry as approach.py)."""
    u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    G = np.array([x, y]) + L * u
    rect = lambda c, hu, hv: np.array([c + a * hu * u + b * hv * v for a in (-1, 1) for b in (-1, 1)])
    return [rect(G, 0.03, 0.16), rect(G + 0.125 * u + gap / 2 * v, 0.1, 0.03),
            rect(G + 0.125 * u - gap / 2 * v, 0.1, 0.03)]

PP_WALL_L, PP_WALL_R = 0.0, 1.0 + np.sqrt(5)
PP_XB_LO, PP_XB_HI = 0.245, 2.991          # base x limits
PP_SAFE_LO, PP_SAFE_HI = 0.75, PP_WALL_R - 0.75   # rotate only here
PP_Y_ROT = 1.45        # max base y while rotating (sweep stays below the ceiling)
PP_Y_MAX = 1.74
PP_Y_TOUCH = 1.80      # commanded into the top wall (base top hits y=2.0)
PP_L_MIN, PP_L_MAX = 0.24, 0.48
PP_BASE_R = 0.24
PP_SWEEP = 0.5         # radius swept by the retracted arm + open fingers when rotating
PP_GAP_MIN, PP_GAP_MAX = 0.12, 0.32
PP_FT = 0.06            # finger thickness (sideways)
PP_TIP = 0.2            # fingertip distance beyond plate centre along u (measured)
PP_PLATE_HW, PP_PLATE_HT = 0.16, 0.03
PP_TOP_OFF = 0.06       # plate centre height above object top at grasp
PP_IN_M = 0.004         # min inner clearance finger <-> object while descending
PP_OUT_M = 0.006        # min clearance finger/plate <-> neighbours / walls
PP_CLEAR_TOP = 0.58     # tallest possible object top
PP_CLEAR_M = 0.05       # held object bottom clearance during transport
PP_PLACE_H = 0.005      # object bottom height above table at release
PP_TABLE = 0.1


def _pp_rect(state, o):
    d = {f: float(state.get(o, f)) for f in ("x", "y", "theta", "width", "height")}
    c, s = np.cos(d["theta"]), np.sin(d["theta"])
    hw, hh = d["width"] / 2, d["height"] / 2
    xs = [d["x"] + a * hw * c - b * hh * s for a in (-1, 1) for b in (-1, 1)]
    ys = [d["y"] + a * hw * s + b * hh * c for a in (-1, 1) for b in (-1, 1)]
    d["l"], d["r"], d["bot"], d["top"] = min(xs), max(xs), min(ys), max(ys)
    d["held"] = float(state.get(o, "held"))
    # x where the block will rest after release: it pivots about its lowest corner
    # and settles flat (exact for small tilts; relevant if the grasp tilted it)
    i = int(np.argmin(ys))
    d["rest_x"] = d["x"] if abs(np.sin(d["theta"])) < 1e-3 else xs[i] - np.sign(xs[i] - d["x"]) * hw
    return d


def _pp_parse(state, name):
    r, b, others = None, None, []
    for nm in state.get_object_names():
        o = state.get_object_from_name(nm)
        tn = o.type.name
        if tn == "kin_robot":
            r = {f: float(state.get(o, f)) for f in ("x", "y", "theta", "arm_joint", "finger_gap")}
        elif nm == name:
            b = _pp_rect(state, o)
        elif tn in ("target_block", "dyn_rectangle"):
            others.append(_pp_rect(state, o))
    return r, b, others


class _PickPlacePrim:
    def __init__(self, action_space, obj_name, target_x):
        self.lo, self.hi = action_space.low * 0.999, action_space.high * 0.999
        self.name, self.tx = obj_name, float(target_x)
        self.phase, self.done, self.success = "init", False, False
        self.info = ""
        self.steps = 0
        self.phase_steps = {}

    # ---- geometry ----------------------------------------------------
    @staticmethod
    def plan_grasp(state, obj_name):
        """Return (gap, gripper_x, clearance) for the top-down approach, or None."""
        _, b, others = _pp_parse(state, obj_name)
        if b is None:
            return None
        w = b["r"] - b["l"]
        tip_y = b["top"] + PP_TOP_OFF - PP_TIP            # lowest fingertip height
        plate_y = b["top"] + PP_TOP_OFF - PP_PLATE_HT      # plate bottom height
        g_lo = w + 2 * PP_IN_M + PP_FT
        if g_lo > PP_GAP_MAX + 1e-9:
            return None
        best = None
        for g in np.unique(np.r_[np.linspace(g_lo, PP_GAP_MAX, 9), g_lo]):
            inner = g / 2 - PP_FT / 2
            c_lo, c_hi = b["r"] + PP_IN_M - inner, b["l"] - PP_IN_M + inner
            for c in np.linspace(c_lo, c_hi, 15) if c_hi > c_lo + 1e-9 else [b["x"]]:
                if not (PP_XB_LO <= c <= PP_XB_HI):
                    continue
                fo = g / 2 + PP_FT / 2
                parts = [(c - fo, c - inner, tip_y), (c + inner, c + fo, tip_y),
                         (c - PP_PLATE_HW, c + PP_PLATE_HW, plate_y)]
                clr = min(c - fo - PP_WALL_L, PP_WALL_R - c - fo)
                for o in others:
                    for (pl, pr, py) in parts:
                        if o["top"] > py - 0.01:
                            clr = min(clr, max(o["l"] - pr, pl - o["r"]))
                in_m = min(b["l"] - (c - inner), (c + inner) - b["r"])
                score = min(in_m, clr - PP_OUT_M)
                if clr >= PP_OUT_M and in_m >= PP_IN_M and (best is None or score > best[2] + 1e-6):
                    best = (float(g), float(c), float(score))
        return best

    @staticmethod
    def feasible(state, obj_name):
        return _PickPlacePrim.plan_grasp(state, obj_name) is not None

    # ---- helpers -----------------------------------------------------
    def _act(self, r, tx=None, ty=None, tth=None, tarm=None, tgap=None):
        a = np.zeros(5)
        if tx is not None: a[0] = tx - r["x"]
        if ty is not None: a[1] = ty - r["y"]
        if tth is not None: a[2] = _pp_wrap(tth - r["theta"])
        if tarm is not None: a[3] = tarm - r["arm_joint"]
        if tgap is not None: a[4] = tgap - r["finger_gap"]
        return np.clip(a, self.lo, self.hi).astype(np.float64)

    def _act_g(self, r, g_target, tx=None, tgap=None, l_lo=PP_L_MIN, l_hi=PP_L_MAX):
        """Move the gripper plate (theta=-pi/2) to height g_target using arm + base
        together (plate moves up to 0.15/step); arm kept within [l_lo, l_hi]."""
        L, g = r["arm_joint"], r["y"] - r["arm_joint"]
        dg = g_target - g
        dl = float(np.clip(-dg, -self.hi[3], self.hi[3]))
        dl = float(np.clip(L + dl, l_lo, l_hi) - L)
        if r["y"] + dg + dl > PP_Y_MAX:      # ceiling: use the arm instead
            dl = float(np.clip(PP_Y_MAX - r["y"] - dg, l_lo - L, dl))
        ty = min(r["y"] + dg + dl, PP_Y_MAX)
        return self._act(r, tx=tx, ty=ty, tth=-np.pi / 2, tarm=L + dl, tgap=tgap)

    @staticmethod
    def _lowest(r):
        parts = _pp_gripper_parts(r["x"], r["y"], r["theta"], r["arm_joint"], r["finger_gap"])
        return min(r["y"] - 0.24, min(p[:, 1].min() for p in parts))

    def _lim(self, n=60):
        if self.phase != getattr(self, "_lp", None):
            self._lp, self.pk = self.phase, 0
        self.pk += 1
        return self.pk <= n

    def _fail(self, why):
        self.done, self.success, self.info = True, False, why
        return None

    @staticmethod
    def _max_top(objs, lo, hi):
        tops = [o["top"] for o in objs if o["r"] > lo and o["l"] < hi]
        return max(tops) if tops else PP_TABLE

    def _clear_h(self, b, others, x_to):
        """Height the held object's bottom must keep while moving its rest x to x_to."""
        d = x_to - b["rest_x"]
        reach = max(self.gap / 2 + PP_FT / 2 + abs(self.off_x), (b["r"] - b["l"]) / 2) + 0.02
        lo, hi = min(b["x"], b["x"] + d) - reach, max(b["x"], b["x"] + d) + reach
        tops = [o["top"] for o in others if o["r"] > lo and o["l"] < hi]
        return max(tops) + PP_CLEAR_M if tops else PP_TABLE + PP_PLACE_H

    # ---- main --------------------------------------------------------
    def step(self, state):
        self.steps += 1
        r, b, others = _pp_parse(state, self.name)
        a = self._step(state, r, b, others)
        if a is not None:
            self.phase_steps[self.phase] = self.phase_steps.get(self.phase, 0) + 1
        return a

    def _step(self, state, r, b, others):
        TH = -np.pi / 2
        if self.phase == "init":
            plan = self.plan_grasp(state, self.name)
            if plan is None:
                return self._fail("infeasible approach")
            self.gap, self.cx, _ = plan
            self.g_grasp = b["top"] + PP_TOP_OFF
            # full extension: base travel from the top wall to the grasp height is shortest
            self.l_grasp = PP_L_MAX
            # head toward the object right away, but stay in the wall-safe band until rotated
            self.xs = float(np.clip(self.cx, PP_SAFE_LO, PP_SAFE_HI))
            self.phase = "approach"
        if self.phase == "approach":
            # lift (arm retracted) -> once clear of objects move to a safe x away from the
            # walls -> rotate there (y in [PP_Y_ROT_LO, PP_Y_ROT]) -> bump the top wall.
            if not self._lim(80):
                return self._fail("approach timeout")
            # stuck detection: a robot part penetrating a static body freezes the robot
            key = (round(r["x"], 4), round(r["y"], 4), round(r["theta"], 4), round(r["arm_joint"], 4))
            self._same = self._same + 1 if key == getattr(self, "_key", None) else 0
            self._key = key
            if self._same >= 4:
                return self._fail("robot frozen (static contact)")
            allo = others + [b]
            low = self._lowest(r)
            aligned = abs(_pp_wrap(TH - r["theta"])) < 1e-3
            x0, x1 = min(r["x"], self.xs) - PP_SWEEP, max(r["x"], self.xs) + PP_SWEEP
            clear = low > self._max_top(allo, x0, x1) + 0.03
            if not aligned:
                in_band = PP_SAFE_LO - 1e-3 <= r["x"] <= PP_SAFE_HI + 1e-3
                sw = PP_SWEEP + 0.05     # rotation may happen while translating
                rot_ok = (in_band and r["y"] <= PP_Y_ROT + 0.03 and r["y"] - PP_SWEEP
                          > self._max_top(allo, r["x"] - sw, r["x"] + sw) + 0.03)
                return self._act(r, tx=self.xs if clear else None, ty=PP_Y_ROT,
                                 tth=TH if rot_ok else None, tarm=PP_L_MIN,
                                 tgap=self.gap if clear else None)
            # Rotations make the gripper drift sideways off the arm axis (hidden state,
            # not in the observation).  A static collision reverts the robot and
            # re-centres the gripper, so bump the base into the top wall once.
            if getattr(self, "_ty", None) is not None and r["y"] < self._ty - 0.01:
                self.phase = "descend"      # motion was blocked -> contact happened
            else:
                self._ty = r["y"] + self.hi[1]
                return self._act(r, tx=self.cx, ty=PP_Y_TOUCH, tth=TH, tarm=PP_L_MIN, tgap=self.gap)
        if self.phase == "descend":
            g = r["y"] - r["arm_joint"]
            done_x = abs(r["x"] - self.cx) < 1e-3 and abs(r["finger_gap"] - self.gap) < 2e-3
            if done_x and abs(g - self.g_grasp) < 1e-3 and abs(r["arm_joint"] - PP_L_MAX) < 1e-3:
                self.phase = "close"
            elif not self._lim(60):
                return self._fail("descent blocked")
            else:
                reach = self.gap / 2 + PP_FT / 2 + 0.02
                path_top = self._max_top(others + [b], min(r["x"], self.cx) - reach, max(r["x"], self.cx) + reach)
                gt = self.g_grasp if done_x else max(self.g_grasp, path_top + 0.03 + PP_TIP)
                return self._act_g(r, gt, tx=self.cx, tgap=self.gap, l_lo=self.l_grasp, l_hi=self.l_grasp)
        if self.phase == "close":
            if b["held"] > 0.5:
                self.off_x = b["x"] - r["x"]
                self.phase = "carry"
            elif self._lim(12) and r["finger_gap"] > PP_GAP_MIN + 1e-3:
                return self._act(r, tx=self.cx, ty=r["y"], tth=TH, tarm=r["arm_joint"], tgap=PP_GAP_MIN)
            else:
                return self._fail("grasp not detected")
        if self.phase == "carry" and b["held"] < 0.5:
            return self._fail("object dropped")
        if self.phase == "carry":
            # lift / traverse / lower concurrently: the object's bottom always stays
            # PP_CLEAR_M above every object under the remaining path.
            if not (PP_XB_LO <= r["x"] + self.tx - b["rest_x"] <= PP_XB_HI):
                return self._fail("target x unreachable")
            if not self._lim(80):
                return self._fail("carry timeout")
            dx = self.tx - b["rest_x"]
            g = r["y"] - r["arm_joint"]
            h_need = self._clear_h(b, others, self.tx)
            # base circle must clear objects under its remaining path
            bt = self._max_top(others, min(r["x"], r["x"] + dx) - PP_BASE_R - 0.02,
                               max(r["x"], r["x"] + dx) + PP_BASE_R + 0.02)
            l_lo = float(np.clip(bt + 0.02 + PP_BASE_R - g, PP_L_MIN, PP_L_MAX))
            if abs(dx) > 1e-3:
                move = b["bot"] >= h_need - 0.005
                # do not dip below what the rest of the path needs
                return self._act_g(r, g + h_need - b["bot"], tx=r["x"] + dx if move else None, l_lo=l_lo)
            place = max(h_need, PP_TABLE + PP_PLACE_H)
            if abs(b["bot"] - place) > 1e-3:
                return self._act_g(r, g + place - b["bot"], tx=r["x"] + dx, l_lo=l_lo)
            if place > PP_TABLE + PP_PLACE_H + 1e-3:
                return self._fail("target region occupied")
            self.phase = "release"
        if self.phase == "release":
            if self._lim(15) and (b["held"] > 0.5 or r["finger_gap"] < min(PP_GAP_MAX, self.gap + 0.04) - 1e-3):
                return self._act(r, tgap=PP_GAP_MAX)
            self.phase = "retreat"
        if self.phase == "retreat":
            g = r["y"] - r["arm_joint"]
            gt = PP_CLEAR_TOP + PP_CLEAR_M + PP_TIP
            if self._lim() and g < gt - 2e-3:
                return self._act_g(r, gt)
            t = b["theta"] % np.pi
            upright = min(t, np.pi - t) < 0.05 and abs(b["bot"] - PP_TABLE) < 0.01
            self.err = b["x"] - self.tx
            self.done = True
            self.success = bool(upright and abs(self.err) <= 0.01 and b["held"] < 0.5)
            self.info = f"err={self.err:+.4f} th={b['theta']:+.3f} bot={b['bot']:.3f}"
            return None
        return self._act(r)
