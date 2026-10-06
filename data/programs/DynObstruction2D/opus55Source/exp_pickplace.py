"""Top-down pick-and-place primitive for DynObstruction2DEnv (step-wise, WallUnjam style).

PickPlace(action_space, obj_name, target_x).step(state) -> action or None (see .success).
Grasp: theta=-pi/2, arm fully extended (L=0.48), gripper plate centre G placed
PP_TOP_OFF=0.06 above the object top (object top then sits between the plate
bottom (G-0.03) and the grasp-detection limit (G-0.10)); fingers open to the
narrowest gap that straddles the object, then close at max speed until held.
"""
import numpy as np



def _wu_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wu_gripper_parts(x, y, th, L, gap):
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


class PickPlace:
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
        return PickPlace.plan_grasp(state, obj_name) is not None

    # ---- helpers -----------------------------------------------------
    def _act(self, r, tx=None, ty=None, tth=None, tarm=None, tgap=None):
        a = np.zeros(5)
        if tx is not None: a[0] = tx - r["x"]
        if ty is not None: a[1] = ty - r["y"]
        if tth is not None: a[2] = _wu_wrap(tth - r["theta"])
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
        parts = _wu_gripper_parts(r["x"], r["y"], r["theta"], r["arm_joint"], r["finger_gap"])
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
            aligned = abs(_wu_wrap(TH - r["theta"])) < 1e-3
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
