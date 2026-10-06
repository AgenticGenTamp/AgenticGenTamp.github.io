"""Wall-unjam primitive: moves a dynamic block that is (nearly) flush against a side
wall away from that wall ("squeeze-out" by lowering the gripper into the block's
near-wall top corner, then pushing it away at table level)."""
import numpy as np

WALL_L, WALL_R = 0.0, 1.0 + np.sqrt(5)
Y_HIGH, L_MAX, ARM_MIN = 1.45, 0.48, 0.24
TIP_FINAL = 0.12        # lowest fingertip height at the bottom of the descent
SINK_TOL = 0.06         # abort if the block sinks this far into the table
V_DESC, V_PUSH = 0.0499, 0.02
PUSH_STEPS, G_TARGET, G_NEED = 30, 0.45, 0.30


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rect_xext(o):
    c, s = np.cos(o["theta"]), np.sin(o["theta"])
    hw, hh = o["width"] / 2, o["height"] / 2
    xs = [o["x"] + sx * hw * c - sy * hh * s for sx in (-1, 1) for sy in (-1, 1)]
    return min(xs), max(xs)


def _bottom_y(o):
    c, s = np.cos(o["theta"]), np.sin(o["theta"])
    return o["y"] - abs(o["width"] / 2 * s) - abs(o["height"] / 2 * c)


def _rest_y(o):
    return 0.1 + (o["height"] / 2 if abs(np.cos(o["theta"])) > 0.7 else o["width"] / 2)


def _gripper_parts(x, y, th, L, gap):
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
        if tth is not None: a[2] = _wrap(tth - r["theta"])
        if tarm is not None: a[3] = tarm - r["arm_joint"]
        if tgap is not None: a[4] = tgap - r["finger_gap"]
        return np.clip(a, self.lo, self.hi).astype(np.float64)

    def wall_gap(self, b):
        l, rr = _rect_xext(b)
        return min(l - WALL_L, WALL_R - rr)

    def _ok(self, b):
        t = b["theta"] % (np.pi / 2)
        flat = min(t, np.pi / 2 - t) < 0.05 and abs(b["y"] - _rest_y(b)) < 0.012
        return flat and self.wall_gap(b) >= G_NEED

    def _setup(self, b):
        l, rr = _rect_xext(b)
        self.d = -1.0 if (WALL_R - rr) < (l - WALL_L) else 1.0  # direction away from wall
        if self.configs is None:   # (finger_gap, tilt toward wall)
            self.configs = [(0.12, 0.4), (0.32, 0.0), (0.12, 0.5), (0.12, 0.3), (0.12, 0.6), (0.32, 0.2)]
        self._load_config()

    def _load_config(self):
        self.gap, a = self.configs[self.ci]
        self.th = -np.pi / 2 - self.d * a
        xs = np.concatenate([p[:, 0] for p in _gripper_parts(0.0, 1.0, self.th, L_MAX, self.gap)])
        if self.d < 0:
            self.bx = min(WALL_R - 0.012 - xs.max(), WALL_R - 0.245)
        else:
            self.bx = max(WALL_L + 0.012 - xs.min(), WALL_L + 0.245)
        self.xs = float(np.clip(self.bx + self.d * 0.3, 0.6, WALL_R - 0.6))
        low = min(p[:, 1].min() for p in _gripper_parts(0.0, 0.0, self.th, L_MAX, self.gap)[1:])
        self.y_final = TIP_FINAL - low

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
            if self._lim() and (r["y"] < Y_HIGH - 0.01 or abs(r["arm_joint"] - ARM_MIN) > 2e-3):
                return self._act(r, ty=Y_HIGH, tarm=ARM_MIN)
            self.phase = "rotate"
        if self.phase == "rotate":
            if self._lim() and (abs(r["x"] - self.xs) > 2e-3 or abs(_wrap(self.th - r["theta"])) > 1e-3
                                or abs(r["finger_gap"] - self.gap) > 2e-3):
                # first move (arm retracted) to a safe x away from the wall, then rotate there
                far = abs(r["x"] - self.xs) > 0.05
                return self._act(r, tx=self.xs, ty=Y_HIGH, tth=None if far else self.th,
                                 tarm=ARM_MIN, tgap=self.gap)
            self.phase = "travel"
        if self.phase == "travel":
            if self._lim() and abs(r["x"] - self.bx) > 2e-3:
                return self._act(r, tx=self.bx, ty=Y_HIGH, tth=self.th, tarm=ARM_MIN, tgap=self.gap)
            self.phase = "extend"
        if self.phase == "extend":
            if self._lim() and abs(r["arm_joint"] - L_MAX) > 2e-3:
                return self._act(r, tx=self.bx, ty=Y_HIGH, tth=self.th, tarm=L_MAX, tgap=self.gap)
            self.phase = "descend"
        if self.phase == "descend":
            if b["y"] < _rest_y(b) - SINK_TOL:
                self.phase = "retreat"      # block being crushed into table: abort
            elif self._lim() and abs(r["y"] - self.y_final) > 1e-3:
                return self._act(r, tx=self.bx, ty=max(self.y_final, r["y"] - V_DESC),
                                 tth=self.th, tarm=L_MAX, tgap=self.gap)
            else:
                self.phase, self.k = "push", 0
        if self.phase == "push":
            if self.k < PUSH_STEPS and self.wall_gap(b) < G_TARGET and _bottom_y(b) > 0.1 - SINK_TOL:
                self.k += 1
                return self._act(r, tx=r["x"] + self.d * V_PUSH, ty=self.y_final,
                                 tth=self.th, tarm=L_MAX, tgap=self.gap)
            self.phase, self.k = "retreat", 0
        if self.phase == "retreat":
            if r["y"] < Y_HIGH - 0.01 and self.k < 80:
                self.k += 1
                return self._act(r, ty=Y_HIGH)
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
            l, rr = _rect_xext(b)
            self.d = -1.0 if (WALL_R - rr) < (l - WALL_L) else 1.0
            self._load_config()
            self.phase = "lift"
            return self.step(state)
        return self._act(r)
