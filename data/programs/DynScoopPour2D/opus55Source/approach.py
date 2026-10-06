import numpy as np

LIM = np.array([0.03, 0.03, np.pi / 32, 0.08, 0.015]) * 0.999
DOWN = -np.pi / 2
WALL_L, WALL_R, WALL_TOP = 1.70, 1.80, 1.5
WALL_C = 1.75
ARM = 0.4
SWEEP_DEPTH = -0.075   # hook corner y while sweeping underground
CARGO_Y = 1.53         # hook corner y needed for cargo to clear wall
DUMP_XV = 2.4          # hook corner x when dumping
DUMP_DEPTH = -0.25
TILT = DOWN - 0.15
ACC = 0.004
B2_EXTRA = 0.015
VCORNER = 0.012
SLIDE_V = 0.003
FAR_X = 3.405
OLD_FAR = [(0.09, 0.0012, 0.01, 0.03, 0.21, 0.0012), (0.092, 0.0012, 0.01, 0.025, 0.21, 0.0012),
           (0.09, 0.0015, 0.01, 0.045, 0.21, 0.0012), (0.09, 0.0008, 0.006, 0.03, 0.21, 0.0012),
           (0.088, 0.0012, 0.01, 0.04, 0.21, 0.0011)]
FAR_PARAMS = [(0.09, 0.19, 0.0015, 0.025, 0.005, 10), (0.092, 0.188, 0.002, 0.03, 0.005, 10),
              (0.088, 0.192, 0.001, 0.025, 0.004, 12)]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    # ------------------------------------------------------------------ utils
    def _parse(self, state):
        self.robot = None
        self.hook = None
        self.objs = []
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            t = o.type.name
            if t == "kin_robot":
                self.robot = {f: float(state.get(o, f)) for f in
                              ["x", "y", "theta", "arm_joint", "finger_gap"]}
            elif t == "hook":
                self.hook = {f: float(state.get(o, f)) for f in
                             ["x", "y", "theta", "held"]}
            elif t in ("small_circle", "small_square"):
                if t == "small_circle":
                    r = float(state.get(o, "radius"))
                else:
                    r = float(state.get(o, "size")) * 0.5
                self.objs.append((float(state.get(o, "x")),
                                  float(state.get(o, "y")), r))

    def _toward(self, tgt):
        r = self.robot
        cur = [r["x"], r["y"], r["theta"], r["arm_joint"], r["finger_gap"]]
        a = np.zeros(5)
        for i, t in enumerate(tgt):
            if t is None:
                continue
            a[i] = wrap(t - cur[i]) if i == 2 else t - cur[i]
        m = max(abs(a[0]), abs(a[1]))
        if m > LIM[0]:
            a[:2] *= LIM[0] / m
        return np.clip(a, -LIM, LIM).astype(np.float32)

    def _smooth(self, a, tgt, vc=0.0):
        r = self.robot
        d = np.array([tgt[0] - r["x"], tgt[1] - r["y"]])
        dist = np.max(np.abs(d))
        if dist < 1e-9:
            return a
        sp = min(0.03, np.sqrt(vc * vc + 2 * ACC * dist))
        vd = d / dist * sp
        dv = np.clip(vd - self.v, -ACC, ACC)
        v = self.v + dv
        # never overshoot
        if vc <= 0:
            for i in range(2):
                if abs(v[i]) > abs(d[i]) and np.sign(v[i]) == np.sign(d[i]):
                    v[i] = d[i]
        a = a.copy()
        a[:2] = np.clip(v, -LIM[:2], LIM[:2])
        return a.astype(np.float32)

    def _reached(self, tgt, tol=2e-3):
        r = self.robot
        cur = [r["x"], r["y"], r["theta"], r["arm_joint"], r["finger_gap"]]
        for i, t in enumerate(tgt):
            if t is None:
                continue
            d = wrap(t - cur[i]) if i == 2 else t - cur[i]
            if abs(d) > tol:
                return False
        return True

    # ---------------------------------------------------------------- episode
    def reset(self, state, info):
        self._parse(state)
        self.n_total = len(self.objs)
        self.queue = []
        self.phase = "start"
        self.off = None
        self.wait = 0
        self.grasp_tries = 0
        self.v = np.zeros(2)
        self.DN, self.TL = DOWN, TILT
        self.gshift = 0.0
        self.far_mode = False
        self.last_pose = None
        self.stall = 0

    def _off(self, th):
        c, si = np.cos(th), np.sin(th)
        rx, ry = self.rel
        return (c * rx - si * ry, si * rx + c * ry)

    def _rob_for(self, hx, hy, th):
        ox, oy = self._off(th)
        return (hx - ox, hy - oy)

    def _left_objs(self):
        return [(x, y, r) for (x, y, r) in self.objs if x < WALL_C]

    def get_action(self, state):
        self._parse(state)
        if self.phase == "grasp" and self.far_mode and self.hook["held"] > 0.5 and self.queue:
            self.queue = []
        for _ in range(8):
            if not self.queue:
                self._plan()
            if not self.queue:
                return np.zeros(5, dtype=np.float32)
            item = self.queue[0]
            if item[0] == "raw":
                self.queue.pop(0)
                return np.asarray(item[1], dtype=np.float32)
            if item[0] == "fslide":
                a = self._fslide(item[1])
                if a is None:
                    self.queue.pop(0)
                    continue
                return a
            if item[0] == "wait":
                if item[1] <= 0:
                    self.queue.pop(0)
                    continue
                self.queue[0] = ("wait", item[1] - 1)
                return np.zeros(5, dtype=np.float32)
            tgt = item[1]
            r = self.robot
            pose = np.array([r["x"], r["y"], r["theta"], r["arm_joint"], r["finger_gap"]])
            if self.last_pose is not None and np.abs(pose - self.last_pose).max() < 1e-4:
                self.stall += 1
            else:
                self.stall = 0
            self.last_pose = pose
            vc = item[3] if len(item) > 3 else 0.0
            if vc > 0 and max(abs(tgt[0] - r["x"]), abs(tgt[1] - r["y"])) < \
                    max(np.max(np.abs(self.v)), 2e-3) + 1e-4:
                self.queue.pop(0)
                continue
            if self._reached(tgt) or self.stall >= 6:
                self.queue.pop(0)
                self.stall = 0
                continue
            a = self._toward(tgt)
            if len(item) > 2 and item[2]:
                a = self._smooth(a, tgt, item[3] if len(item) > 3 else 0.0)
            self.v = np.array(a[:2], dtype=float)
            return a
        return np.zeros(5, dtype=np.float32)

    def _fslide(self, p):
        """closed-loop slide of the left finger face over the hook's top-left corner"""
        r, h = self.robot, self.hook
        if h["held"] > 0.5 or p["s"] < p["s_end"]:
            return None
        th = p["th"]
        u = np.array([np.cos(th), np.sin(th)])
        w = np.array([-np.sin(th), np.cos(th)])
        P = np.array([h["x"], h["y"]]) + self._rotv(h["theta"] - DOWN, (-0.05, 0.5))
        closing = p["s"] <= p["s_close"]
        gap_next = r["finger_gap"] - (p["dg"] if closing else 0.0)
        lat = p["lat"]
        G = P - p["s"] * u - (gap_next / 2 + 0.02 - p["eps"] + lat) * w
        B = G - ARM * u
        d = np.array([B[0] - r["x"], B[1] - r["y"]])
        m = np.max(np.abs(d))
        if m > 0.006:
            d *= 0.006 / m
        if lat > 0:
            p["lat"] = max(0.0, lat - 0.006)
        else:
            p["s"] -= p["v"] if closing or p["s"] - 0.006 < p["s_close"] else 0.006
        a = np.array([d[0], d[1], th - r["theta"], ARM - r["arm_joint"],
                      -p["dg"] if closing and lat <= 0 else 0.0])
        return np.clip(a, -LIM * 0.999, LIM * 0.999).astype(np.float32)

    @staticmethod
    def _rotv(a, v):
        c, s = np.cos(a), np.sin(a)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    # --------------------------------------------------------- path planning
    def _valid(self, x, y, mode):
        """mode: 'free' (no hook), 'hook' (hook held, no cargo),
        'cargo' (hook carrying objects)."""
        if x < 0.199 or x > 3.301 or y < 0.199 or y > 2.801:
            return False
        if abs(x - WALL_C) < 0.05 + 0.2 + 0.006 and y < WALL_TOP + 0.22:
            return False
        if mode == "free":
            if abs(x - WALL_C) < 0.05 + 0.15 and y - ARM - 0.14 < WALL_TOP + 0.02:
                return False
        else:
            ox, oy = self.off
            if abs(x + self.gshift - WALL_C) < 0.05 + 0.15 and \
                    y - ARM - 0.14 < WALL_TOP + 0.02:
                return False
            if mode == "cargo":
                cx, cy = x + ox, y + oy
                if cx > WALL_L - 0.12 and cx - 0.55 < WALL_R + 0.08 and \
                        cy < CARGO_Y:
                    return False
        return True

    def _seg_ok(self, a, b, mode):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 0.02) + 2
        for t in np.linspace(0, 1, n):
            if not self._valid(a[0] + t * (b[0] - a[0]),
                               a[1] + t * (b[1] - a[1]), mode):
                return False
        return True

    @staticmethod
    def _cost(a, b):
        return max(abs(b[0] - a[0]), abs(b[1] - a[1]))

    def _lift_valid(self, p, mode):
        y = p[1]
        while not self._valid(p[0], y, mode) and y < 2.78:
            y += 0.02
        return (p[0], y)

    def _path(self, a, b, mode):
        pre, post = [], []
        if not self._valid(a[0], a[1], mode):
            a2 = self._lift_valid(a, mode)
            pre = [a2]
            a = a2
        if not self._valid(b[0], b[1], mode):
            b2 = self._lift_valid(b, mode)
            post = [b]
            b = b2
        return pre + self._path2(a, b, mode) + post

    def _path2(self, a, b, mode):
        if self._seg_ok(a, b, mode):
            return [b]
        best, bc = None, 1e9
        for wx in np.arange(0.25, 3.26, 0.05):
            for wy in np.arange(1.7, 2.79, 0.05):
                w = (wx, wy)
                c = self._cost(a, w) + self._cost(w, b)
                if c >= bc:
                    continue
                if self._valid(wx, wy, mode) and self._seg_ok(a, w, mode) \
                        and self._seg_ok(w, b, mode):
                    best, bc = [w, b], c
        # two waypoints joined by a horizontal segment
        xs = np.arange(0.25, 3.26, 0.04)
        for wy in np.arange(1.7, 2.79, 0.04):
            A = [wx for wx in xs if abs(wx - a[0]) <= abs(wy - a[1]) + 0.06
                 and self._valid(wx, wy, mode) and self._seg_ok(a, (wx, wy), mode)]
            if not A:
                continue
            B = [wx for wx in xs if abs(wx - b[0]) <= abs(wy - b[1]) + 0.06
                 and self._valid(wx, wy, mode) and self._seg_ok((wx, wy), b, mode)]
            for w1 in A:
                for w2 in B:
                    c = self._cost(a, (w1, wy)) + abs(w2 - w1) + self._cost((w2, wy), b) + 0.1
                    if c < bc and self._seg_ok((w1, wy), (w2, wy), mode):
                        best, bc = [(w1, wy), (w2, wy), b], c
        if best is not None:
            return best
        h = 2.75
        return [(a[0], h), (b[0], h), b]

    def _go_path(self, a, b, mode, gap=None, th=None, smooth=False):
        if th is None:
            th = self.DN
        if self.off is not None:
            self.off = self._off(th)
        ws = self._path(a, b, mode)
        for i, w in enumerate(ws):
            vc = VCORNER if (smooth and i < len(ws) - 1) else 0.0
            self.queue.append(("go", (w[0], w[1], th, ARM, gap), smooth, vc))

    # --------------------------------------------------------------- planning
    def _old_far(self, r, X, Y):
        k = (self.grasp_tries // 2) % len(OLD_FAR)
        alpha, pdx, marg, hh, gap, dg = OLD_FAR[k]
        th = DOWN + alpha
        self.far_mode = True
        c, si = np.cos(th), np.sin(th)
        rot = lambda v: np.array([c * v[0] - si * v[1], si * v[0] + c * v[1]])
        P = np.array([X - 0.05, Y + 0.5])
        G = P - rot((hh, gap / 2 + marg))
        B = G - rot((ARM, 0))
        B2 = P - rot((hh, gap / 2 + marg + B2_EXTRA)) - rot((ARM, 0))
        if r["y"] < B[1] + 0.149:
            self.queue.append(("go", (r["x"], B[1] + 0.15, None, None, None)))
        self.queue.append(("go", (B2[0], B2[1] + 0.15, th, ARM, gap)))
        self.queue.append(("go", (B2[0], B2[1], th, ARM, gap)))
        self.queue.append(("go", (B[0], B[1], th, ARM, gap)))
        self.phase = "grasp"
        self.press = [pdx, 0, 0, 0, -dg]
        self.max_press = int(1.5 * marg / max(pdx - dg / 2, 2e-4)) + 10
        self.wait = 0

    def _plan(self):
        r = self.robot
        if self.phase == "start":
            y0 = r["y"]
            if abs(r["x"] - WALL_C) < 0.6:
                y0 = max(y0, 2.2)
            self.queue.append(("go", (r["x"], y0, DOWN, None, 0.09)))
            self.phase = "approach"
            return
        if self.phase == "approach" and self.hook["x"] > 3.494:
            self.queue.append(("go", (r["x"], max(r["y"], 0.9), DOWN, ARM, 0.09)))
            self.phase = "far_approach"
            return
        if self.phase == "approach" and self.hook["x"] > FAR_X:
            X = self.hook["x"]
            px = X - 0.6
            self._go_path((r["x"], r["y"]), (px, 0.9), "free", 0.09)
            self.queue.append(("go", (px, 0.56, DOWN, ARM, 0.09)))
            self.phase = "push"
            return
        if self.phase == "push":
            if self.hook["x"] > 3.498 or r["x"] > 3.29:
                self.queue.append(("raw", [0.003, 0, 0, 0, 0]))
                self.queue.append(("wait", 2))
                self.queue.append(("go", (r["x"] + 0.003, 0.9, DOWN, ARM, 0.09)))
                self.phase = "far_approach"
                return
            dx = 0.03 if self.hook["x"] < 3.44 else 0.005
            self.queue.append(("go", (r["x"] + dx, 0.56, DOWN, ARM, 0.09)))
            return
        if self.phase == "far_approach":
            # slide the tilted left finger down along its own axis so that its
            # outer face rubs over the hook's top-left corner, closing the gap
            # during the last part of the slide.
            X, Y = self.hook["x"], self.hook["y"]
            if self.grasp_tries % 2 == 0:
                self._old_far(r, X, Y)
                return
            k = (self.grasp_tries // 2) % len(FAR_PARAMS)
            alpha, gf, eps, hh, v, K = FAR_PARAMS[k]
            K = int(K)
            dg = 0.0012
            th = DOWN + alpha
            self.far_mode = True
            u = np.array([np.cos(th), np.sin(th)])
            w = np.array([-np.sin(th), np.cos(th)])
            P = np.array([X - 0.05, Y + 0.5])
            Gf = P - hh * u - (gf / 2 + 0.02 - eps) * w
            Gc = Gf - K * v * u - K * (dg / 2) * w
            gc = gf + K * dg
            L = 0.10
            Gs = Gc - L * u
            Bs = Gs - ARM * u
            Bclr = Bs - 0.03 * w
            if r["y"] < Bclr[1] + 0.12:
                self.queue.append(("go", (r["x"], Bclr[1] + 0.12, None, None, None)))
            self.queue.append(("go", (Bclr[0], Bclr[1] + 0.12, th, ARM, gc)))
            self.queue.append(("go", (Bclr[0], Bclr[1], th, ARM, gc)))
            self.queue.append(("fslide", dict(th=th, s=hh + K * v + L, s_close=hh + K * v,
                                              s_end=hh - 0.006, v=v, dg=dg, eps=eps, lat=0.03)))
            self.phase = "grasp"
            self.press = [0, 0, 0, 0, -dg]
            self.max_press = 2
            self.wait = 0
            return
        if self.phase == "approach":
            X, Y = self.hook["x"], self.hook["y"]
            self.press = [0.006, 0, 0, 0, -0.002]
            self.far_mode = False
            self.max_press = 6
            ax = min(X - 0.123, 3.29)
            gy = Y + 0.53 + ARM
            above = (ax, gy + 0.12)
            self._go_path((r["x"], r["y"]), above, "free", 0.09)
            self.queue.append(("go", (ax, gy, DOWN, ARM, 0.09)))
            self.phase = "grasp"
            self.wait = 0
            return
        if self.phase == "grasp":
            if self.hook["held"] > 0.5:
                dx, dy = self.hook["x"] - r["x"], self.hook["y"] - r["y"]
                c, si = np.cos(-r["theta"]), np.sin(-r["theta"])
                self.rel = (c * dx - si * dy, si * dx + c * dy)
                self.hrel = wrap(self.hook["theta"] - r["theta"])
                self.DN = DOWN - self.hrel
                self.TL = TILT - self.hrel
                self.off = self._off(self.DN)
                self.gshift = (ARM + 0.07) * np.cos(self.DN)
                if r["finger_gap"] > 0.1:
                    self.queue.append(("go", (r["x"], r["y"], r["theta"], ARM, 0.09)))
                    return
                self.phase = "to_left"
                self._plan()
                return
            if self.far_mode and self.wait > 0 and (
                    abs(wrap(self.hook["theta"] - DOWN)) > 0.006 or
                    (r["x"] > 3.2995 and self.wait > 4)):
                self.grasp_tries += 1
                self.queue = [("go", (r["x"] - 0.03, r["y"] + 0.04, r["theta"], ARM, None)),
                              ("wait", 3)]
                self.phase = "far_approach"
                return
            self.queue.append(("raw", self.press))
            self.wait += 1
            if self.wait > self.max_press:
                self.grasp_tries += 1
                self.queue = [("go", (r["x"] - 0.04, r["y"] + 0.1, DOWN, ARM, 0.09))]
                self.phase = "approach"
            return
        self.off = self._off(self.DN)
        offx, offy = self.off
        if self.phase == "to_left":
            xv = self._choose_descent()
            bx = xv - offx
            by_above = 0.3 - offy
            self._go_path((r["x"], r["y"]), (bx, by_above), "hook")
            self.queue.append(("go", (bx, SWEEP_DEPTH - offy, self.DN, ARM, None)))
            self.phase = "sweep"
            return
        if self.phase == "sweep":
            xv = self.hook["x"]
            stop = self._sweep_stop(xv)
            self.queue.append(("go", (stop - offx, SWEEP_DEPTH - offy, self.DN, ARM, None)))
            self.phase = "lift"
            return
        if self.phase == "lift":
            hx = self.hook["x"]
            tb = self._rob_for(hx, SWEEP_DEPTH + 0.1, self.TL)
            self.off = self._off(self.TL)
            y1 = max(tb[1], 0.3 - self.off[1])
            dump_b = self._rob_for(DUMP_XV, DUMP_DEPTH, self.TL)
            self._go_path((tb[0], y1), dump_b, "cargo", th=self.TL, smooth=True)
            self.phase = "after_dump"
            return
        if self.phase == "after_dump":
            b = self._rob_for(DUMP_XV, DUMP_DEPTH, self.DN)
            self.queue.append(("go", (b[0], b[1], self.DN, ARM, None)))
            self.off = self._off(self.DN)
            offx, offy = self.off
            bx = min(DUMP_XV + 0.58 - offx, 3.29)
            self.queue.append(("go", (bx, b[1], self.DN, ARM, None)))
            self.queue.append(("go", (bx, 0.3 - offy, self.DN, ARM, None)))
            self.phase = "to_left"
            return

    def _choose_descent(self):
        # descending onto objects is fine: they pop up onto the bar
        return min(1.6, 1.49 + self.off[0])

    def _sweep_stop(self, xv):
        objs = [o for o in self._left_objs() if o[0] < xv]
        stop = 0.56
        if objs:
            minx = min(x - rr for (x, y, rr) in objs)
            stop = max(stop, min(xv, minx + 0.45))
        stop = min(stop, 1.485 + self._off(self.TL)[0])
        return max(stop, 0.22 + self.off[0])
