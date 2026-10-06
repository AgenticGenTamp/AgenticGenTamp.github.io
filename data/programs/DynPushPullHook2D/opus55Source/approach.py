import numpy as np

WALL_Y = 1.775
BASE_YMAX = 1.45
BASE_YMIN = 0.26
BASE_XMIN = 0.26
BASE_XMAX = 3.24
ARM_MIN, ARM_MAX = 0.24, 0.48
GRIP_X = 0.045  # bar end position in gripper frame at grasp


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(t, v):
    c, s = np.cos(t), np.sin(t)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class GeneratedApproach:
    USE_LEFT = True
    LEFT_MARGIN = 0.1
    V = 0.047
    GAP0 = 0.24
    RISE = True
    DIRECT = True
    DIRECT_APP = True
    TDIAG = 0.0
    DRAG = 0.1
    TWOSEG = True
    EXCL = True
    TWOAPP = True
    SHORT_SWEEP = True
    DRAG_ABORT = 0.2
    DM = 0.06
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.lo = action_space.low
        self.hi = action_space.high

    # ------------------------------------------------------------------ utils
    def _read(self, state):
        d = {}
        for o in state.get_objects(state.type_features and list(state.type_features)[0]) if False else []:
            pass
        robot = hook = target = None
        obs = []
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            t = o.type.name if hasattr(o.type, "name") else str(o.type)
            if t == "kin_robot":
                robot = o
            elif t == "hook":
                hook = o
            elif t == "target_block":
                target = o
            elif t == "dyn_rectangle":
                obs.append(o)
        g = lambda o, f: float(state.get(o, f))
        r = dict(x=g(robot, "x"), y=g(robot, "y"), th=g(robot, "theta"),
                 arm=g(robot, "arm_joint"), gap=g(robot, "finger_gap"))
        h = dict(x=g(hook, "x"), y=g(hook, "y"), th=g(hook, "theta"),
                 w=g(hook, "width"), l1=g(hook, "length_side1"),
                 l2=g(hook, "length_side2"), held=g(hook, "held"))
        tb = dict(x=g(target, "x"), y=g(target, "y"), th=g(target, "theta"),
                  w=g(target, "width"), h=g(target, "height"))
        ob = [dict(x=g(o, "x"), y=g(o, "y"), th=g(o, "theta"), w=g(o, "width"),
                   h=g(o, "height")) for o in obs]
        return r, h, tb, ob

    @staticmethod
    def _extent(b):
        c, s = abs(np.cos(b["th"])), abs(np.sin(b["th"]))
        hx = 0.5 * (b["w"] * c + b["h"] * s)
        hy = 0.5 * (b["w"] * s + b["h"] * c)
        return b["x"] - hx, b["x"] + hx, b["y"] - hy, b["y"] + hy

    def _act(self, r, tgt, dgrip=0.0):
        bx, by, th, arm = tgt
        a = np.zeros(5)
        a[0] = bx - r["x"]
        a[1] = by - r["y"]
        a[2] = wrap(th - r["th"])
        a[3] = arm - r["arm"]
        a[4] = dgrip
        # scale translation jointly to keep direction
        n = max(abs(a[0]) / 0.049, abs(a[1]) / 0.049, 1.0)
        a[0] /= n
        a[1] /= n
        a[2] = np.clip(a[2], -0.064, 0.064)
        a[3] = np.clip(a[3], -0.099, 0.099)
        a[4] = np.clip(a[4], -0.0199, 0.0199)
        return a

    def _hook_rects(self, h, infl=0.03):
        """Hook rectangles as (center-frame origin, theta, xmin, xmax, ymin, ymax)."""
        C = np.array([h["x"], h["y"]])
        return [(C, h["th"], -h["l1"] - infl, infl, -h["w"] - infl, infl),
                (C, h["th"], -h["w"] - infl, infl, -h["l2"] - infl, infl)]

    def _grip_hits(self, bx, by, th, arm, rects):
        u = np.array([np.cos(th), np.sin(th)])
        n = np.array([-u[1], u[0]])
        G = np.array([bx, by]) + arm * u
        for a in (-0.03, 0.08, 0.2):
            for b in (-0.17, -0.08, 0.0, 0.08, 0.17):
                p = G + a * u + b * n
                for (C, ht, x0, x1, y0, y1) in rects:
                    q = rot(-ht, p - C)
                    if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                        return True
        return False

    def _robot_hits(self, bx, by, th, arm, rects):
        if self._grip_hits(bx, by, th, arm, rects):
            return True
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            p = np.array([bx + 0.25 * np.cos(a), by + 0.25 * np.sin(a)])
            for (C, ht, x0, x1, y0, y1) in rects:
                q = rot(-ht, p - C)
                if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                    return True
        return False

    def _app_seg(self, P0, th0, arm0, P1, th1, rects):
        d = wrap(th1 - th0)
        dist = float(np.linalg.norm(P1 - P0))
        K = int(max(1, np.ceil(dist / 0.3), np.ceil(abs(d) / 0.3)))
        for k in range(K):
            for fp in np.linspace(k / K, (k + 1) / K, 4):
                for ft in np.linspace(k / K, (k + 1) / K, 4):
                    P = P0 + fp * (P1 - P0)
                    for arm in (arm0, ARM_MIN):
                        if self._robot_hits(P[0], P[1], th0 + ft * d, arm, rects):
                            return None
        return [(float(P0[0] + f * (P1[0] - P0[0])), float(P0[1] + f * (P1[1] - P0[1])),
                 wrap(th0 + f * d), ARM_MIN) for f in np.linspace(0, 1, K + 1)[1:]]

    def _direct_app(self, r, h, pre, th1, y1=None):
        rects = self._hook_rects(h, infl=0.05)
        P0 = np.array([r["x"], r["y"]])
        P1 = np.array([pre[0], pre[1]])
        w = self._app_seg(P0, r["th"], r["arm"], P1, th1, rects)
        if w is not None or not self.TWOAPP or y1 is None:
            return w
        for M in (np.array([pre[0], y1]), np.array([r["x"], y1]),
                  np.array([0.5 * (r["x"] + pre[0]), y1])):
            w1 = self._app_seg(P0, r["th"], r["arm"], M, th1, rects)
            if w1 is None:
                continue
            w2 = self._app_seg(M, th1, ARM_MIN, P1, th1, rects)
            if w2 is not None:
                return w1 + w2
        return None

    def _safe_heading(self, r, th_goal, h):
        """Intermediate heading turning in a direction whose sweep avoids the hook."""
        d0 = wrap(th_goal - r["th"])
        if abs(d0) < 0.07:
            return th_goal
        rects = self._hook_rects(h)
        opts = []
        for d in (d0, d0 - np.sign(d0) * 2 * np.pi):
            hit = False
            for f in np.linspace(0, 1, max(2, int(abs(d) / 0.1) + 1))[1:]:
                if self._grip_hits(r["x"], r["y"], r["th"] + f * d, r["arm"], rects):
                    hit = True
                    break
            opts.append((hit, abs(d), d))
        opts.sort()
        d = opts[0][2]
        return wrap(r["th"] + np.sign(d) * min(abs(d), 0.5))

    def _act_safe(self, r, tgt, dgrip, h):
        th = self._safe_heading(r, tgt[2], h)
        return self._act(r, (tgt[0], tgt[1], th, tgt[3]), dgrip)

    @staticmethod
    def _close(r, tgt, tol=0.01, ttol=0.02):
        return (abs(r["x"] - tgt[0]) < tol and abs(r["y"] - tgt[1]) < tol
                and abs(wrap(r["th"] - tgt[2])) < ttol and abs(r["arm"] - tgt[3]) < tol)

    # -------------------------------------------------------------- episode
    def reset(self, state, info):
        self.phase = "start"
        self.wps = []
        self.off = None
        self.steps = 0
        self.close_count = 0
        self.n_sweeps = 0
        self.n_retrav = 0
        self.diag_off = False
        self.n_abort = 0
        self.trav_tx = None
        self.moves = []

    def _pre(self, E, u, dmin=0.24):
        """Pre-grasp base position on the bar axis behind the bar end."""
        fin = E - (GRIP_X + ARM_MIN) * u
        for d in (0.34, 0.3, 0.27, dmin):
            pre = E - (d + ARM_MIN) * u
            pre[1] = min(pre[1], BASE_YMAX)
            if pre[0] >= BASE_XMIN and pre[1] >= BASE_YMIN and fin[1] <= BASE_YMAX:
                return pre
        return None

    def _bar_end(self, h):
        # center of long bar free end
        return np.array([h["x"], h["y"]]) + rot(h["th"], [-h["l1"], -h["w"] / 2])

    def _ik(self, C, th):
        """Base pose so that hook corner is at C with robot heading th."""
        u = np.array([np.cos(th), np.sin(th)])
        p = C - rot(th, self.off)
        best = None
        for arm in np.linspace(ARM_MAX, ARM_MIN, 13):
            B = p - arm * u
            pen = max(0, B[1] - BASE_YMAX) + max(0, BASE_YMIN - B[1]) + \
                max(0, BASE_XMIN - B[0]) + max(0, B[0] - BASE_XMAX)
            if best is None or pen < best[0] - 1e-9:
                best = (pen, B, arm)
        return best

    def _ik_tilt(self, C, th0=np.pi / 2, max_k=30, prefer=1):
        for k in range(0, max_k):
            for sgn in (prefer, -prefer):
                th = th0 + sgn * 0.03 * k
                pen, B, arm = self._ik(C, th)
                if pen < 1e-6:
                    return (B[0], B[1], th, arm), True
        pen, B, arm = self._ik(C, th0)
        return (float(np.clip(B[0], BASE_XMIN, BASE_XMAX)), float(np.clip(B[1], BASE_YMIN, BASE_YMAX)), th0, arm), False

    def get_action(self, state):
        self.steps += 1
        r, h, tb, ob = self._read(state)
        hold = 0.0 if self.phase in ("start", "approach", "pregrasp", "push") else -0.02

        if self.phase == "start":
            E = self._bar_end(h)
            u = np.array([np.cos(h["th"]), np.sin(h["th"])])
            pre = self._pre(E, u)
            if pre is None:
                self.phase = "push"
                self.push_stage = 0
            else:
                self.phase = "approach"
                self.app_stage = 0
                self.app_wps = None

        if self.phase == "push":
            return self._push(r, h)

        if self.phase == "approach":
            E = self._bar_end(h)
            th = h["th"]
            u = np.array([np.cos(th), np.sin(th)])
            pre = self._pre(E, u)
            if pre is None:
                self.phase = "push"
                self.push_stage = 0
                return self._push(r, h)
            fin = E - (GRIP_X + ARM_MIN) * u
            C = np.array([h["x"], h["y"]])
            hk_min = min((C + rot(th, [lx, ly]))[1] for lx in (-h["l1"], -h["w"], 0.0)
                         for ly in (-h["w"], 0.0))
            hk_min = min(hk_min, min((C + rot(th, [lx, -h["l2"]]))[1] for lx in (-h["w"], 0.0)))
            y1 = float(np.clip(hk_min - 0.5, BASE_YMIN + 0.01, pre[1]))
            # narrow the gripper while approaching (keep >= 0.2)
            G = np.array([r["x"], r["y"]]) + r["arm"] * np.array([np.cos(r["th"]), np.sin(r["th"])])
            lat = abs(float(np.dot(G - E, [-u[1], u[0]])))
            gap_t = min(0.32, self.GAP0 + 2 * lat) if self.app_stage >= 2 else 0.32
            dg = float(np.clip(gap_t - r["gap"], -0.02, 0.02))
            if self.app_stage == 0 and self.DIRECT_APP and getattr(self, "app_wps", None) is None:
                self.app_wps = self._direct_app(r, h, pre, th, y1) or []
                if self.app_wps:
                    self.app_stage = -1
            if self.app_stage == -1:
                tgt = self.app_wps[0]
                last = len(self.app_wps) == 1
                if self._close(r, tgt, 0.01 if last else 0.04, 0.02 if last else 0.06):
                    self.app_wps.pop(0)
                    if not self.app_wps:
                        self.app_stage = 2
                    else:
                        tgt = self.app_wps[0]
                if self.app_stage == -1:
                    return self._act(r, tgt, dg)
            if self.app_stage == 0:
                # move low, retract arm and align heading simultaneously
                tgt = (r["x"], min(r["y"], y1), th, ARM_MIN)
                if abs(r["y"] - tgt[1]) < 0.03 and abs(r["arm"] - ARM_MIN) < 0.03:
                    self.app_stage = 1
            if self.app_stage == 1:
                tgt = (pre[0], y1, th, ARM_MIN)
                if self._close(r, tgt, 0.02, 0.03):
                    self.app_stage = 2
            if self.app_stage == 2:
                tgt = (pre[0], pre[1], th, ARM_MIN)
                if self._close(r, tgt, 0.01, 0.02):
                    self.app_stage = 3
            if self.app_stage == 3:
                tgt = (fin[0], fin[1], th, ARM_MIN)
                if self._close(r, tgt, 0.006, 0.02):
                    self.phase = "grasp"
                    self.close_count = 0
            if self.app_stage <= 1:
                return self._act_safe(r, tgt, dg, h)
            return self._act(r, tgt, dg)

        if self.phase == "grasp":
            self.close_count += 1
            if h["held"] > 0.5:
                G = np.array([r["x"], r["y"]]) + r["arm"] * np.array([np.cos(r["th"]), np.sin(r["th"])])
                self.off = rot(-r["th"], np.array([h["x"], h["y"]]) - G)
                self.dth = wrap(h["th"] - r["th"])
                self.hl1, self.hl2, self.hw = h["l1"], h["l2"], h["w"]
                self.phase = "plan"
            elif self.close_count > 14 or r["gap"] <= 0.125:
                # failed; reopen and retry
                self.phase = "reopen"
                self.reopen_count = 0
            else:
                return self._act(r, (r["x"], r["y"], r["th"], r["arm"]), -0.02)

        if self.phase == "reopen":
            self.reopen_count += 1
            if self.reopen_count > 12:
                self.phase = "start"
            u = np.array([np.cos(r["th"]), np.sin(r["th"])])
            return self._act(r, (r["x"] - 0.03 * u[0], r["y"] - 0.03 * u[1], r["th"], ARM_MIN), 0.02)

        if self.phase in ("plan", "exec", "moves"):
            if h["held"] < 0.5:
                self.phase = "start"
                return self._act(r, (r["x"], r["y"], r["th"], r["arm"]), 0.02)
            G = np.array([r["x"], r["y"]]) + r["arm"] * np.array([np.cos(r["th"]), np.sin(r["th"])])
            self.off = rot(-r["th"], np.array([h["x"], h["y"]]) - G)

        if self.phase == "plan":
            self._plan(r, h, tb, ob)
            self.phase = "exec"

        if self.phase == "exec":
            while self.wps:
                tgt, tol = self.wps[0]
                if self._close(r, tgt, tol, 0.03):
                    self.wps.pop(0)
                else:
                    break
            if not self.wps:
                self.phase = "moves"
                self._plan_moves(r, h, tb, ob)
            else:
                return self._act(r, self.wps[0][0], -0.02)

        if self.phase == "moves":
            return self._run_moves(r, h, tb, ob)

        return self._act(r, (r["x"], r["y"], r["th"], r["arm"]), hold)

    # ---------------------------------------------------------------- push
    def _push(self, r, h):
        """Push the hook's short bar rightwards so the long bar end is reachable."""
        E = self._bar_end(h)
        u = np.array([np.cos(h["th"]), np.sin(h["th"])])
        pre = self._pre(E, u, 0.32)
        if pre is not None:
            self.phase = "approach"
            self.app_stage = 0
            self.app_wps = None
            return self._act(r, (r["x"], r["y"], r["th"], ARM_MIN), 0.02)
        C = np.array([h["x"], h["y"]])
        th, w = h["th"], h["w"]
        if self.push_stage == 0:
            sb_pts = [C + rot(th, [lx, ly]) for lx in (-w, 0.0) for ly in (-h["l2"], 0.0)]
            sb_min = min(p[1] for p in sb_pts)
            sb_left = C + rot(th, [-w, -0.35])
            need = BASE_XMIN + 0.04 - (E[0] - (0.32 + ARM_MIN) * u[0])
            D = max(need, 0.05) + 0.06
            goal = sb_left[0] - ARM_MAX - 0.19 + D
            start = max(goal - D - 0.15, BASE_XMIN + 0.01)
            # long bar lower edge + end face sample points
            pts = [C + rot(th, [lx, -w]) for lx in np.linspace(-h["l1"], -w, 40)]
            pts += [C + rot(th, [-h["l1"], ly]) for ly in np.linspace(-w, 0, 4)]
            by = min(sb_left[1] + 0.05, BASE_YMAX)
            for p in pts:
                # base circle sweeping x in [start, goal]
                dx = max(start - p[0], p[0] - goal, 0.0)
                if dx < 0.24:
                    by = min(by, p[1] - np.sqrt(0.0576 - dx * dx) - 0.03)
                # gripper (extended) sweeping
                if start + ARM_MAX - 0.05 <= p[0] <= goal + ARM_MAX + 0.22:
                    by = min(by, p[1] - 0.19)
            if by < BASE_YMIN or by + 0.16 < sb_min + 0.04:
                if E[0] > BASE_XMIN + 0.3:
                    self.push_stage = 10
                    return self._base_push(r, h, E, u)
                by = max(by, BASE_YMIN)
            self.push_plan = (start, goal, by)
            self.push_stage = 1
        start, goal, by = self.push_plan
        if self.push_stage == 1:
            tgt = (r["x"], min(r["y"], by), 0.0, ARM_MIN)
            if self._close(r, tgt, 0.03, 0.05):
                self.push_stage = 2
            return self._act_safe(r, tgt, -0.02, h)
        if self.push_stage == 2:
            tgt = (start, by, 0.0, ARM_MIN)
            if self._close(r, tgt, 0.02, 0.03):
                self.push_stage = 3
            return self._act_safe(r, tgt, -0.02, h)
        if self.push_stage == 3:
            tgt = (start, by, 0.0, ARM_MAX)
            if self._close(r, tgt, 0.02, 0.03):
                self.push_stage = 4
            return self._act(r, tgt, -0.02)
        if self.push_stage == 4:
            tgt = (goal, by, 0.0, ARM_MAX)
            if abs(r["x"] - goal) < 0.01:
                self.push_stage = 5
            return self._act(r, tgt, -0.02)
        tgt = (goal - 0.08, by, 0.0, ARM_MIN)
        if self._close(r, tgt, 0.02, 0.03):
            self.push_stage = 0
            self.phase = "start"
        return self._act(r, tgt, -0.02)

    def _base_push(self, r, h, E, u):
        """Push the long bar end rightwards with the robot base (arm tucked)."""
        hd = -np.pi / 2 if E[1] - 0.5 > 0.0 else np.pi / 2
        ey = float(np.clip(E[1], BASE_YMIN, BASE_YMAX))
        if self.push_stage == 10:
            tgt = (BASE_XMIN + 0.01, r["y"], hd, ARM_MIN)
            if self._close(r, tgt, 0.03, 0.1):
                self.push_stage = 11
            return self._act(r, tgt, -0.02)
        if self.push_stage == 11:
            tgt = (BASE_XMIN + 0.01, ey, hd, ARM_MIN)
            if self._close(r, tgt, 0.02, 0.05):
                self.push_stage = 12
                need = BASE_XMIN + 0.05 - (E[0] - (0.32 + ARM_MIN) * u[0])
                self.push_goal = E[0] - 0.24 + max(need, 0.05) + 0.03
            return self._act(r, tgt, -0.02)
        if self.push_stage == 12:
            tgt = (self.push_goal, ey, hd, ARM_MIN)
            if abs(r["x"] - self.push_goal) < 0.01:
                self.push_stage = 13
            return self._act(r, tgt, -0.02)
        tgt = (self.push_goal - 0.1, ey, hd, ARM_MIN)
        if abs(r["x"] - tgt[0]) < 0.01:
            self.push_stage = 0
            self.phase = "start"
        return self._act(r, tgt, -0.02)

    # ---------------------------------------------------------------- plan
    def _corner(self, r):
        u = np.array([np.cos(r["th"]), np.sin(r["th"])])
        G = np.array([r["x"], r["y"]]) + r["arm"] * u
        return G + rot(r["th"], self.off)

    def _feasible(self, C, th):
        pen, B, arm = self._ik(C, th)
        return (float(np.clip(B[0], BASE_XMIN, BASE_XMAX)), float(np.clip(B[1], BASE_YMIN, BASE_YMAX)), th, arm), pen

    def _cmin(self):
        return BASE_YMIN + ARM_MIN + self.off[0]

    def _cmax(self):
        return BASE_YMAX + ARM_MAX + self.off[0]

    def _blocked(self, tb, ob):
        tl, tr, tbm, tt = self._extent(tb)
        for o in ob:
            ol, orr, obm, ot = self._extent(o)
            if orr > tl - 0.05 and ol < tr + 0.05 and ot > WALL_Y and o["y"] < tb["y"] and obm < tbm:
                return True
        return False

    def _sweep_left_ok(self, tb, ob, w, l2):
        tl, tr, tbm, tt = self._extent(tb)
        cx_L, cx_R = tl - l2 - w - 0.08, tr + 0.08
        if cx_R > BASE_XMAX + 0.03:
            return False
        cy_s = tbm - 0.04
        tot = 0.0
        for o in ob:
            ol, orr, obm, ot = self._extent(o)
            if orr > cx_L and ol < cx_R + w and obm < cy_s and ot > WALL_Y:
                tot += orr - ol
        return cx_L - tot > self.LEFT_MARGIN

    def _hook_top(self, by, th):
        """Max y of held hook with base at height by, heading th, arm min."""
        u = np.array([np.cos(th), np.sin(th)])
        C = np.array([0.0, by]) + ARM_MIN * u + rot(th, self.off)
        ht = th + self.dth
        ys = [(C + rot(ht, [lx, ly]))[1] for lx, ly in
              ((0, 0), (-self.hl1, 0), (-self.hl1, -self.hw), (0, -self.hl2), (-self.hw, -self.hl2))]
        return max(ys)

    def _plan(self, r, h, tb, ob):
        """Waypoints (base space) to bring the hook vertical at the insertion x."""
        tl, tr, tbm, tt = self._extent(tb)
        w, l2 = h["w"], h["l2"]
        wps = []
        ylow = BASE_YMIN + 0.01
        cx_ins = tl - l2 - w - 0.08
        self.sweep_dir = None
        if self._blocked(tb, ob) and tbm - 0.04 > WALL_Y + 0.1:
            self.sweep_dir = "R"
            if self.USE_LEFT and self._sweep_left_ok(tb, ob, w, l2):
                self.sweep_dir = "L"
                cx_ins = tr + 0.08
        ins, ok = self._ik_tilt(np.array([cx_ins, WALL_Y + 0.15]), max_k=22)
        th_ins = ins[2]
        pen, B, arm = self._ik(np.array([cx_ins, WALL_Y + 0.15]), th_ins)
        bx = float(np.clip(B[0], BASE_XMIN, BASE_XMAX))
        dth = wrap(th_ins - r["th"])

        def safe_until(y0):
            # furthest fraction of the rotation keeping the hook below the wall
            best = r["th"]
            for f in np.linspace(0, 1, 41):
                th = r["th"] + f * dth
                if self._hook_top(y0, th) > WALL_Y - 0.08:
                    break
                best = th
            return best

        if self.DIRECT:
            objs = [o for o in [tb] + list(ob) if self._extent(o)[3] > WALL_Y]
            if self.sweep_dir:
                cy_goal = tbm - 0.04
            else:
                cy_goal = min(tt + w + 0.05, self._cmax() - 0.01)
            _, C0 = self._hook_pts(bx, ylow, th_ins)
            y_top = float(np.clip(ylow + cy_goal - C0[1] - 0.01, ylow, BASE_YMAX))
            P0 = np.array([r["x"], r["y"]])
            objs2 = [tb] + [o for o in objs[1:] if self._hook_clear(bx, ylow, th_ins, [o], m=self.DM)]
            tries = [objs] + ([objs2] if (self.EXCL and len(objs2) < len(objs)) else [])
            for objs in tries:
                for yf in np.linspace(y_top, ylow, 6):
                    P1 = np.array([bx, yf])
                    dist = np.linalg.norm(P1 - P0)
                    K = int(max(np.ceil(abs(dth) / 0.4), np.ceil(dist / 0.4), 1))
                    ok = True
                    for f in np.linspace(0, 1, 4 * K + 1)[1:]:
                        P = P0 + f * (P1 - P0)
                        if not self._hook_clear(P[0], P[1], r["th"] + f * dth, objs, m=self.DM):
                            ok = False
                            break
                    if ok:
                        self.wps = [((float(P0[0] + f * (P1[0] - P0[0])), float(P0[1] + f * (P1[1] - P0[1])),
                                      wrap(r["th"] + f * dth), ARM_MIN), 0.03) for f in np.linspace(0, 1, K + 1)[1:]]
                        return
                if self.TWOSEG:
                    Pm = np.array([bx, ylow])
                    for fr in (0.75, 0.5, 0.25, 0.0):
                        thm = r["th"] + fr * dth
                        w1 = self._seg(P0, r["th"], Pm, thm, objs)
                        if w1 is None:
                            continue
                        for yf in np.linspace(y_top, ylow, 6):
                            w2 = self._seg(Pm, thm, np.array([bx, yf]), r["th"] + dth, objs)
                            if w2 is not None:
                                self.wps = w1 + w2
                                return
        th_a = safe_until(max(r["y"], ylow))
        th_b = safe_until(ylow)
        if abs(th_b - r["th"]) < abs(th_a - r["th"]):
            th_b = th_a
        wps.append(((r["x"], ylow, th_a, ARM_MIN), 0.03))
        wps.append(((bx, ylow, th_b, ARM_MIN), 0.03))
        rem = wrap(th_ins - th_b)
        n = int(np.ceil(abs(rem) / 0.5))
        ths = np.linspace(th_b, th_b + rem, n + 1)
        y_fin = ylow
        if self.RISE:
            cy_goal = (tbm - 0.04) if self.sweep_dir else min(tt + w + 0.05, self._cmax() - 0.01)
            _, C0 = self._hook_pts(bx, ylow, th_ins)
            y_fin = float(np.clip(ylow + cy_goal - C0[1] - 0.01, ylow, BASE_YMAX))
        objs = [o for o in [tb] + list(ob) if self._extent(o)[3] > WALL_Y]
        py, pth = ylow, ths[0]
        for th in ths[1:]:
            y = ylow
            if y_fin > ylow + 0.02:
                for yc in np.linspace(y_fin, py, 8):
                    if (self._hook_clear(bx, yc, th, objs) and
                            self._hook_clear(bx, 0.5 * (yc + py), 0.5 * (th + pth), objs)):
                        y = float(yc)
                        break
                y = max(y, py) if self._hook_clear(bx, max(y, py), th, objs) else y
            wps.append(((bx, y, wrap(th), ARM_MIN), 0.03))
            py, pth = y, th
        self.wps = wps

    def _seg(self, P0, th0, P1, th1, objs):
        d = th1 - th0
        dist = np.linalg.norm(P1 - P0)
        K = int(max(np.ceil(abs(d) / 0.4), np.ceil(dist / 0.4), 1))
        for f in np.linspace(0, 1, 4 * K + 1)[1:]:
            P = P0 + f * (P1 - P0)
            if not self._hook_clear(P[0], P[1], th0 + f * d, objs, m=self.DM):
                return None
        return [((float(P0[0] + f * (P1[0] - P0[0])), float(P0[1] + f * (P1[1] - P0[1])),
                  wrap(th0 + f * d), ARM_MIN), 0.03) for f in np.linspace(0, 1, K + 1)[1:]]

    def _hook_pts(self, bx, by, th):
        u = np.array([np.cos(th), np.sin(th)])
        C = np.array([bx, by]) + ARM_MIN * u + rot(th, self.off)
        ht = th + self.dth
        R = np.array([[np.cos(ht), -np.sin(ht)], [np.sin(ht), np.cos(ht)]])
        loc = [(lx, ly) for lx in np.linspace(-self.hl1, 0, 29) for ly in (0.0, -self.hw)]
        loc += [(lx, ly) for ly in np.linspace(-self.hl2, 0, 12) for lx in (0.0, -self.hw)]
        return C + np.array(loc) @ R.T, C

    def _hook_clear(self, bx, by, th, objs, m=0.05):
        P, _ = self._hook_pts(bx, by, th)
        for o in objs:
            ol, orr, obm, ot = self._extent(o)
            if np.any((P[:, 0] > ol - m) & (P[:, 0] < orr + m) & (P[:, 1] > obm - m) & (P[:, 1] < ot + m)):
                return False
        return True

    def _plan_moves(self, r, h, tb, ob):
        tl, tr, tbm, tt = self._extent(tb)
        w, l2 = h["w"], h["l2"]
        C = self._corner(r)
        cx_L = tl - l2 - w - 0.08
        cx_R = tr + 0.08
        cy_hi = min(tt + w + 0.05, self._cmax() - 0.01)
        cy_s = tbm - 0.04
        moves = []
        cy_now = C[1]
        swept = False
        if (getattr(self, "sweep_dir", None) == "L" and self.n_sweeps == 0
                and self._blocked(tb, ob) and cy_s > WALL_Y + 0.1 and C[0] > cx_L + 0.3):
            self.n_sweeps += 1
            swept = True
            moves.append(("goto", (C[0], cy_s)))
            moves.append(("goto", (cx_L, cy_s)))
        elif C[0] > cx_L + 0.02:
            # retreat left at current height (only when below target)
            moves.append(("goto", (cx_L, min(cy_now, cy_s))))
        if not swept and self._blocked(tb, ob) and cy_s > WALL_Y + 0.1 and self.n_sweeps < 2:
            self.n_sweeps += 1
            moves.append(("goto", (cx_L, cy_s)))
            cx_end = cx_R
            if self.SHORT_SWEEP:
                cx_end = cx_L + 0.1
                for o in ob:
                    ol, orr, obm, ot = self._extent(o)
                    if orr > tl - 0.05 and ol < tr + 0.05 and ot > WALL_Y and o["y"] < tb["y"] and obm < tbm:
                        if ot > cy_s - w + 0.03:
                            cx_end = max(cx_end, tr + 0.06 - l2)
                        else:
                            cx_end = cx_R
                cx_end = min(cx_end, cx_R)
            moves.append(("goto", (cx_end, cy_s)))
            moves.append(("goto", (cx_L, cy_s)))
        moves.append(("goto", (cx_L, cy_hi)))
        moves.append(("traverse", None))
        moves.append(("pull", None))
        self.moves = moves
        self.cmd = C.copy()
        self.move_init = True

    def _corner_act(self, r, Cdes, prefer=1):
        if Cdes[1] < self._cmin() + 0.01:
            prefer = -1
        tgt, ok = self._ik_tilt(np.array(Cdes), max_k=22, prefer=prefer)
        return self._act(r, tgt, -0.02)

    def _run_moves(self, r, h, tb, ob):
        tl, tr, tbm, tt = self._extent(tb)
        w, l2 = h["w"], h["l2"]
        C = self._corner(r)
        if not self.moves:
            self._plan_moves(r, h, tb, ob)
        kind, p = self.moves[0]
        if self.move_init:
            self.move_init = False
            self.trav_goal = None
            self.trav_tx = None
            self.cmd = C.copy()
            self.stall = 0
            self.best_y = C[1]
            self.best_x = C[0]
            self.best_d = 1e9
            self.goto_tx = tb["x"]
        if kind == "goto":
            goal = np.array(p, dtype=float)
            dtx = tb["x"] - self.goto_tx
            if (self.DRAG_ABORT and self.n_abort < 2 and abs(goal[1] - C[1]) < 0.05
                    and abs(dtx) > self.DRAG_ABORT and dtx * (goal[0] - C[0]) > 0):
                # target is being dragged along by the sweep: stop and replan
                self.n_abort += 1
                self.moves = []
                self._plan_moves(r, h, tb, ob)
                return self._corner_act(r, C)
            d = goal - self.cmd
            n = np.linalg.norm(d)
            self.cmd = goal if n < self.V else self.cmd + d / n * self.V
            dist = float(np.linalg.norm(goal - C))
            if dist < 0.02:
                self.moves.pop(0)
                self.move_init = True
            else:
                if dist < getattr(self, "best_d", 1e9) - 0.01:
                    self.best_d = dist
                    self.stall = 0
                else:
                    self.stall += 1
                if self.stall > 20:
                    self.moves.pop(0)
                    self.move_init = True
            return self._corner_act(r, self.cmd)
        if kind == "traverse":
            cy_hi = min(tt + w + 0.05, self._cmax() - 0.01)
            goal = tl - w - 0.04
            for o in ob:
                ol, orr, obm, ot = self._extent(o)
                if ot < WALL_Y + 0.02 or obm > cy_hi - w:
                    continue
                if ol > C[0] + w + 0.02 and ol < goal + w + 0.03:
                    goal = min(goal, ol - w - 0.04)
            goal = max(goal, tb["x"] - l2 + 0.12)
            if self.trav_goal is None:
                self.trav_goal = goal
            goal = min(goal, self.trav_goal)
            cy_c = cy_hi
            if self.trav_tx is None:
                self.trav_tx = tb["x"]
            if tb["x"] - self.trav_tx > self.DRAG:
                self.diag_off = True
            if (self.TDIAG is not None and not self.diag_off and self.n_retrav == 0
                    and C[0] + l2 > tb["x"] + self.TDIAG):
                cy_c = max(min(self.cmd[1], C[1] + 0.06) - self.V, tt + w - 0.1)
            self.cmd = np.array([min(self.cmd[0] + self.V, goal), cy_c])
            if C[0] > self.best_x + 0.01:
                self.best_x = C[0]
                self.stall = 0
            else:
                self.stall += 1
            if self.stall > 25:
                self.stall = 0
                if C[0] + l2 > tl + 0.1:
                    # short bar already overlaps target: pull from here
                    self.moves.pop(0)
                    self.move_init = True
                    self.cx_pull = C[0]
                else:
                    self.moves = []
                    self._plan_moves(r, h, tb, ob)
                return self._corner_act(r, self.cmd)
            if C[0] >= goal - 0.015:
                self.moves.pop(0)
                self.move_init = True
                self.cx_pull = goal
            return self._corner_act(r, self.cmd)
        if kind == "pull":
            self.cmd[1] = max(min(self.cmd[1], C[1] + 0.06) - self.V, 0.5)
            cx = min(self.cx_pull, tl - w - 0.03)
            self.cmd[0] = cx
            if C[1] < self.best_y - 0.01:
                self.best_y = C[1]
                self.stall = 0
            else:
                self.stall += 1
            escaped = tb["x"] > C[0] + l2 + 0.02 or tb["x"] < C[0]
            if escaped and self.stall <= 6 and C[1] - w > tt - 0.02 and self.n_retrav < 2:
                # still above the target: just re-traverse to it
                self.n_retrav += 1
                self.moves = [("traverse", None), ("pull", None)]
                self.cmd = C.copy()
                self.move_init = True
            elif self.stall > 6 or escaped:
                self.moves = []
                self._plan_moves(r, h, tb, ob)
            return self._corner_act(r, self.cmd)
        return self._act(r, (r["x"], r["y"], r["th"], r["arm"]), -0.02)

