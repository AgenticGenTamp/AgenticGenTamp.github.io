"""Approach for StickButton2DEnv.

Robot base drives in a floor region (x in [0.1,3.4], y in [0.1,1.15]) and presses
buttons by touching them (centre distance <= 0.15).  Buttons above y~1.3 are out
of base reach, so the 1.25m stick is grasped from below (robot facing up, arm
extended) and carried vertically above the robot, extending reach to y ~ 2.55.
"""
import itertools
import numpy as np

# --- measured environment constants ---
X_MIN, X_MAX = 0.1008, 3.3992
Y_MIN, Y_MAX = 0.1008, 1.1492
CEIL = 2.50                 # world ceiling: a carried stick may not poke through
BASE_R = 0.10
BUTTON_R = 0.05
PRESS_R = BASE_R + BUTTON_R          # 0.15 base press distance
GRIP_REACH = 0.2128                  # robot centre -> gripper contact face (arm out)
STICK_TOL = 0.05                     # button radius: stick press tolerance
DX_MAX = 0.05
DTH_MAX = 0.19634954
GRASP_OVERHEAD = 3.0                # extra steps for the grasp manoeuvre


def _cheb(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def _cost(p, q):
    return _cheb(p, q) / DX_MAX


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self._init_state()

    def _init_state(self):
        self.carrying = False
        self.grasp_phase = 0
        self.grasp_mode = "below"
        self.grasp_pos = None
        self.grasp_th = np.pi / 2
        self.rot_need = 0.0
        self.grasp_off = None
        self.creep_prev = None
        self.creep_n = 0
        self.last_vac = 0.0
        self.reposition = False
        self.drop_target = (1.75, 0.9)
        self.prev_pos = None
        self.prev_sig = None
        self.prev_act = None
        self.stall = 0
        self.plan = []
        self.bpos = {}
        self.need_stick = set()
        self.task_key = None
        self.task_steps = 0

    # ------------------------------------------------------------------
    # state parsing helpers
    # ------------------------------------------------------------------
    def _parse(self, state):
        robot = None
        buttons = []
        stick = None
        for name in sorted(state.get_object_names()):
            obj = state.get_object_from_name(name)
            feats = state.type_features[obj.type]
            tname = getattr(obj.type, "name", "")
            if tname == "crv_robot" or "arm_joint" in feats:
                robot = obj
            elif "radius" in feats:
                buttons.append(obj)
            elif "height" in feats and "width" in feats:
                if stick is None or name == "stick":
                    stick = obj
        return robot, buttons, stick

    @staticmethod
    def _rf(state, obj, f):
        return float(state.get(obj, f))

    def _pressed(self, state, b):
        return self._rf(state, b, "color_g") > 0.5

    # ------------------------------------------------------------------
    def reset(self, state, info):
        self._init_state()
        try:
            self._build_plan(state)
        except Exception:
            self.plan = []

    # ------------------------------------------------------------------
    # geometry: target poses
    # ------------------------------------------------------------------
    def _base_target(self, cur, b, obstacle, nxt=None):
        """Cheapest robot centre (from cur) that presses button b with the base."""
        bx, by = b
        cands = [(bx, by)]
        for k in range(36):
            a = 2 * np.pi * k / 36
            for rad in (0.11, 0.13):
                cands.append((bx + rad * np.cos(a), by + rad * np.sin(a)))
        # also the axis-clamped point
        cands.append((_clamp(bx, X_MIN, X_MAX), _clamp(by, Y_MIN, Y_MAX)))
        best, bestc = None, 1e18
        ymax = Y_MAX
        xlo, xhi = X_MIN, X_MAX
        if self.carrying:
            ox, oy = self.grasp_off if self.grasp_off else (-0.025, GRIP_REACH)
            ymax = self._carry_ymax(oy)
            xlo, xhi = self._xlim()
        for x, y in cands:
            if not (xlo - 1e-9 <= x <= xhi + 1e-9 and Y_MIN - 1e-9 <= y <= ymax + 1e-9):
                continue
            if np.hypot(x - bx, y - by) > PRESS_R - 0.012:
                continue
            if obstacle is not None:
                sx, sy = obstacle
                if (sx - BASE_R - 0.015 < x < sx + 0.05 + BASE_R + 0.015) and y > sy - BASE_R - 0.015:
                    continue
            c = _cost(cur, (x, y))
            if nxt is not None:
                c += _cost((x, y), nxt)
            if c < bestc:
                bestc, best = c, (x, y)
        if best is not None and nxt is not None:
            bestc = _cost(cur, best)
        return best, bestc

    def _xlim(self):
        if not self.carrying or self.grasp_mode == "below":
            return X_MIN, X_MAX
        ox, oy = self.grasp_off if self.grasp_off else (0.0, 0.0)
        lo, hi = X_MIN, X_MAX
        lo = max(lo, 0.006 - ox)
        hi = min(hi, 3.494 - ox - 0.05)
        if self.grasp_mode == "left":
            hi = min(hi, 3.29)      # extended gripper vs right wall
        else:
            lo = max(lo, 0.21)
        return lo, max(lo, hi)

    def _carry_ymax(self, oy=GRIP_REACH, h=1.25):
        return max(Y_MIN, min(Y_MAX, CEIL - 0.006 - max(oy, 0.0) - h))

    def _stick_target(self, cur, b, ox=-0.025, oy=GRIP_REACH, h=1.25, w=0.05, nxt=None):
        """Cheapest robot centre that presses b with the vertically held stick.

        Stick occupies x in [rx+ox, rx+ox+w], y in [ry+oy, ry+oy+h]; a press needs
        the button centre within STICK_TOL of that rectangle.
        """
        bx, by = b
        tol = STICK_TOL - 0.008
        xlo, xhi = self._xlim()
        ymax = self._carry_ymax(oy)
        # vertical interval assuming no horizontal slack is needed
        ylo_i, yhi_i = by - oy - h, by - oy
        ylo, yhi = max(ylo_i, Y_MIN), min(yhi_i, ymax)
        vgap = 0.0
        if ylo > yhi:
            if ylo_i > ymax:
                vgap = ylo_i - ymax
                ylo = yhi = ymax
            else:
                vgap = Y_MIN - yhi_i
                ylo = yhi = Y_MIN
            if vgap > tol:
                return None, 1e18
        htol = np.sqrt(max(0.0, tol * tol - vgap * vgap))
        xl = max(bx - ox - w - htol, xlo)
        xh = min(bx - ox + htol, xhi)
        if xl > xh:
            # need horizontal slack instead
            cx = _clamp(bx - ox - w / 2.0, xlo, xhi)
            gap = max(0.0, abs(bx - (cx + ox + w / 2.0)) - w / 2.0)
            if gap > tol:
                return None, 1e18
            va = np.sqrt(max(0.0, tol * tol - gap * gap))
            ylo = max(by - oy - h - va, Y_MIN)
            yhi = min(by - oy + va, ymax)
            if ylo > yhi:
                return None, 1e18
            xl = xh = cx
        if nxt is None:
            p = (_clamp(cur[0], xl, xh), _clamp(cur[1], ylo, yhi))
            return p, _cost(cur, p)
        best, bestc = None, 1e18
        for x in np.linspace(xl, xh, 5):
            for y in np.linspace(ylo, yhi, 5):
                c = _cost(cur, (x, y)) + _cost((x, y), nxt)
                if c < bestc:
                    bestc, best = c, (float(x), float(y))
        return best, _cost(cur, best)

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    def _plan_grasp(self, sx, sy, w=0.05):
        """Pick how to grab the stick: from below (preferred) or from a side."""
        gx = _clamp(sx + w / 2.0, X_MIN, X_MAX)
        # gripper (0.07 wide) must overlap the stick footprint [sx, sx+w]
        if gx + 0.03 >= sx and gx - 0.03 <= sx + w:
            return "below", (gx, _clamp(sy - GRIP_REACH, Y_MIN, Y_MAX))
        y_g = _clamp(sy - 0.025, Y_MIN, Y_MAX)
        if sx > 1.75:      # stick hugs the right wall -> grab its left face
            return "left", (_clamp(sx - GRIP_REACH, X_MIN, X_MAX), y_g)
        return "right", (_clamp(sx + w + GRIP_REACH, X_MIN, X_MAX), y_g)

    _GRASP_TH = {"below": np.pi / 2, "left": 0.0, "right": np.pi}

    def _build_plan(self, state):
        robot, buttons, stick = self._parse(state)
        rp = (self._rf(state, robot, "x"), self._rf(state, robot, "y"))
        rem = [b for b in buttons if not self._pressed(state, b)]
        pos = {b.name: (self._rf(state, b, "x"), self._rf(state, b, "y")) for b in rem}
        self.bpos = pos
        sx = sy = None
        if stick is not None:
            sx, sy = self._rf(state, stick, "x"), self._rf(state, stick, "y")
        self.reposition = False
        if stick is not None:
            self.grasp_mode, self.grasp_pos = self._plan_grasp(sx, sy)
        else:
            self.grasp_pos = None
        obstacle = (sx, sy) if stick is not None else None

        rth = self._rf(state, robot, "theta")
        gth = self._GRASP_TH.get(self.grasp_mode, np.pi / 2)
        self.grasp_th = gth
        self.rot_need = abs((gth - rth + np.pi) % (2 * np.pi) - np.pi) / DTH_MAX
        names = list(pos.keys())
        need_stick = set()
        for n in names:
            t, c = self._base_target(rp, pos[n], obstacle)
            if t is None:
                need_stick.add(n)
        self.need_stick = need_stick

        # if the stick must be grabbed from a side, check the resulting hold can
        # reach every button that needs it; otherwise fetch it, drop it in the
        # middle of the arena and re-grab it from below.
        if need_stick and stick is not None and self.grasp_mode != "below" and not self.carrying:
            w = self._rf(state, stick, "width")
            off = ((GRIP_REACH, -0.025) if self.grasp_mode == "left"
                   else (-(GRIP_REACH + w), -0.025))
            save = (self.carrying, self.grasp_off)
            self.carrying, self.grasp_off = True, off
            ok = all(self._stick_target(self.grasp_pos, pos[n], off[0], off[1],
                                        self._rf(state, stick, "height"), w)[0] is not None
                     for n in need_stick)
            self.carrying, self.grasp_off = save
            if not ok:
                self.reposition = True
                xs = sorted(pos[n][0] for n in names)
                xd = _clamp(xs[len(xs) // 2], 0.75, 2.75)
                self.drop_target = (xd, 0.90)
                self.plan = [("grasp", None), ("drop", None)]
                return

        # enumerate orderings (n is small); fall back to greedy for large n
        if len(names) <= 5:
            perms = itertools.permutations(names)
        else:
            perms = [self._greedy_order(names, rp, pos)]
        best_plan, best_cost = None, 1e18
        for perm in perms:
            perm = list(perm)
            first_stick = None
            for i, n in enumerate(perm):
                if n in need_stick:
                    first_stick = i
                    break
            if first_stick is None or self.grasp_pos is None:
                grasp_opts = [None]
            else:
                grasp_opts = list(range(first_stick + 1))
            for gi in grasp_opts:
                cost, plan = self._eval_plan(perm, gi, rp, pos, obstacle)
                if cost < best_cost:
                    best_cost, best_plan = cost, plan
        self.plan = best_plan if best_plan else []

    def _greedy_order(self, names, rp, pos):
        cur, left, order = rp, set(names), []
        while left:
            n = min(left, key=lambda m: _cost(cur, pos[m]))
            order.append(n)
            left.discard(n)
            cur = pos[n]
        return order

    def _eval_plan(self, perm, gi, rp, pos, obstacle):
        """Cost of pressing buttons in order `perm`, grasping the stick before
        index gi (None = never grasp)."""
        carrying_save = self.carrying
        cur = rp
        total = 0.0
        plan = []
        carrying = self.carrying
        for i, n in enumerate(perm):
            if gi is not None and i == gi and not carrying:
                total += _cost(cur, self.grasp_pos) + GRASP_OVERHEAD
                if self.rot_need > total:
                    total = self.rot_need + GRASP_OVERHEAD
                cur = self.grasp_pos
                carrying = True
                plan.append(("grasp", None))
            self.carrying = carrying
            bt, bc = self._base_target(cur, pos[n], None if carrying else obstacle)
            st, sc = (None, 1e18)
            if carrying:
                st, sc = self._stick_target(cur, pos[n])
            if bc <= sc and bt is not None:
                total += bc + self._detour(cur, bt, obstacle, carrying)
                cur = bt
                plan.append(("base", n))
            elif st is not None:
                total += sc
                cur = st
                plan.append(("stick", n))
            else:
                self.carrying = carrying_save
                return 1e18, None
        self.carrying = carrying_save
        return total, plan

    def _detour(self, cur, goal, obstacle, carrying):
        if obstacle is None or carrying:
            return 0.0
        sx, sy = obstacle
        lo = sx - BASE_R - 0.008
        hi = sx + 0.05 + BASE_R + 0.008
        ypass = sy - BASE_R - 0.008
        if (cur[0] < lo and goal[0] > hi) or (cur[0] > hi and goal[0] < lo):
            extra = max(0.0, max(cur[1], goal[1]) - ypass) / DX_MAX
            return extra
        return 0.0

    # ------------------------------------------------------------------
    # low level control
    # ------------------------------------------------------------------
    def _act(self, dx=0.0, dy=0.0, dth=0.0, darm=0.0, vac=None):
        if vac is None:
            vac = 1.0 if self.carrying else 0.0
        self.last_vac = float(vac)
        self.prev_act = np.array([_clamp(dx, -DX_MAX, DX_MAX),
                         _clamp(dy, -DX_MAX, DX_MAX),
                         _clamp(dth, -DTH_MAX, DTH_MAX),
                         _clamp(darm, -0.1, 0.1), vac], dtype=np.float32)
        return self.prev_act

    def _move(self, cur, goal, dth=0.0, darm=0.0):
        """Move toward goal; if actions keep being rejected (collision), the
        environment rejects the WHOLE action, so progressively strip components."""
        xlo, xhi = self._xlim()
        ytop = Y_MAX
        if self.carrying and self.grasp_off is not None:
            ytop = self._carry_ymax(self.grasp_off[1])
        goal = (_clamp(goal[0], xlo, xhi), _clamp(goal[1], Y_MIN, max(Y_MIN, ytop)))
        dx = goal[0] - cur[0]
        dy = goal[1] - cur[1]
        s = self.stall
        if s <= 0:
            return self._act(dx, dy, dth, darm)
        opts = [(dx, dy, 0.0, 0.0),
                (0.0, 0.0, dth, 0.0),
                (dx, 0.0, 0.0, 0.0),
                (0.0, dy, 0.0, 0.0),
                (0.0, 0.0, dth, -0.1),
                (0.0, 0.0, DTH_MAX, -0.1),
                (0.0, -DX_MAX, 0.0, -0.1),
                (-DX_MAX if dx <= 0 else DX_MAX, -DX_MAX, 0.0, -0.1),
                (0.0, DX_MAX, 0.0, -0.1),
                (dx, dy, dth, darm)]
        n = len(opts)
        for k in range(n):
            o = opts[(s - 1 + k) % n]
            if any(abs(z) > 1e-9 for z in o):
                return self._act(*o)
        return self._act(0.0, 0.0, DTH_MAX, -0.1)

    def _route(self, cur, goal, obstacle):
        """Waypoint around the standing stick (it blocks the floor band)."""
        if obstacle is None:
            return goal
        sx, sy = obstacle
        lo = sx - BASE_R - 0.008
        hi = sx + 0.05 + BASE_R + 0.008
        ypass = sy - BASE_R - 0.008
        if ypass < Y_MIN + 1e-6:
            return goal
        inband = lo < cur[0] < hi
        if cur[1] > ypass + 1e-6:
            if inband:
                return (cur[0], ypass)                    # drop straight down
            if cur[0] <= lo:
                if goal[0] <= lo:
                    return goal
                return (min(goal[0], lo - 0.004), ypass)  # descend while closing in
            if goal[0] >= hi:
                return goal
            return (max(goal[0], hi + 0.004), ypass)
        # below the stick: free to slide across, but stay low while crossing
        if (cur[0] <= lo and goal[0] > lo) or (cur[0] >= hi and goal[0] < hi):
            return (goal[0], min(goal[1], ypass))
        return goal

    # ------------------------------------------------------------------
    def get_action(self, state):
        try:
            a = self._get_action(state)
            a = np.asarray(a, dtype=np.float32).reshape(5)
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            lo = np.asarray(self.action_space.low, dtype=np.float32)
            hi = np.asarray(self.action_space.high, dtype=np.float32)
            return np.clip(a, lo, hi).astype(np.float32)
        except Exception:
            v = 1.0 if getattr(self, "carrying", False) else 0.0
            return np.array([0.0, 0.0, 0.0, 0.0, v], dtype=np.float32)

    def _get_action(self, state):
        robot, buttons, stick = self._parse(state)
        if robot is None:
            return self._act()
        rx, ry = self._rf(state, robot, "x"), self._rf(state, robot, "y")
        rth = self._rf(state, robot, "theta")
        arm = self._rf(state, robot, "arm_joint")
        cur = (rx, ry)
        # a stall means: we commanded a change but nothing in the robot moved
        sig = (rx, ry, rth, arm)
        if self.prev_sig is not None and self.prev_act is not None:
            wanted = (abs(self.prev_act[0]) > 1e-9 or abs(self.prev_act[1]) > 1e-9
                      or abs(self.prev_act[2]) > 1e-9 or abs(self.prev_act[3]) > 1e-9)
            same = all(abs(a - b) < 1e-9 for a, b in zip(sig, self.prev_sig))
            if (wanted and same) or (not wanted and self.stall > 0):
                self.stall += 1
            else:
                self.stall = 0
        self.prev_sig = sig
        self.prev_pos = cur

        if self.carrying and self.grasp_phase >= 5:
            self._check_grasp(state, stick, cur)

        byname = {b.name: b for b in buttons}
        obstacle = None
        if stick is not None and not self.carrying:
            obstacle = (self._rf(state, stick, "x"), self._rf(state, stick, "y"))

        # prune finished / impossible tasks
        newplan = []
        for kind, n in self.plan:
            if kind in ("base", "stick"):
                b = byname.get(n)
                if b is None or self._pressed(state, b):
                    continue
            elif kind == "grasp" and self.carrying:
                continue
            newplan.append((kind, n))
        self.plan = newplan
        # drop a leading grasp if nothing left needs the stick
        while self.plan and self.plan[0][0] == "grasp":
            if any(k in ("stick", "drop") for k, _ in self.plan[1:]):
                break
            self.plan.pop(0)

        rem = [b for b in buttons if not self._pressed(state, b)]
        if not rem:
            return self._act()
        if not self.plan:
            self._build_plan(state)
            if not self.plan:
                b = min(rem, key=lambda b: _cost(cur, (self._rf(state, b, "x"),
                                                       self._rf(state, b, "y"))))
                self.plan = [("base", b.name)]

        kind, name = self.plan[0]
        if kind == "stick" and not self.carrying:
            self.plan.insert(0, ("grasp", None))
            self.grasp_phase = 0
            kind, name = self.plan[0]

        if kind == "drop":
            if not self.carrying or stick is None:
                self.plan.pop(0)
                return self.get_action(state)
            ox = self._rf(state, stick, "x") - cur[0]
            oy = self._rf(state, stick, "y") - cur[1]
            xlo, xhi = self._xlim()
            tgt = (_clamp(self.drop_target[0] - ox, xlo, xhi),
                   _clamp(self.drop_target[1] - oy, Y_MIN, Y_MAX))
            if _cheb(cur, tgt) < 3e-3:
                self.plan.pop(0)
                self.carrying = False
                self.grasp_off = None
                self.grasp_phase = 0
                self.creep_prev = None
                self.creep_n = 0
                self.grasp_mode = "below"
                return self._act(0.0, 0.0, 0.0, 0.0, 0.0)
            return self._move(cur, tgt)

        if kind == "grasp":
            return self._grasp_action(state, stick, cur, rth, arm, obstacle)

        b = byname[name]
        bpt = (self._rf(state, b, "x"), self._rf(state, b, "y"))
        nxt = None
        if len(self.plan) > 1:
            k2, n2 = self.plan[1]
            if k2 == "grasp" and stick is not None:
                m, gp = self._plan_grasp(self._rf(state, stick, "x"),
                                         self._rf(state, stick, "y"),
                                         self._rf(state, stick, "width"))
                nxt = gp
            elif n2 in byname:
                nxt = (self._rf(state, byname[n2], "x"), self._rf(state, byname[n2], "y"))
        if kind == "stick" and self.carrying:
            goal, c = self._stick_target(cur, bpt, *self._offsets(state, stick, cur), nxt=nxt)
            if goal is None:
                goal, c = self._base_target(cur, bpt, obstacle, nxt)
        else:
            goal, c = self._base_target(cur, bpt, obstacle, nxt)
            if goal is None and self.carrying:
                goal, c = self._stick_target(cur, bpt, *self._offsets(state, stick, cur), nxt=nxt)
        if goal is None:
            if len(self.plan) > 1:
                self.plan.append(self.plan.pop(0))
                return self._get_action(state)
            goal = (_clamp(bpt[0], X_MIN, X_MAX), _clamp(bpt[1], Y_MIN, Y_MAX))
        goal = self._route(cur, goal, obstacle)
        darm = 0.0
        if not self.carrying and arm > 0.1 + 1e-6:
            darm = -0.1
        dth = 0.0
        if not self.carrying:
            if any(k == "grasp" for k, _ in self.plan):
                tgt_th = self.grasp_th
            else:
                tgt_th = self._park_theta(goal)
            if tgt_th is not None:
                dth = (tgt_th - rth + np.pi) % (2 * np.pi) - np.pi
        return self._move(cur, goal, dth, darm)

    @staticmethod
    def _park_theta(goal):
        """Orientation that keeps the protruding gripper off the nearby wall."""
        vx = vy = 0.0
        if goal[0] < 0.30:
            vx = 1.0
        elif goal[0] > 3.20:
            vx = -1.0
        if goal[1] < 0.30:
            vy = 1.0
        elif goal[1] > 0.95:
            vy = -1.0
        if vx == 0.0 and vy == 0.0:
            return None
        return float(np.arctan2(vy, vx))

    def _offsets(self, state, stick, cur):
        ox = self._rf(state, stick, "x") - cur[0]
        oy = self._rf(state, stick, "y") - cur[1]
        h = self._rf(state, stick, "height")
        w = self._rf(state, stick, "width")
        return ox, oy, h, w

    # ------------------------------------------------------------------
    def _grasp_action(self, state, stick, cur, rth, arm, obstacle):
        if stick is None:
            self.plan = [t for t in self.plan if t[0] != "grasp"]
            return self._act()
        sx = self._rf(state, stick, "x")
        sy = self._rf(state, stick, "y")
        w = self._rf(state, stick, "width")
        mode = self.grasp_mode
        step = 0.013
        if mode == "below":
            gx = _clamp(sx + w / 2.0, X_MIN, X_MAX)
            gy = sy - GRIP_REACH
            goal_th = np.pi / 2
            creep = (0.0, step)
            safe = cur[1] <= gy + 1e-3
        elif mode == "left":
            gx = _clamp(sx - GRIP_REACH, X_MIN, X_MAX)
            gy = _clamp(sy - 0.025, Y_MIN, Y_MAX)
            goal_th = 0.0
            creep = (step, 0.0)
            safe = cur[0] <= gx + 1e-3
        else:
            gx = _clamp(sx + w + GRIP_REACH, X_MIN, X_MAX)
            gy = _clamp(sy - 0.025, Y_MIN, Y_MAX)
            goal_th = np.pi
            creep = (-step, 0.0)
            safe = cur[0] >= gx - 1e-3
        stage = (_clamp(gx - creep[0], X_MIN, X_MAX), _clamp(gy - creep[1], Y_MIN, Y_MAX))
        dth = (goal_th - rth + np.pi) % (2 * np.pi) - np.pi
        idx = 0 if creep[0] != 0.0 else 1

        if self.grasp_phase == 0:
            darm = 0.0
            if safe and arm < 0.2 - 1e-6:
                darm = 0.1
            elif not safe and arm > 0.1 + 1e-6:
                darm = -0.1
            goal = stage
            if mode == "below" and cur[1] > stage[1] + 1e-3 and abs(cur[0] - gx) > 0.02:
                goal = (cur[0], stage[1])       # descend clear of the stick first
            goal = self._route(cur, goal, obstacle)
            if (abs(dth) > 0.02 and abs(cur[0] - stage[0]) < 0.25
                    and abs(cur[1] - stage[1]) < 0.25):
                return self._act(0.0, 0.0, dth, darm, 0.0)   # turn before closing in
            if (abs(cur[0] - stage[0]) < 2.5e-3 and abs(cur[1] - stage[1]) < 2.5e-3
                    and abs(dth) < 1e-3 and arm > 0.2 - 1e-6):
                self.grasp_phase = 2
                self.creep_prev = None
            else:
                return self._move(cur, goal, dth, darm)

        if self.grasp_phase == 2:
            blocked = (self.creep_prev is not None
                       and abs(cur[idx] - self.creep_prev) < 1e-7)
            if self.creep_n > 0 or blocked:
                self.grasp_phase = 3
            else:
                self.creep_prev = cur[idx]
                self.creep_n += 1
                return self._act(creep[0], creep[1], 0.0, 0.0, 1.0)

        if self.grasp_phase == 3:
            if self.last_vac < 0.5:
                return self._act(0.0, 0.0, 0.0, 0.0, 1.0)

        self.carrying = True
        self.grasp_off = (sx - cur[0], sy - cur[1])
        self.grasp_phase = 5
        return self.get_action(state)

    def _check_grasp(self, state, stick, cur):
        """Detect a failed / lost grasp and fall back to re-grasping."""
        if not self.carrying or stick is None or self.grasp_off is None:
            return True
        off = (self._rf(state, stick, "x") - cur[0], self._rf(state, stick, "y") - cur[1])
        if abs(off[0] - self.grasp_off[0]) > 0.004 or abs(off[1] - self.grasp_off[1]) > 0.004:
            self.carrying = False
            self.grasp_phase = 2
            self.creep_prev = None
            self.creep_n = 0
            self.grasp_off = None
            self.plan = [("grasp", None)] + [t for t in self.plan if t[0] != "grasp"]
            return False
        return True
