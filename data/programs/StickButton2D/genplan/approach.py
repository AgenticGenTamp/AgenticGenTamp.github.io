"""Policy for StickButton2DEnv (variable object count).

HONEST STATUS
-------------
Verified working: the base-ram phase presses floor buttons (button2 on seed 0
turns green every run).

NOT working, cause unknown: the stick grasp.  Five attempts at an analytic
"servo the pad onto the stick" controller have all ended with the robot parked
near the stick, vacuum on, arm extended, stick unmoved.  I cannot execute this
environment, so my previous geometric explanations were speculation presented
as diagnosis.  Competing hypotheses I cannot currently separate:
  (a) my suction-pad position model is wrong, so the pad never overlaps;
  (b) collisions reject the steps that would reach a valid grasp pose;
  (c) _grabbed latches but _verify immediately clears it;
  (d) my RectangleType corner-vs-center convention is wrong, misplacing the
      stick by ~2.5cm.

Design response:
  1. Do all base-reachable buttons FIRST, unconditionally.  Instances whose
     buttons are all low then succeed regardless of the grasp bug.
  2. Replace the exact-pose grasp servo with a coarse SWEEP over base positions
     and headings near the stick's low end, vacuum on throughout.  A sweep is
     robust to a systematically wrong pad model; an exact servo is not.
  3. Record diagnostics on self.diag so the next run yields evidence rather
     than another guess.

Seed 0 specifically cannot be solved without the stick: button0 (y=1.80) and
button1 (y=2.29) are both beyond base reach (~y<=1.30).
"""

from __future__ import annotations

import math

import numpy as np


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def _seg_pt(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    ab = b - a
    d = float(ab @ ab)
    if d < 1e-12:
        return float(np.hypot(*(p - a)))
    t = float((p - a) @ ab) / d
    t = min(1.0, max(0.0, t))
    return float(np.hypot(*(p - (a + t * ab))))


class _Pose:
    __slots__ = ("x", "y", "theta")

    def __init__(self, x, y, theta):
        self.x = float(x); self.y = float(y); self.theta = float(theta)

    def __mul__(self, o):
        r = _rot(self.theta) @ np.array([o.x, o.y])
        return _Pose(self.x + r[0], self.y + r[1], _wrap(self.theta + o.theta))

    def inverse(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return _Pose(-self.x * c - self.y * s, self.x * s - self.y * c, -self.theta)


class GeneratedApproach:
    PRESSED_G = 0.5
    WMAXX = 3.5
    TABLE_Y = 1.25
    WALL = 0.10

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives
        lo = np.asarray(action_space.low, dtype=np.float64)
        hi = np.asarray(action_space.high, dtype=np.float64)
        self.lo, self.hi = lo, hi
        self.mdx = float(min(abs(lo[0]), hi[0]))
        self.mdy = float(min(abs(lo[1]), hi[1]))
        self.mdt = float(min(abs(lo[2]), hi[2]))
        self.mda = float(min(abs(lo[3]), hi[3]))
        self.rng = np.random.default_rng(5)
        self._init_ep()

    # ---------------------------------------------------------------- setup

    def _init_ep(self):
        self.t = 0
        self.holding = False
        self.g2s = None
        self.sdims = None
        self.slen = 1.25
        self.sthk = 0.05
        self.prev = None
        self.stuck = 0
        self.escape = 0
        self.eact = None
        self.target = None
        self.tage = 0
        self.fail = {}
        self.mode = "ram"
        self.base_r = 0.1
        self.arm_len = 0.2
        self.gw = 0.01
        self.gh = 0.07
        # sweep state
        self.sw_i = 0
        self.sw_hold = 0
        self.sw_goal = None
        self.sw_th = 0.0
        self.grasp_t = 0
        self.poke_t = 0
        self._set_bounds()
        # diagnostics -- report these after a run
        self.diag = {
            "min_pad_gap": 9.9,
            "grabbed_fired": False,
            "verify_unlatched": 0,
            "rejected_steps": 0,
            "grasp_steps": 0,
            "ram_pressed": 0,
            "reachable_at_reset": None,
        }

    def _set_bounds(self):
        self.ymax = self.TABLE_Y - self.base_r - 0.010
        self.xmin = self.WALL + self.base_r + 0.010
        self.xmax = self.WMAXX - self.WALL - self.base_r - 0.010
        self.ymin = self.WALL + self.base_r + 0.010

    def reset(self, state, info):
        self._init_ep()
        r = self._robot(state)
        if r is not None:
            self.base_r = float(state.get(r, "base_radius"))
            self.arm_len = float(state.get(r, "arm_length"))
            self.gw = float(state.get(r, "gripper_width"))
            self.gh = float(state.get(r, "gripper_height"))
            self._set_bounds()
        try:
            self.diag["reachable_at_reset"] = sum(
                1 for b in self._buttons(state) if self._ramable(state, b))
        except Exception:
            pass

    # -------------------------------------------------------------- parsing

    def _robot(self, s):
        for n in s.get_object_names():
            o = s.get_object_from_name(n)
            if o.type.name == "crv_robot":
                return o
        return None

    def _stick(self, s):
        try:
            o = s.get_object_from_name("stick")
            if o is not None:
                return o
        except Exception:
            pass
        for n in s.get_object_names():
            o = s.get_object_from_name(n)
            if o.type.name == "rectangle":
                try:
                    if float(s.get(o, "static")) < 0.5:
                        return o
                except Exception:
                    continue
        return None

    def _buttons(self, s):
        out = []
        for n in sorted(s.get_object_names()):
            if n.startswith("button"):
                o = s.get_object_from_name(n)
                if o.type.name == "circle":
                    out.append(o)
        return out

    def _pressed(self, s, b):
        try:
            return (float(s.get(b, "color_g")) > self.PRESSED_G
                    and float(s.get(b, "color_r")) < 0.5)
        except Exception:
            return False

    def _todo(self, s):
        return [b for b in self._buttons(s) if not self._pressed(s, b)]

    def _xy(self, s, o):
        return np.array([float(s.get(o, "x")), float(s.get(o, "y"))])

    # ----------------------------------------------------------- kinematics

    def _pad_at(self, xy, th, arm):
        d = arm + self.gw + self.gw / 2.0
        return np.array([xy[0] + math.cos(th) * d, xy[1] + math.sin(th) * d])

    def _pad(self, s, r):
        return self._pad_at(np.array([float(s.get(r, "x")), float(s.get(r, "y"))]),
                            float(s.get(r, "theta")),
                            float(s.get(r, "arm_joint")))

    def _tip(self, s, r):
        x = float(s.get(r, "x")); y = float(s.get(r, "y"))
        th = float(s.get(r, "theta")); arm = float(s.get(r, "arm_joint"))
        d = arm + self.gw / 2.0
        return _Pose(x + math.cos(th) * d, y + math.sin(th) * d, th)

    @staticmethod
    def _sseg(s, st):
        # RectangleType: (x, y) is the CORNER; extents width x height rotated
        # by theta.  If this convention is wrong it is hypothesis (d).
        x = float(s.get(st, "x")); y = float(s.get(st, "y"))
        th = float(s.get(st, "theta"))
        w = float(s.get(st, "width")); h = float(s.get(st, "height"))
        R = _rot(th)
        c = np.array([x, y]) + R @ np.array([w / 2.0, h / 2.0])
        half = R @ (np.array([0.0, h / 2.0]) if h >= w else np.array([w / 2.0, 0.0]))
        return c - half, c + half, w, h

    def _pred(self, s, r, xy, th, arm=None):
        if self.g2s is None:
            return None
        if arm is None:
            arm = float(s.get(r, "arm_joint"))
        d = arm + self.gw / 2.0
        tip = _Pose(xy[0] + math.cos(th) * d, xy[1] + math.sin(th) * d, th)
        w0 = tip * self.g2s
        w, h = self.sdims if self.sdims else (0.05, 1.25)
        R = _rot(w0.theta)
        c = np.array([w0.x, w0.y]) + R @ np.array([w / 2.0, h / 2.0])
        half = R @ (np.array([0.0, h / 2.0]) if h >= w else np.array([w / 2.0, 0.0]))
        return c - half, c + half

    # --------------------------------------------------------------- action

    def _act(self, dx, dy, dth, da, vac):
        a = np.array([dx, dy, dth, da, vac], dtype=np.float64)
        a = np.minimum(np.maximum(a, self.lo), self.hi)
        return a.astype(np.float32)

    def _inb(self, p):
        return (self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax)

    def _clampxy(self, p):
        return np.array([float(np.clip(p[0], self.xmin, self.xmax)),
                         float(np.clip(p[1], self.ymin, self.ymax))])

    def _limit(self, cur, dx, dy):
        nx, ny = cur[0] + dx, cur[1] + dy
        if nx < self.xmin:
            dx = self.xmin - cur[0]
        elif nx > self.xmax:
            dx = self.xmax - cur[0]
        if ny < self.ymin:
            dy = self.ymin - cur[1]
        elif ny > self.ymax:
            dy = self.ymax - cur[1]
        return dx, dy

    def _go(self, cur, goal, scale=1.0):
        d = goal - cur
        n = float(np.hypot(*d))
        if n < 1e-9:
            return 0.0, 0.0
        step = min(n, math.hypot(self.mdx, self.mdy) * scale)
        v = d / n * step
        return self._limit(cur, float(np.clip(v[0], -self.mdx, self.mdx)),
                           float(np.clip(v[1], -self.mdy, self.mdy)))

    def _turn(self, cur, goal, scale=1.0):
        return float(np.clip(_wrap(goal - cur), -self.mdt * scale, self.mdt * scale))

    # ---------------------------------------------------------------- stuck

    def _track(self, s, r):
        p = (float(s.get(r, "x")), float(s.get(r, "y")), float(s.get(r, "theta")))
        if self.prev is not None:
            m = (abs(p[0] - self.prev[0]) + abs(p[1] - self.prev[1])
                 + abs(_wrap(p[2] - self.prev[2])))
            if m < 1e-5:
                self.stuck += 1
                self.diag["rejected_steps"] += 1
            else:
                self.stuck = 0
        self.prev = p

    def _mk_escape(self, s, r):
        th = float(s.get(r, "theta"))
        cur = np.array([float(s.get(r, "x")), float(s.get(r, "y"))])
        vac = 1.0 if (self.holding or self.mode == "fetch") else 0.0
        c = int(self.rng.integers(0, 4))
        if c == 0:
            dx, dy, dt = -math.cos(th) * self.mdx, -math.sin(th) * self.mdy, 0.0
        elif c == 1:
            sg = 1.0 if self.rng.random() < 0.5 else -1.0
            dx = -math.sin(th) * self.mdx * sg
            dy = math.cos(th) * self.mdy * sg
            dt = 0.0
        elif c == 2:
            dx = dy = 0.0
            dt = self.mdt * (1.0 if self.rng.random() < 0.5 else -1.0)
        else:
            dx = self.rng.uniform(-1, 1) * self.mdx
            dy = self.rng.uniform(-1, 1) * self.mdy
            dt = self.rng.uniform(-1, 1) * self.mdt
        mid = (self.xmin + self.xmax) / 2.0
        dx += 0.35 * self.mdx * (1.0 if cur[0] < mid else -1.0)
        dy -= 0.25 * self.mdy
        dx, dy = self._limit(cur, dx, dy)
        return self._act(dx, dy, dt, 0.0, vac)

    # ----------------------------------------------------------------- main

    def get_action(self, state):
        self.t += 1
        r = self._robot(state)
        if r is None:
            return self._act(0, 0, 0, 0, 0)
        self.base_r = float(state.get(r, "base_radius"))
        self.arm_len = float(state.get(r, "arm_length"))
        self.gw = float(state.get(r, "gripper_width"))
        self.gh = float(state.get(r, "gripper_height"))
        self._set_bounds()
        st = self._stick(state)

        if st is not None and not self.holding:
            gap = _seg_pt(*self._sseg(state, st)[:2], self._pad(state, r))
            if gap < self.diag["min_pad_gap"]:
                self.diag["min_pad_gap"] = gap
            if self._grabbed(state, r, st):
                self.diag["grabbed_fired"] = True
                self.holding = True
                tip = self._tip(state, r)
                w0 = _Pose(float(state.get(st, "x")), float(state.get(st, "y")),
                           float(state.get(st, "theta")))
                self.g2s = tip.inverse() * w0
                w = float(state.get(st, "width")); h = float(state.get(st, "height"))
                self.sdims = (w, h); self.slen = max(w, h); self.sthk = min(w, h)
                self.mode = "poke"
                self.target = None
                self.poke_t = 0

        self._track(state, r)

        if self.escape > 0:
            self.escape -= 1
            return self.eact
        if self.stuck >= 6:
            self.stuck = 0
            self.escape = 8
            self.eact = self._mk_escape(state, r)
            return self.eact

        if self.holding and st is not None:
            self._verify(state, r, st)

        todo = self._todo(state)
        if not todo:
            return self._act(0, 0, 0, 0, 1.0 if self.holding else 0.0)

        cur = np.array([float(state.get(r, "x")), float(state.get(r, "y"))])

        if self.holding:
            return self._poke(state, r, st, todo)

        # Phase 1: every base-reachable button, unconditionally, first.
        ram = [b for b in todo if self._ramable(state, b)]
        if ram:
            return self._ram(state, r, ram)

        if st is None:
            return self._ram(state, r, todo)

        self.mode = "fetch"
        self.grasp_t += 1
        self.diag["grasp_steps"] = self.grasp_t
        return self._sweep_grasp(state, r, st)

    # ------------------------------------------------------------ ram phase

    def _ramable(self, s, b):
        p = self._xy(s, b)
        rad = float(s.get(b, "radius"))
        return p[1] <= self.ymax + self.base_r + rad - 0.004

    def _ram(self, s, r, cands):
        rx = float(s.get(r, "x")); ry = float(s.get(r, "y"))
        rth = float(s.get(r, "theta"))
        arm = float(s.get(r, "arm_joint"))
        cur = np.array([rx, ry])

        tgt = self._pick(s, cur, cands)
        if tgt is None:
            return self._act(0, 0, 0, -self.mda, 0.0)
        p = self._xy(s, tgt)
        rad = float(s.get(tgt, "radius"))
        d = float(np.hypot(*(p - cur)))

        if d <= self.base_r + rad + 0.004:
            self.fail[tgt.name] = self.fail.get(tgt.name, 0) + 1
            j = self.rng.uniform(-1, 1, 2)
            dx, dy = self._limit(cur, j[0] * self.mdx * 0.6, j[1] * self.mdy * 0.6)
            return self._act(dx, dy, 0.0, 0.0, 0.0)

        goal = self._clampxy(p if p[1] <= self.ymax else np.array([p[0], self.ymax]))
        dx, dy = self._go(cur, goal)
        dth = self._turn(rth, math.atan2(p[1] - ry, p[0] - rx), 0.4)
        da = -self.mda if arm > self.base_r + 1e-6 else 0.0
        return self._act(dx, dy, dth, da, 0.0)

    def _pick(self, s, cur, cands):
        if not cands:
            return None
        if self.target is not None and self.tage < 160:
            for b in cands:
                if b.name == self.target:
                    self.tage += 1
                    return b
        best, bd = None, float("inf")
        for b in cands:
            p = self._xy(s, b)
            d = float(np.hypot(*(p - cur))) + 4.0 * self.fail.get(b.name, 0)
            if d < bd:
                bd, best = d, b
        self.target = best.name if best else None
        self.tage = 0
        return best

    # ---------------------------------------------------------- grasp phase

    def _grabbed(self, s, r, st):
        if float(s.get(r, "vacuum")) <= 0.5:
            return False
        pad = self._pad(s, r)
        a, b, w, h = self._sseg(s, st)
        tol = min(w, h) / 2.0 + max(self.gh, self.gw) / 2.0 + 0.035
        return _seg_pt(a, b, pad) <= tol

    def _verify(self, s, r, st):
        # Deliberately permissive: an over-eager un-latch was hypothesis (c).
        tip = self._tip(s, r)
        w0 = _Pose(float(s.get(st, "x")), float(s.get(st, "y")),
                   float(s.get(st, "theta")))
        rel = tip.inverse() * w0
        if self.g2s is None:
            self.g2s = rel
            return
        a, b, _, _ = self._sseg(s, st)
        if _seg_pt(a, b, np.array([tip.x, tip.y])) > 0.30:
            self.diag["verify_unlatched"] += 1
            self.holding = False
            self.g2s = None
            self.mode = "fetch"
        else:
            self.g2s = rel

    def _sweep_cells(self, s, st):
        """Coarse grid of (base_pos, heading) probes around the stick's low end.

        A sweep, not a computed target: robust to a systematically wrong pad
        model, which an exact servo is not.
        """
        a, b, w, h = self._sseg(s, st)
        self.sdims = (w, h); self.slen = max(w, h); self.sthk = min(w, h)
        low = a if a[1] <= b[1] else b
        high = b if a[1] <= b[1] else a
        ax = high - low
        n = float(np.hypot(*ax))
        axn = ax / max(n, 1e-9)

        cells = []
        # radii bracket the nominal pad distance generously
        nominal = self.arm_len + 1.5 * self.gw
        radii = [nominal - 0.06, nominal - 0.03, nominal, nominal + 0.03]
        # grasp points along the exposed lower part of the stick
        for frac in (0.02, 0.08, 0.15, 0.24, 0.34):
            gp = low + axn * (n * frac)
            if gp[1] > self.TABLE_Y - 0.01 and abs(axn[1]) > 1e-6:
                continue
            for angi in range(16):
                th = -math.pi + angi * (2 * math.pi / 16)
                u = np.array([math.cos(th), math.sin(th)])
                for rr in radii:
                    base = gp - u * rr
                    if not self._inb(base):
                        continue
                    if _seg_pt(a, b, base) < self.base_r + self.sthk / 2.0 + 0.004:
                        continue
                    cells.append((base, th))
        return cells

    def _sweep_grasp(self, s, r, st):
        rx = float(s.get(r, "x")); ry = float(s.get(r, "y"))
        rth = float(s.get(r, "theta"))
        arm = float(s.get(r, "arm_joint"))
        cur = np.array([rx, ry])

        cells = self._sweep_cells(s, st)
        if not cells:
            j = self.rng.uniform(-1, 1, 2)
            dx, dy = self._limit(cur, j[0] * self.mdx, j[1] * self.mdy)
            return self._act(dx, dy, self.rng.uniform(-1, 1) * self.mdt,
                             self.mda, 1.0)

        # Order cells by distance from the start so the sweep is contiguous.
        if self.sw_goal is None:
            cells.sort(key=lambda c: np.hypot(*(c[0] - cur)))
            self._cells = cells
            self.sw_i = 0
            self.sw_goal, self.sw_th = cells[0]

        goal, want_th = self.sw_goal, self.sw_th
        pos_err = float(np.hypot(*(goal - cur)))
        ang_err = abs(_wrap(want_th - rth))

        # Vacuum on and arm out for the entire sweep so contact latches
        # the instant it occurs, whatever the true pad geometry is.
        da = self.mda if arm < self.arm_len - 1e-6 else 0.0

        arrived = pos_err < 0.02 and ang_err < 0.06
        if arrived:
            self.sw_hold += 1
        if (self.sw_hold > 4) or (self.grasp_t % 26 == 0 and self.grasp_t > 0):
            # advance the sweep
            self.sw_hold = 0
            self.sw_i += 1
            pool = getattr(self, "_cells", cells)
            self.sw_goal, self.sw_th = pool[self.sw_i % len(pool)]
            goal, want_th = self.sw_goal, self.sw_th
            pos_err = float(np.hypot(*(goal - cur)))
            ang_err = abs(_wrap(want_th - rth))

        if pos_err > 0.02:
            dx, dy = self._go(cur, goal)
            dth = self._turn(rth, want_th)
            return self._act(dx, dy, dth, da, 1.0)

        dth = self._turn(rth, want_th)
        # small dither while parked, to cross nearby contact configurations
        j = self.rng.uniform(-1, 1, 2) * 0.35
        dx, dy = self._limit(cur, j[0] * self.mdx, j[1] * self.mdy)
        return self._act(dx, dy, dth, da, 1.0)

    # ----------------------------------------------------------- poke phase

    def _poke(self, s, r, st, todo):
        self.poke_t += 1
        rx = float(s.get(r, "x")); ry = float(s.get(r, "y"))
        rth = float(s.get(r, "theta"))
        arm = float(s.get(r, "arm_joint"))
        cur = np.array([rx, ry])

        tgt = self._pick(s, cur, todo)
        if tgt is None:
            return self._act(0, 0, 0, 0, 1.0)
        p = self._xy(s, tgt)
        rad = float(s.get(tgt, "radius"))
        contact = rad + self.sthk / 2.0
        da = self.mda if arm < self.arm_len - 1e-6 else 0.0

        seg = self._sseg(s, st)[:2] if st is not None else None
        gap = _seg_pt(seg[0], seg[1], p) if seg is not None else 9.9

        if gap <= contact + 0.003:
            d = p - cur
            n = float(np.hypot(*d))
            dx = dy = 0.0
            if n > 1e-9:
                dx, dy = self._go(cur, cur + d / n * 0.05, 0.4)
            return self._act(dx, dy, 0.0, da, 1.0)

        near = self.arm_len + self.gw
        far = near + self.slen
        dvec = p - cur
        dn = float(np.hypot(*dvec))

        if dn > far - 0.06 or dn < near - 0.02:
            want = min(max(0.55 * (near + far), near + 0.06), far - 0.12)
            u = dvec / max(dn, 1e-9)
            goal = p - u * want
            if not self._inb(goal):
                yb = min(self.ymax, p[1])
                rr = want * want - (p[1] - yb) ** 2
                cl = []
                if rr > 1e-6:
                    off = math.sqrt(rr)
                    for c in (np.array([p[0] - off, yb]),
                              np.array([p[0] + off, yb])):
                        if self._inb(c):
                            cl.append(c)
                goal = (min(cl, key=lambda c: np.hypot(*(c - cur)))
                        if cl else self._clampxy(np.array([p[0], yb])))
            goal = self._clampxy(goal)
            dx, dy = self._go(cur, goal)
            dth = self._turn(rth, math.atan2(p[1] - goal[1], p[0] - goal[0]))
            return self._act(dx, dy, dth, da, 1.0)

        best, best_s = None, gap + 1e-9
        for dth_c in (0.0, self.mdt, -self.mdt, self.mdt * 0.5, -self.mdt * 0.5,
                      self.mdt * 0.2, -self.mdt * 0.2):
            nth = _wrap(rth + dth_c)
            for angi in range(24):
                ang = -math.pi + angi * (2 * math.pi / 24)
                for sc in (1.0, 0.4):
                    mv = np.array([math.cos(ang) * self.mdx * sc,
                                   math.sin(ang) * self.mdy * sc])
                    nxy = cur + mv
                    if not self._inb(nxy):
                        continue
                    pr = self._pred(s, r, nxy, nth)
                    if pr is None:
                        continue
                    g = _seg_pt(pr[0], pr[1], p)
                    if g < best_s:
                        best_s, best = g, (mv, dth_c)
        if best is not None:
            mv, dt = best
            dx, dy = self._limit(cur, mv[0], mv[1])
            return self._act(dx, dy, dt, da, 1.0)

        if self.poke_t % 80 == 0:
            self.fail[tgt.name] = self.fail.get(tgt.name, 0) + 1
            self.target = None
        j = self.rng.uniform(-1, 1, 2)
        dx, dy = self._limit(cur, j[0] * self.mdx * 0.6, j[1] * self.mdy * 0.6)
        return self._act(dx, dy, self.mdt * 0.8, da, 1.0)