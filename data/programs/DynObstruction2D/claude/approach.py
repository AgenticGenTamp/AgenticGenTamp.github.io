"""Policy for DynObstruction2DEnv (variable object count).

Key facts discovered empirically about this environment:

* Termination is purely geometric and is evaluated every step: the two
  BOTTOM vertices of the target block must lie inside the target surface's
  x-range and be at/below the surface top (y ~ 0.1).  It fires even while
  the block is still HELD by the gripper.
* A held object is rigidly welded to the (kinematic) gripper; it can be
  driven straight through other dynamic objects and through the table.
  Therefore obstructions sitting on the surface do NOT have to be cleared:
  we simply press the held block down onto the surface, shoving them away.
* Grasping is top-down.  The fingertips must dip ~0.11..0.26 below the
  object's top, the gripper must be open (0.32) while descending, and the
  block must be within ~0.09 of the gripper axis when the gripper closes.
  Blocks wider than ~0.44 cannot be grasped at all -> we push those along
  the table instead (pushing with the extended arm is very accurate).
"""
from __future__ import annotations

import math

import numpy as np

TABLE = 0.1
X_MIN, X_MAX = 0.245, 2.990
WALL_L, WALL_R = 0.0, 3.2358          # reachable base centre range
DOWN = -math.pi / 2
GAP_OPEN = 0.32
GAP_MIN = 0.12
ARM_MIN = 0.24
ARM = 0.48                            # arm length used everywhere
TIP = 0.2005                          # fingertip beyond arm_joint
BAR_HALF = 0.16                       # half width of gripper base bar
TRAVEL_Y = 1.42
Y_MAX = 1.74
# (lateral offset, descent depth robot_y - block_top) ladder for grasp tries
LADDER = [(0.00, 0.49), (0.00, 0.44), (0.06, 0.49), (-0.06, 0.49),
          (0.10, 0.44), (-0.10, 0.44), (0.00, 0.47)]


class O:
    pass


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.low = np.asarray(action_space.low, dtype=np.float64) * 0.985
        self.high = np.asarray(action_space.high, dtype=np.float64) * 0.985
        self.reset(None, None)

    # ------------------------------------------------------------------ setup
    def reset(self, state, info):
        self.stage = "raise"
        self.attempt = 0
        self.timer = 0
        self.t = 0
        self.push_dir = 0.0
        self.push_target = None
        self.last_bx = None
        self.stall = 0
        self.carry_y = TRAVEL_Y
        self.ty = TRAVEL_Y
        self.press_extra = 0.0
        self.dither = 0.0
        self.dither_i = 0
        self.settled = 0
        self.frozen = 0
        self.prev_pose = None
        self.prev_act = None
        self.planned = False
        self.push_retry = 0
        self.switched = False
        self.pd_count = 0
        self.settled = 0
        self.frozen = 0
        self.prev_pose = None
        self.prev_act = None

    # ------------------------------------------------------------------ state
    def _parse(self, state):
        robot = block = surf = None
        obst = []
        for name in state.get_object_names():
            ob = state.get_object_from_name(name)
            tn = ob.type.name
            o = O()
            o.name = name
            if tn == "kin_robot":
                for f in ("x", "y", "theta", "arm_joint", "finger_gap"):
                    setattr(o, f, float(state.get(ob, f)))
                robot = o
                continue
            o.x = float(state.get(ob, "x"))
            o.y = float(state.get(ob, "y"))
            o.theta = float(state.get(ob, "theta"))
            o.w = float(state.get(ob, "width"))
            o.h = float(state.get(ob, "height"))
            try:
                o.held = float(state.get(ob, "held"))
            except Exception:
                o.held = 0.0
            o.top = o.y + o.h / 2.0
            o.left = o.x - o.w / 2.0
            o.right = o.x + o.w / 2.0
            if tn == "target_block":
                block = o
            elif tn == "target_surface":
                surf = o
            else:
                obst.append(o)
        return robot, block, surf, obst

    def _a(self, dx=0.0, dy=0.0, dth=0.0, darm=0.0, dg=0.0):
        v = np.array([dx, dy, dth, darm, dg], dtype=np.float64)
        if not np.all(np.isfinite(v)):
            v = np.zeros(5)
        return np.clip(v, self.low, self.high)

    def _goto(self, r, x=None, y=None, th=None, arm=None, gap=None, tol=3e-3):
        vals = (r.x, r.y, r.theta, r.arm_joint, r.finger_gap)
        a = [0.0] * 5
        done = True
        for i, tgt in enumerate((x, y, th, arm, gap)):
            if tgt is None:
                continue
            d = tgt - vals[i]
            a[i] = d
            if abs(d) > tol:
                done = False
        return self._a(*a), done

    # ------------------------------------------------------------------- main
    def get_action(self, state):
        self.t += 1
        r, b, s, obst = self._parse(state)
        if r is None or b is None or s is None:
            return self._a()
        if not self.planned:
            self.planned = True
            try:
                self._plan(r, b, s, obst)
            except Exception:
                pass
        pose = (r.x, r.y, r.theta, r.arm_joint, r.finger_gap)
        if (self.prev_pose is not None
                and max(abs(a - c) for a, c in zip(pose, self.prev_pose)) < 4e-4
                and self.prev_act is not None
                and float(np.max(np.abs(self.prev_act))) > 1e-4):
            self.frozen += 1
        else:
            self.frozen = 0
        self.prev_pose = pose
        if (self.t >= 480 and b.held < 0.5 and not self.switched):
            self.switched = True
            self.attempt = 0
            self.press_extra = 0.0
            self.dither = 0.0
            self._go("raise" if self.stage.startswith("push") else "push_setup")
        if (self.t > 900 and b.held < 0.5
                and not self.stage.startswith("push")):
            self._go("push_setup")
        try:
            act = getattr(self, "_st_" + self.stage)(r, b, s, obst)
        except Exception:
            act = self._a()
        if self.frozen >= 2:
            act = act * 0.45
            if self.frozen >= 4:
                act[1] = 0.0
            if self.frozen >= 7:
                act[3] = 0.0
                act[2] = 0.0
            if self.frozen >= 10:
                act = act * 0.0
                act[1] = 0.02      # jiggle up to escape
        self.prev_act = act
        return act

    def _travel_y(self, obst, b=None):
        mtop = TABLE
        for o in obst:
            if o.held < 0.5:
                mtop = max(mtop, o.top)
        if b is not None and b.held < 0.5:
            mtop = max(mtop, b.top)
        return min(Y_MAX, max(TRAVEL_Y, mtop + 0.03 + ARM + TIP))

    def _plan(self, r, b, s, obst):
        """Pushing the block along the table is far more reliable than the
        top-down pinch (which only works for widths <= ~0.3), so push
        whenever everything that gets swept ahead still fits on the table."""
        d = 1.0 if s.x > b.x else -1.0
        half = max(0.0, (s.w - b.w) / 2.0)
        target = s.x - d * max(0.0, half - 0.022)
        front = target + d * b.w / 2.0
        need = 0.0
        for o in obst:
            if (o.x - b.x) * d > 0.0:
                need += o.w + 0.012
        room = (WALL_R - front) if d > 0 else (front - WALL_L)
        graspable = b.w <= 0.30
        if room >= need + 0.02 and not (graspable and need <= 0.0):
            self.push_dir = d
            self.push_target = target
            self._go("push_lift")

    def _go(self, stage):
        self.stage = stage
        self.timer = 0

    # ------------------------------------------------------- grasp approach
    def _st_raise(self, r, b, s, obst):
        """Retract the arm and lift straight up to a safe height."""
        self.ty = self._travel_y(obst, b)
        act, done = self._goto(r, y=self.ty, arm=ARM_MIN)
        self.timer += 1
        if done or self.timer > 60:
            self._go("orient")
        return act

    def _st_orient(self, r, b, s, obst):
        act, done = self._goto(r, y=self.ty, th=DOWN, arm=ARM, gap=GAP_OPEN)
        self.timer += 1
        if done or self.timer > 60:
            self._go("over")
        return act

    def _grasp_pose(self, b):
        off, depth = LADDER[min(self.attempt, len(LADDER) - 1)]
        tx = min(max(b.x + off, X_MIN), X_MAX)
        return tx, b.top + depth

    def _st_over(self, r, b, s, obst):
        if b.held > 0.5:
            self._go("carry")
            return self._a()
        tx, gy = self._grasp_pose(b)
        dx = tx - r.x
        y_t = max(self.ty, gy) if abs(dx) > 0.22 else gy
        act, _ = self._goto(r, x=tx, y=y_t, th=DOWN, arm=ARM, gap=GAP_OPEN)
        self.timer += 1
        if (abs(dx) < 5e-3 and abs(r.y - gy) < 5e-3) or self.timer > 160:
            self._go("close")
        return act

    def _st_close(self, r, b, s, obst):
        if b.held > 0.5:
            self._go("carry")
            return self._a()
        relx = b.x - r.x
        if abs(relx) > 0.095 or self.timer > 12 or r.finger_gap <= GAP_MIN + 1e-3:
            # misaligned or the gripper is shut: give up on this attempt
            self.attempt += 1
            if self.attempt >= len(LADDER):
                self._go("push_setup")
            else:
                self._go("reopen")
            return self._a()
        self.timer += 1
        return self._a(0, 0, 0, 0, -0.0195)

    def _st_reopen(self, r, b, s, obst):
        self.ty = self._travel_y(obst, b)
        act, done = self._goto(r, y=self.ty, gap=GAP_OPEN)
        self.timer += 1
        if done or self.timer > 60:
            self._go("over")
        return act

    # -------------------------------------------------------------- carrying
    def _st_carry(self, r, b, s, obst):
        if b.held < 0.5:
            self.attempt += 1
            self._go("raise" if self.attempt < len(LADDER) else "push_setup")
            return self._a()
        th = b.theta
        sn, cs = math.sin(th), math.cos(th)
        want_x = s.x - (b.h / 2.0) * sn + self.dither
        # lower the block until its HIGHEST bottom vertex is at/below 0.096
        want_y = (0.096 + (b.h / 2.0) * abs(cs) - (b.w / 2.0) * abs(sn)
                  - self.press_extra)
        ex, ey = want_x - b.x, want_y - b.y
        # rotating is BLOCKED (and zeroes the whole action) when low, so only
        # straighten the block while we are still up at travel height
        dth = 0.0
        if abs(th) > 0.008 and r.y > 1.00:
            dth = float(np.clip(-th, -0.15, 0.15))
        # clearance height while travelling horizontally
        lo, hi = min(b.x, want_x) - b.w, max(b.x, want_x) + b.w
        mtop = TABLE
        for o in obst:
            if o.held < 0.5 and o.right > lo and o.left < hi:
                mtop = max(mtop, o.top)
        rel_y = b.y - r.y
        safe_r_y = min(Y_MAX, mtop + 0.04 + b.h / 2.0 - rel_y)
        if abs(ex) > 0.05:
            dy = safe_r_y - r.y
            if dy > 0.004:
                return self._a(np.clip(ex, -0.025, 0.025), dy, dth, 0, 0)
            return self._a(ex, min(0.0, max(dy, -0.03)), dth, 0, 0)
        # final press: descend as far as the table allows
        margin = r.y - r.arm_joint - TIP - (TABLE + 0.002)
        dy = float(np.clip(ey, -0.049, 0.049))
        if dy < -margin:
            dy = -max(0.0, margin)
        self.timer += 1
        if self.frozen >= 2:
            dy = 0.0
        if abs(ex) < 0.006 and (abs(dy) < 0.0025 or margin <= 0.002
                                or self.frozen >= 2):
            self.settled += 1
        if self.settled > 4:
            self.settled = 0
            if self.press_extra < 0.05 and margin > 0.002:
                self.press_extra += 0.006
            else:
                # search sideways a little; the geometric test is tight
                self.dither_i += 1
                k = (self.dither_i + 1) // 2
                self.dither = 0.008 * k * (1 if self.dither_i % 2 else -1)
                if k > 4:
                    self.dither = 0.0
                    self.dither_i = 0
                    self.press_extra = 0.0
                    self.attempt += 1
                    self._go("raise" if self.attempt < len(LADDER)
                             else "push_setup")
        return self._a(ex, dy, 0.0, 0.0, 0)

    # --------------------------------------------------------------- pushing
    def _st_push_setup(self, r, b, s, obst):
        self.push_dir = 1.0 if s.x > b.x else -1.0
        d = self.push_dir
        half = max(0.0, (s.w - b.w) / 2.0)
        # aim a little past the near edge, leaving room for what we sweep
        self.push_target = s.x - d * max(0.0, half - 0.022)
        self.attempt = 0
        self._go("push_lift")
        return self._a()

    def _st_push_lift(self, r, b, s, obst):
        act, done = self._goto(r, y=TRAVEL_Y, th=DOWN, arm=ARM, gap=GAP_MIN)
        self.timer += 1
        if done or self.timer > 60:
            self._go("push_over")
        return act

    def _st_push_over(self, r, b, s, obst):
        d = self.push_dir
        ry = TABLE + TIP + ARM + 0.020
        start_x = min(max(b.x - d * (b.w / 2.0 + BAR_HALF + 0.04), X_MIN), X_MAX)
        dx = start_x - r.x
        y_t = TRAVEL_Y if abs(dx) > 0.18 else ry
        act, _ = self._goto(r, x=start_x, y=y_t, th=DOWN, arm=ARM, gap=GAP_MIN)
        self.timer += 1
        if ((abs(dx) < 5e-3 and abs(r.y - ry) < 0.02)
                or self.frozen >= 3 or self.timer > 120):
            self.last_bx = b.x
            self.stall = 0
            self._go("push")
        return act

    def _st_push(self, r, b, s, obst):
        d = self.push_dir
        err = (self.push_target - b.x) * d
        self.timer += 1
        if err < 0.003:
            self._go("push_done")
            return self._a()
        if self.timer > 8:
            if abs(b.x - (self.last_bx if self.last_bx is not None else b.x)) < 0.0015:
                self.stall += 1
            else:
                self.stall = 0
        self.last_bx = b.x
        if self.stall > 30 or (abs(b.theta) > 0.07 and self.timer > 10):
            self._go("push_done")
        step = min(0.049, max(0.005, err * 0.9))
        return self._a(d * step, 0, 0, 0, 0)

    def _st_push_done(self, r, b, s, obst):
        """At the nominal target: creep forward across the legal window."""
        d = self.push_dir
        err = (self.push_target - b.x) * d
        self.timer += 1
        if self.timer == 1:
            self.pd_count += 1
        if self.timer > 70 or self.frozen > 6 or self.pd_count > 5:
            self._go("settle")
            return self._a()
        if err > 0.003:
            return self._a(d * float(np.clip(err, -0.012, 0.012)), 0, 0, 0, 0)
        if err < -0.02:      # got knocked backwards: push again
            self._go("push")
            return self._a()
        if self.timer % 5 == 0:
            lim = (s.w - b.w) / 2.0 - 0.014
            nxt = self.push_target + d * 0.006
            if abs(nxt - s.x) <= max(0.0, lim):
                self.push_target = nxt
            else:
                # window exhausted -> settle, then fall back to grasping
                self.attempt = 0
                self.press_extra = 0.0
                self.dither = 0.0
                self._go("settle")
        return self._a(d * 0.004, 0, 0, 0, 0)

    def _st_settle(self, r, b, s, obst):
        """Back the pusher off and let the block come to rest."""
        self.timer += 1
        d = self.push_dir
        if self.timer < 12:
            return self._a(-d * 0.03, 0, 0, 0, 0)
        if self.timer < 22:
            return self._a()
        if self.push_retry < 1:
            self.push_retry += 1
            lim = max(0.0, (s.w - b.w) / 2.0 - 0.014)
            self.push_target = s.x + d * min(lim, 0.012)
            if (self.push_target - b.x) * d < 0.004:
                self.push_target = b.x + d * 0.006
            self._go("push_over")
            return self._a()
        self._go("raise")
        return self._a()
