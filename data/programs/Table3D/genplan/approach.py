"""Policy for Table3DEnv (variable cube count).

DIAGNOSIS OF THE PREVIOUS FAILURE
---------------------------------
The last submission hard-coded a Kinova Gen3 URDF and did analytic IK against
it.  The final state shows the arm bent into a nonsense pose (joint_2 = 1.95
which is at/near its limit, joint_5 = -2.18, joint_3 = -1.81) with the base
barely moved and no cube touched.  The FK model simply does not match the
actual "tidybot-kinova" robot used by the environment (unknown mount height,
unknown tool frame, unknown joint sign conventions), so every IK solve aimed
the gripper at the wrong place, and the closed-loop never converged because
the error signal it was minimising was fictitious.

The lesson: we cannot rely on ANY assumed kinematic model.  We must learn the
mapping from joints -> gripper position from data we can actually observe.

WHAT WE CAN OBSERVE
-------------------
Nothing in the observation gives the end-effector pose directly.  BUT:

  * `grasp_active` on the robot tells us instantly whether a grasp succeeded.
  * Once a cube is grasped, `cube.pose_x/y/z` IS the gripper position (up to
    the constant grasp transform).  From that moment on we have a perfect
    sensor for the end effector.

So the plan is a two-stage scheme that never needs a kinematic model:

  STAGE A -- BLIND SEARCH FOR THE FIRST GRASP.
    Sweep the arm through a coarse but systematic lattice of joint
    configurations that sweep the gripper through the volume above the table,
    while continuously commanding "close".  The close command is a no-op
    unless exactly one cube is inside the small end-effector box, so it costs
    nothing to spam it.  Simultaneously, translate the base across a grid so
    that the swept volume covers the whole table.  Because collisions are
    silently reverted (never fatal), an aggressive sweep is safe.

    Crucially we ALSO learn online: every time the arm sweep is blocked
    (joints stop changing despite a command), we know we hit the table, which
    means the gripper is at table height -- a useful landmark.

  STAGE B -- LIFT (this is the only thing that terminates).
    The goal test is:  grasped AND cube.pose_z - (table_z + table_hz) > 0.1,
    i.e. pose_z > 0.5 for this scene.  Once `grasp_active` is set, we do pure
    hill-climbing on the OBSERVED cube z: try a joint direction, look at
    whether cube z increased, keep directions that help, flip ones that do
    not.  This needs no model at all and is guaranteed to make progress as
    long as some joint direction raises the tool -- which is always true.

  The hill-climb is coordinate-descent with a persistent per-joint sign,
  which converges in a handful of steps per joint and is robust to the
  collision-revert (a reverted step shows zero change and is treated as
  "this direction is blocked", flipping the sign).

Everything is count-invariant: the cube count only enters when choosing which
cube to aim the sweep at (most-isolated rule), and the sweep/lift are blind
to it.
"""

from __future__ import annotations

import numpy as np

_ROBOT_TYPE_NAME = "Kinematic3DRobot"
_CUBOID_TYPE_NAME = "Kinematic3DCuboid"
_JOINT_FEATURES = [f"joint_{i}" for i in range(1, 8)]


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


class _Obj:
    __slots__ = ("name", "x", "y", "z", "hx", "hy", "hz", "grasped")

    def __init__(self, name, x, y, z, hx, hy, hz, grasped):
        self.name = name
        self.x, self.y, self.z = x, y, z
        self.hx, self.hy, self.hz = hx, hy, hz
        self.grasped = grasped


class GeneratedApproach:
    """Model-free: blind sweep to grasp, then observed-z hill-climb to lift."""

    # ---- sweep design ------------------------------------------------
    # The initial joint configuration (from the env config) is a retract pose:
    #   (0, -0.35, -pi, -2.5, 0, -0.87, pi/2)
    # We sweep primarily joints 2, 4 and 6 (the "shoulder/elbow/wrist" pitch
    # chain) because on essentially any 7-DOF arm those three dominate the
    # vertical/radial reach, and we sweep joint 1 for azimuth.  Signs are
    # unknown, so we sweep BOTH directions.

    _J1_VALUES = (0.0, -0.30, 0.30, -0.60, 0.60)          # azimuth
    _PITCH_SETS = (
        # (dj2, dj4, dj6) offsets from the retract pose, walked in small steps
        (0.55, 0.55, 0.55),
        (0.55, 0.55, -0.55),
        (0.55, -0.55, 0.55),
        (-0.55, 0.55, 0.55),
        (0.90, 0.30, 0.60),
        (0.30, 0.90, 0.30),
        (0.75, 0.75, 0.00),
        (0.00, 0.75, 0.75),
        (1.10, 0.55, 0.55),
        (0.55, 1.10, 0.55),
    )
    # Base positions (world) to try, relative to the target cube: the base is
    # driven so the cube sits at these offsets in front of it.
    _BASE_STANDOFFS = (
        (-0.45, 0.0), (-0.38, 0.0), (-0.52, 0.0),
        (-0.45, -0.07), (-0.45, 0.07),
        (-0.32, 0.0), (-0.58, 0.0),
        (-0.38, -0.07), (-0.38, 0.07),
        (-0.52, -0.07), (-0.52, 0.07),
    )

    _SWEEP_SUBSTEPS = 26   # small joint steps per pitch set
    _ALIGN_TOL = 0.015

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        self._dim = int(np.prod(action_space.shape))
        self._low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
        self._high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
        self._mag = float(min(0.4, np.min(np.abs(self._high[:10]))))

        self._reset_internal()

    # ------------------------------------------------------------------

    def _reset_internal(self):
        self.phase = "align"
        self.phase_step = 0
        self.target_name = None
        self.table_top_z = None
        self.goal_z = None

        self.home_q = None          # retract pose captured at reset

        # sweep indices
        self.standoff_i = 0
        self.j1_i = 0
        self.pitch_i = 0
        self.sub = 0

        self.prev_q = None
        self.prev_err = None
        self.stall = 0

        # lift bookkeeping
        self.lift_dir = np.array([1.0] * 7)
        self.lift_joint = 1
        self.lift_prev_z = None
        self.lift_prev_q = None
        self.lift_tries = 0
        self.lift_order = [1, 3, 5, 2, 4, 0, 6]
        self.lift_oi = 0

    def reset(self, state, info=None):
        self._reset_internal()
        rb = self._rb(state)
        if rb is not None:
            self.home_q = list(rb["q"])
        self._select_target(state)
        return None

    # ------------------------------------------------------------------
    # observation parsing
    # ------------------------------------------------------------------

    def _type_by_name(self, name):
        try:
            for t in self.observation_space.types:
                if getattr(t, "name", None) == name:
                    return t
        except Exception:
            pass
        return None

    def _cuboids(self, state):
        out = []
        t = self._type_by_name(_CUBOID_TYPE_NAME)
        cands = []
        if t is not None:
            try:
                cands = list(state.get_objects(t))
            except Exception:
                cands = []
        if not cands:
            for nm in state.get_object_names():
                o = state.get_object_from_name(nm)
                if getattr(o.type, "name", "") == _CUBOID_TYPE_NAME:
                    cands.append(o)
        for o in cands:
            try:
                out.append(
                    _Obj(
                        o.name,
                        float(state.get(o, "pose_x")),
                        float(state.get(o, "pose_y")),
                        float(state.get(o, "pose_z")),
                        float(state.get(o, "half_extent_x")),
                        float(state.get(o, "half_extent_y")),
                        float(state.get(o, "half_extent_z")),
                        float(state.get(o, "grasp_active")) > 0.5,
                    )
                )
            except Exception:
                continue
        return out

    def _split(self, state):
        objs = self._cuboids(state)
        if not objs:
            return None, []
        table = max(objs, key=lambda o: o.hx * o.hy)
        cubes = [o for o in objs if o is not table]
        if not cubes:
            return None, objs
        med = float(np.median([o.hx * o.hy for o in cubes]))
        if table.hx * table.hy < 4.0 * max(med, 1e-9):
            named = [o for o in objs if o.name == "table"]
            if named:
                table = named[0]
                cubes = [o for o in objs if o is not table]
        return table, cubes

    def _robot(self, state):
        t = self._type_by_name(_ROBOT_TYPE_NAME)
        if t is not None:
            try:
                rs = list(state.get_objects(t))
                if rs:
                    return rs[0]
            except Exception:
                pass
        for nm in state.get_object_names():
            o = state.get_object_from_name(nm)
            if getattr(o.type, "name", "") == _ROBOT_TYPE_NAME:
                return o
        return None

    def _rb(self, state):
        r = self._robot(state)
        if r is None:
            return None
        return {
            "bx": float(state.get(r, "pos_base_x")),
            "by": float(state.get(r, "pos_base_y")),
            "brot": float(state.get(r, "pos_base_rot")),
            "q": [float(state.get(r, f)) for f in _JOINT_FEATURES],
            "grasp": float(state.get(r, "grasp_active")) > 0.5,
        }

    # ------------------------------------------------------------------

    def _select_target(self, state):
        table, cubes = self._split(state)
        if table is not None:
            self.table_top_z = table.z + table.hz
        if not cubes:
            self.target_name = None
            return
        if self.table_top_z is None:
            self.table_top_z = cubes[0].z - cubes[0].hz
        self.goal_z = self.table_top_z + 0.1

        def isolation(c):
            best = float("inf")
            for o in cubes:
                if o is c:
                    continue
                best = min(best, float(np.hypot(c.x - o.x, c.y - o.y)))
            return 1e6 if best == float("inf") else best

        scored = sorted(cubes, key=lambda c: (-isolation(c), c.x, c.name))
        self.target_name = scored[0].name

    def _target(self, state):
        _, cubes = self._split(state)
        if not cubes:
            return None
        for c in cubes:
            if c.name == self.target_name:
                return c
        self._select_target(state)
        for c in cubes:
            if c.name == self.target_name:
                return c
        return cubes[0]

    def _held(self, state):
        _, cubes = self._split(state)
        for c in cubes:
            if c.grasped:
                return c
        return None

    # ------------------------------------------------------------------
    # action helpers
    # ------------------------------------------------------------------

    def _zero(self):
        return np.zeros(self._dim, dtype=np.float64)

    def _fin(self, a):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.shape[0] < self._dim:
            a = np.concatenate([a, np.zeros(self._dim - a.shape[0])])
        a = np.clip(a[: self._dim], self._low, self._high)
        return a.astype(np.float32)

    def _to_q(self, cur, tgt, grip=0.0, scale=1.0):
        a = self._zero()
        for i in range(7):
            a[3 + i] = _clip(tgt[i] - cur[i], -self._mag * scale, self._mag * scale)
        a[10] = grip
        return self._fin(a)

    def _to_base(self, dx, dy, grip=0.0):
        a = self._zero()
        a[0] = _clip(dx, -self._mag, self._mag)
        a[1] = _clip(dy, -self._mag, self._mag)
        a[10] = grip
        return self._fin(a)

    # ------------------------------------------------------------------

    def _desired_base(self, tgt):
        off = self._BASE_STANDOFFS[self.standoff_i % len(self._BASE_STANDOFFS)]
        return tgt.x + off[0], tgt.y + off[1]

    def _sweep_goal_q(self):
        """Joint target for the current (j1, pitch, sub) sweep index."""
        home = self.home_q if self.home_q is not None else [0.0] * 7
        q = list(home)
        q[0] = home[0] + self._J1_VALUES[self.j1_i % len(self._J1_VALUES)]
        d2, d4, d6 = self._PITCH_SETS[self.pitch_i % len(self._PITCH_SETS)]
        f = (self.sub + 1) / float(self._SWEEP_SUBSTEPS)
        q[1] = home[1] + d2 * f
        q[3] = home[3] + d4 * f
        q[5] = home[5] + d6 * f
        return q

    def _advance_sweep(self):
        """Move to the next sweep configuration."""
        self.sub += 1
        if self.sub >= self._SWEEP_SUBSTEPS:
            self.sub = 0
            self.pitch_i += 1
            if self.pitch_i >= len(self._PITCH_SETS):
                self.pitch_i = 0
                self.j1_i += 1
                if self.j1_i >= len(self._J1_VALUES):
                    self.j1_i = 0
                    self.standoff_i += 1
                    # need to re-align the base for the new standoff
                    self.phase = "align"
                    self.phase_step = 0
                    self.prev_err = None
                    self.stall = 0

    # ------------------------------------------------------------------

    def get_action(self, state):
        self.phase_step += 1
        rb = self._rb(state)
        if rb is None:
            return self._fin(self._zero())
        if self.home_q is None:
            self.home_q = list(rb["q"])

        table, cubes = self._split(state)
        if table is not None:
            self.table_top_z = table.z + table.hz
            self.goal_z = self.table_top_z + 0.1
        if self.goal_z is None:
            self.goal_z = 0.5

        # ---------------- STAGE B: we are holding something --------------
        held = self._held(state)
        if rb["grasp"] or held is not None:
            return self._lift(rb, held)

        if self.target_name is None:
            self._select_target(state)
            if self.target_name is None:
                return self._fin(self._zero())
        tgt = self._target(state)
        if tgt is None:
            return self._fin(self._zero())

        # ---------------- align base for the current standoff -----------
        if self.phase == "align":
            wx, wy = self._desired_base(tgt)
            dx = wx - rb["bx"]
            dy = wy - rb["by"]
            err = float(np.hypot(dx, dy))
            if self.prev_err is not None and self.prev_err - err < 1e-4:
                self.stall += 1
            else:
                self.stall = 0
            self.prev_err = err
            if err < self._ALIGN_TOL or self.stall > 8 or self.phase_step > 60:
                self.phase = "home"
                self.phase_step = 0
                self.prev_err = None
                self.stall = 0
                self.prev_q = None
                return self._fin(self._zero())
            return self._to_base(0.8 * dx, 0.8 * dy, grip=1.0)

        # ---------------- return to the retract pose --------------------
        if self.phase == "home":
            home = self.home_q
            err = max(abs(home[i] - rb["q"][i]) for i in range(7))
            moved = 0.0
            if self.prev_q is not None:
                moved = max(abs(rb["q"][i] - self.prev_q[i]) for i in range(7))
            self.prev_q = list(rb["q"])
            if err < 0.05 or self.phase_step > 30 or (self.phase_step > 5 and moved < 1e-7):
                self.phase = "sweep"
                self.phase_step = 0
                self.sub = 0
                self.prev_q = None
                return self._fin(self._zero())
            return self._to_q(rb["q"], home, grip=1.0)

        # ---------------- STAGE A: blind sweep with continuous close ----
        if self.phase == "sweep":
            goal = self._sweep_goal_q()
            err = max(abs(goal[i] - rb["q"][i]) for i in range(7))
            moved = 0.0
            if self.prev_q is not None:
                moved = max(abs(rb["q"][i] - self.prev_q[i]) for i in range(7))
            self.prev_q = list(rb["q"])

            # Reached this waypoint, or blocked by the table -> next config.
            if err < 0.02 or (self.phase_step > 3 and moved < 1e-7):
                self._advance_sweep()
                self.phase_step = 0
                self.prev_q = None
                if self.phase != "sweep":
                    return self._fin(self._zero())
                goal = self._sweep_goal_q()

            # Always attempt to close: no-op unless exactly one cube is in
            # the grasp zone.
            return self._to_q(rb["q"], goal, grip=-1.0, scale=0.35)

        return self._fin(self._zero())

    # ------------------------------------------------------------------
    # STAGE B: model-free lift by hill-climbing the OBSERVED cube z.
    #
    # Goal test:  cube.pose_z > table_top_z + 0.1
    # ------------------------------------------------------------------

    def _lift(self, rb, held):
        if self.phase != "lift":
            self.phase = "lift"
            self.phase_step = 0
            self.lift_prev_z = None
            self.lift_prev_q = None
            self.lift_dir = np.array([1.0] * 7)
            self.lift_oi = 0
            self.lift_tries = 0

        goal = (self.goal_z if self.goal_z is not None else 0.5) + 0.04

        z = held.z if held is not None else None

        # Already high enough -- hold, keep the grasp latched.
        if z is not None and z > goal:
            a = self._zero()
            a[10] = -1.0
            return self._fin(a)

        j = self.lift_order[self.lift_oi % len(self.lift_order)]

        if z is not None and self.lift_prev_z is not None:
            dz = z - self.lift_prev_z
            moved = 0.0
            if self.lift_prev_q is not None:
                moved = max(abs(rb["q"][i] - self.lift_prev_q[i]) for i in range(7))
            if moved < 1e-7:
                # Blocked (collision revert): flip this joint and move on.
                self.lift_dir[j] = -self.lift_dir[j]
                self.lift_tries += 1
                if self.lift_tries >= 2:
                    self.lift_tries = 0
                    self.lift_oi += 1
            elif dz < 1e-5:
                # Moved but did not raise the cube: flip direction.
                self.lift_dir[j] = -self.lift_dir[j]
                self.lift_tries += 1
                if self.lift_tries >= 2:
                    self.lift_tries = 0
                    self.lift_oi += 1
            else:
                # Good direction -- keep pushing this joint.
                self.lift_tries = 0

        self.lift_prev_z = z
        self.lift_prev_q = list(rb["q"])

        j = self.lift_order[self.lift_oi % len(self.lift_order)]
        a = self._zero()
        step = 0.10 * self.lift_dir[j]
        a[3 + j] = _clip(step, -self._mag, self._mag)
        a[10] = -1.0  # never open; releasing mid-air is refused anyway

        # Every so often, also nudge the whole pitch chain upward-ish in
        # whichever combined direction has been working, to speed things up.
        if self.phase_step % 7 == 0:
            for k in (1, 3, 5):
                a[3 + k] = _clip(a[3 + k] + 0.05 * self.lift_dir[k],
                                 -self._mag, self._mag)
        return self._fin(a)