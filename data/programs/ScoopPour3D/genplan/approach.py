"""
GeneratedApproach for ScoopPour3D (variable object count).

HONEST PREAMBLE (read before trusting this):
---------------------------------------------
I do not have access to the task JSON (tasks/ScoopPour3D/ScoopPour3D-o*.json),
so I do not know the exact goal predicate, the exact goal region bounds, or the
exact semantics of the reward calculator that actually runs for this scene type.
Everything below is inferred from:
  * the two dumped initial states (bin_yellow_0 at y ~= -0.20, bin_green_0 at
    y ~= +0.20, both at x ~= 0.50, z ~= 0.459, on the kitchen island top),
  * the cubes all starting inside/above the yellow bin,
  * the env name / description ("transfer a pile of objects from one bin to
    another"),
  * the action space layout stated in the card: base pose (3), arm joints (7),
    gripper (1) = 11 dims, with config.act_delta=True by default.

Because of that uncertainty, this policy is written to be *robust and safe*
rather than clever:

STRATEGY
--------
The single simple strategy that scales to ANY cube count is: do not manipulate
cubes individually. Instead move the SOURCE CONTAINER (the yellow bin, which is
a mujoco_movable_object with a free joint) so that its contents come to rest at
the target. Concretely the policy:

  1. Identifies, from the state alone and with no hardcoded names, which bin is
     the "source" (the one that currently contains the most cubes) and which is
     the "target" (the other bin / the bin containing fewest cubes).
  2. Drives the holonomic base to a standoff pose beside the source bin, facing
     it.
  3. Reaches the arm down, closes the gripper on the source bin's rim, lifts,
     translates the base toward the target bin, and tips/pours.
  4. If at any point progress stalls (cubes stop moving toward the target for a
     long stretch), it falls back to a slow "sweep" behaviour: it drives the
     base back and forth along the line between the two bins with the arm
     extended low, which nudges loose cubes toward the target.

All of steps 2-4 are expressed as a small scripted state machine over base/arm
targets. Every action is clipped into action_space, and every observation read
is defensive (missing feature -> default), so this module will not crash even
if my assumptions about naming or geometry are wrong. If the strategy simply
does not achieve the goal, it will still run for the full horizon returning
valid in-bounds actions rather than erroring.

I want to be explicit: I am NOT confident this solves the task. It is my best
grounded guess given what I could actually read. The things most likely to be
wrong are (a) whether grasping/moving the source bin satisfies the goal check,
and (b) the exact gripper-open/closed convention (env stores pos_gripper in
[0,1] and writes ctrl = pos*255; I assume larger = more closed, and I hedge by
making the grasp value configurable and by not depending on a firm grasp for
the fallback sweep).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Small defensive helpers. Every one of these tolerates missing objects and
# missing features, because the exact schema per instance is not guaranteed.
# ----------------------------------------------------------------------------


def _safe_get(state, obj, feature, default=0.0):
    """state.get(obj, feature) but never raises."""
    try:
        val = state.get(obj, feature)
    except Exception:
        return default
    try:
        f = float(val)
    except Exception:
        return default
    if not np.isfinite(f):
        return default
    return f


def _objects_by_type_name(state, type_name):
    """All objects whose type (or an ancestor type) is named `type_name`."""
    out = []
    for name in state.get_object_names():
        try:
            obj = state.get_object_from_name(name)
        except Exception:
            continue
        t = getattr(obj, "type", None)
        while t is not None:
            if getattr(t, "name", None) == type_name:
                out.append(obj)
                break
            t = getattr(t, "parent", None)
    return out


def _xyz(state, obj):
    return np.array(
        [
            _safe_get(state, obj, "x", 0.0),
            _safe_get(state, obj, "y", 0.0),
            _safe_get(state, obj, "z", 0.0),
        ],
        dtype=np.float64,
    )


def _bb(state, obj):
    return np.array(
        [
            _safe_get(state, obj, "bb_x", 0.0),
            _safe_get(state, obj, "bb_y", 0.0),
            _safe_get(state, obj, "bb_z", 0.0),
        ],
        dtype=np.float64,
    )


def _wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# Scene parsing: figure out which objects are cubes, which are bins, which is
# the scoop, and where the robot is -- all without hardcoding index layout.
# ----------------------------------------------------------------------------


class _Scene:
    """A parsed view of one ObjectCentricState."""

    # Names starting with this prefix are the count-defining objects.
    CUBE_PREFIX = "cube_"

    def __init__(self, state):
        self.movables = _objects_by_type_name(state, "mujoco_movable_object")
        self.robots = _objects_by_type_name(state, "mujoco_tidybot_robot")
        self.robot = self.robots[0] if self.robots else None

        self.cubes = []
        self.bins = []
        self.tools = []

        for obj in self.movables:
            nm = getattr(obj, "name", "") or ""
            low = nm.lower()
            if low.startswith(self.CUBE_PREFIX):
                self.cubes.append(obj)
            elif "bin" in low or "bowl" in low or "box" in low or "tray" in low:
                self.bins.append(obj)
            elif "scoop" in low or "spoon" in low or "ladle" in low:
                self.tools.append(obj)

        # Fallback classification purely by size, in case the naming differs.
        if not self.cubes or not self.bins:
            sized = []
            for obj in self.movables:
                if obj in self.tools:
                    continue
                vol = float(np.prod(np.maximum(_bb(state, obj), 1e-6)))
                sized.append((vol, obj))
            if sized:
                sized.sort(key=lambda t: t[0])
                # Small things are cubes, big things are containers.
                if not self.cubes:
                    med = sized[len(sized) // 2][0]
                    self.cubes = [o for v, o in sized if v <= max(med, 1e-9)]
                if not self.bins:
                    self.bins = [o for v, o in sized if v > 8.0 * max(sized[0][0], 1e-12)]

        # Deduplicate while preserving order.
        self.cubes = list(dict.fromkeys(self.cubes))
        self.bins = list(dict.fromkeys(self.bins))
        self.tools = list(dict.fromkeys(self.tools))


def _bin_contains_count(state, bin_obj, cubes, xy_pad=0.02, z_span=0.30):
    """How many cubes lie within the (padded) horizontal footprint of a bin."""
    bp = _xyz(state, bin_obj)
    bb = _bb(state, bin_obj)
    hx = 0.5 * bb[0] + xy_pad
    hy = 0.5 * bb[1] + xy_pad
    if hx <= 0.0:
        hx = 0.20
    if hy <= 0.0:
        hy = 0.15
    n = 0
    for c in cubes:
        cp = _xyz(state, c)
        if abs(cp[0] - bp[0]) <= hx and abs(cp[1] - bp[1]) <= hy:
            if abs(cp[2] - bp[2]) <= z_span:
                n += 1
    return n


# ----------------------------------------------------------------------------
# The approach.
# ----------------------------------------------------------------------------


class GeneratedApproach:
    # --- tunables ------------------------------------------------------------
    # Base standoff distance from the bin centre when approaching, in metres.
    STANDOFF = 0.62
    # Gripper command interpreted as "closed" / "open" in the raw action units
    # for the gripper channel. Resolved against action_space bounds in __init__.
    GRIP_CLOSED_FRAC = 1.0
    GRIP_OPEN_FRAC = 0.0

    # Per-step magnitude caps (the env is delta-controlled by default, so these
    # are deltas; if it turns out to be absolute control the clipping below
    # still keeps us legal, just slower).
    MAX_BASE_STEP = 0.06        # metres per step, per axis
    MAX_YAW_STEP = 0.10         # radians per step
    MAX_JOINT_STEP = 0.06       # radians per step, per joint

    # Phase step budgets (control steps). Deliberately generous; the horizon is
    # 300 + 150*count, so even at count=10 we have 1800 steps.
    T_SETTLE = 20
    T_APPROACH = 260
    T_DESCEND = 120
    T_GRASP = 40
    T_LIFT = 90
    T_TRANSFER = 300
    T_POUR = 140
    T_RELEASE = 40

    # Arm joint presets (7-vector, radians). These mirror the "home-ish" pose
    # seen in the dumped initial states and small perturbations of it. They are
    # intentionally close to the observed start pose so that delta control does
    # not fling the arm.
    HOME = np.array([0.0, -0.349, 3.1416, -2.548, 0.0, -0.873, 1.5708])
    REACH_DOWN = np.array([0.0, 0.30, 3.1416, -2.05, 0.0, -0.60, 1.5708])
    CARRY = np.array([0.0, -0.15, 3.1416, -2.40, 0.0, -0.80, 1.5708])
    TIP = np.array([0.0, 0.10, 3.1416, -2.20, 0.0, -0.20, 1.5708])

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives if primitives is not None else {}

        # Resolve action dimensionality and bounds defensively.
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        shape = getattr(action_space, "shape", None)
        if shape is not None and len(shape) >= 1:
            self.adim = int(shape[0])
        elif low is not None:
            self.adim = int(np.asarray(low).size)
        else:
            self.adim = 11

        if low is None:
            self.low = np.full(self.adim, -1.0, dtype=np.float64)
        else:
            self.low = np.asarray(low, dtype=np.float64).reshape(-1)[: self.adim]
        if high is None:
            self.high = np.full(self.adim, 1.0, dtype=np.float64)
        else:
            self.high = np.asarray(high, dtype=np.float64).reshape(-1)[: self.adim]

        # Guard against infinite bounds so our clipping is meaningful.
        self.low = np.where(np.isfinite(self.low), self.low, -1.0)
        self.high = np.where(np.isfinite(self.high), self.high, 1.0)

        # Channel layout: base(3) | arm(7) | gripper(1). If the dimension is
        # different, degrade gracefully.
        self.i_base = slice(0, min(3, self.adim))
        self.i_arm = slice(3, min(10, self.adim))
        self.i_grip = 10 if self.adim > 10 else None

        if self.i_grip is not None:
            g_lo = self.low[self.i_grip]
            g_hi = self.high[self.i_grip]
            self.grip_closed = g_lo + self.GRIP_CLOSED_FRAC * (g_hi - g_lo)
            self.grip_open = g_lo + self.GRIP_OPEN_FRAC * (g_hi - g_lo)
        else:
            self.grip_closed = 0.0
            self.grip_open = 0.0

        self._reset_internal()

    # -- lifecycle ------------------------------------------------------------

    def _reset_internal(self):
        self.t = 0
        self.phase = "settle"
        self.phase_t = 0
        self.source = None
        self.target = None
        self.src_xy0 = None
        self.tgt_xy0 = None
        self.best_delivered = -1
        self.stall_counter = 0
        self.sweep_dir = 1.0
        self.failed_once = False

    def reset(self, state, info):
        self._reset_internal()
        try:
            self._identify_bins(state)
        except Exception:
            self.source = None
            self.target = None
        return None

    # -- scene reasoning ------------------------------------------------------

    def _identify_bins(self, state):
        sc = _Scene(state)
        self.scene = sc
        if len(sc.bins) >= 2 and sc.cubes:
            counts = [(_bin_contains_count(state, b, sc.cubes), b) for b in sc.bins]
            counts.sort(key=lambda t: t[0])
            self.target = counts[0][1]   # fewest cubes -> destination
            self.source = counts[-1][1]  # most cubes  -> origin
        elif len(sc.bins) == 2:
            # No cubes readable; pick arbitrarily but deterministically by name.
            b = sorted(sc.bins, key=lambda o: getattr(o, "name", ""))
            self.source, self.target = b[0], b[1]
        elif len(sc.bins) == 1:
            self.source = sc.bins[0]
            self.target = None
        if self.source is not None:
            self.src_xy0 = _xyz(state, self.source)[:2].copy()
        if self.target is not None:
            self.tgt_xy0 = _xyz(state, self.target)[:2].copy()

    def _delivered_count(self, state):
        """How many cubes are currently over the target bin."""
        if self.target is None:
            return 0
        try:
            sc = _Scene(state)
            return _bin_contains_count(state, self.target, sc.cubes)
        except Exception:
            return 0

    # -- action construction --------------------------------------------------

    def _blank(self):
        return np.zeros(self.adim, dtype=np.float64)

    def _finish(self, a):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.size < self.adim:
            a = np.concatenate([a, np.zeros(self.adim - a.size)])
        a = a[: self.adim]
        a = np.where(np.isfinite(a), a, 0.0)
        a = np.clip(a, self.low, self.high)
        return a.astype(np.float32)

    def _base_delta_toward(self, state, goal_xy, goal_yaw):
        """Clipped base delta driving toward (goal_xy, goal_yaw)."""
        r = self.scene.robot if getattr(self, "scene", None) else None
        if r is None:
            return np.zeros(3)
        bx = _safe_get(state, r, "pos_base_x", 0.0)
        by = _safe_get(state, r, "pos_base_y", 0.0)
        bth = _safe_get(state, r, "pos_base_rot", 0.0)
        dx = float(np.clip(goal_xy[0] - bx, -self.MAX_BASE_STEP, self.MAX_BASE_STEP))
        dy = float(np.clip(goal_xy[1] - by, -self.MAX_BASE_STEP, self.MAX_BASE_STEP))
        dth = _wrap_angle(goal_yaw - bth)
        dth = float(np.clip(dth, -self.MAX_YAW_STEP, self.MAX_YAW_STEP))
        return np.array([dx, dy, dth])

    def _arm_delta_toward(self, state, target_joints):
        r = self.scene.robot if getattr(self, "scene", None) else None
        if r is None:
            return np.zeros(7)
        cur = np.array(
            [_safe_get(state, r, "pos_arm_joint%d" % (i + 1), 0.0) for i in range(7)]
        )
        d = np.asarray(target_joints, dtype=np.float64)[:7] - cur
        return np.clip(d, -self.MAX_JOINT_STEP, self.MAX_JOINT_STEP)

    def _standoff_pose(self, bin_xy, away_from_xy):
        """A base (x, y, yaw) offset from bin_xy, on the side away from
        away_from_xy, facing the bin."""
        v = np.asarray(bin_xy, dtype=np.float64) - np.asarray(
            away_from_xy, dtype=np.float64
        )
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            v = np.array([-1.0, 0.0])
            n = 1.0
        u = v / n
        stand = np.asarray(bin_xy, dtype=np.float64) + u * self.STANDOFF
        yaw = float(np.arctan2(bin_xy[1] - stand[1], bin_xy[0] - stand[0]))
        return stand, yaw

    # -- main -----------------------------------------------------------------

    def get_action(self, state):
        try:
            return self._get_action_inner(state)
        except Exception:
            # Absolutely never raise from a policy; a zero action is always legal
            # after clipping.
            return self._finish(self._blank())

    def _get_action_inner(self, state):
        self.t += 1
        self.phase_t += 1

        # (Re)parse the scene each step; object identities are stable but poses
        # are not, and this keeps us robust to any set_state jumps.
        self.scene = _Scene(state)
        if self.source is None or self.target is None:
            self._identify_bins(state)

        # Progress / stall bookkeeping.
        delivered = self._delivered_count(state)
        if delivered > self.best_delivered:
            self.best_delivered = delivered
            self.stall_counter = 0
        else:
            self.stall_counter += 1

        # If everything appears delivered, hold still (cheap, and avoids
        # knocking cubes back out).
        if self.scene.cubes and delivered >= len(self.scene.cubes):
            return self._finish(self._blank())

        # Long stall -> switch to the sweep fallback permanently.
        if self.stall_counter > 700 and self.phase != "sweep":
            self.phase = "sweep"
            self.phase_t = 0

        if self.source is None or self.target is None:
            return self._finish(self._blank())

        src = _xyz(state, self.source)
        tgt = _xyz(state, self.target)
        a = self._blank()

        # ---------------- phase machine ----------------
        if self.phase == "settle":
            # Let the sim settle; hold home pose, gripper open.
            a[self.i_arm] = self._arm_delta_toward(state, self.HOME)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_open
            if self.phase_t >= self.T_SETTLE:
                self.phase, self.phase_t = "approach", 0

        elif self.phase == "approach":
            stand, yaw = self._standoff_pose(src[:2], tgt[:2])
            a[self.i_base] = self._base_delta_toward(state, stand, yaw)[
                : (self.i_base.stop - self.i_base.start)
            ]
            a[self.i_arm] = self._arm_delta_toward(state, self.HOME)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_open
            r = self.scene.robot
            close = False
            if r is not None:
                bx = _safe_get(state, r, "pos_base_x", 0.0)
                by = _safe_get(state, r, "pos_base_y", 0.0)
                close = float(np.hypot(stand[0] - bx, stand[1] - by)) < 0.05
            if close or self.phase_t >= self.T_APPROACH:
                self.phase, self.phase_t = "descend", 0

        elif self.phase == "descend":
            a[self.i_arm] = self._arm_delta_toward(state, self.REACH_DOWN)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_open
            if self.phase_t >= self.T_DESCEND:
                self.phase, self.phase_t = "grasp", 0

        elif self.phase == "grasp":
            a[self.i_arm] = self._arm_delta_toward(state, self.REACH_DOWN)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_closed
            if self.phase_t >= self.T_GRASP:
                self.phase, self.phase_t = "lift", 0

        elif self.phase == "lift":
            a[self.i_arm] = self._arm_delta_toward(state, self.CARRY)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_closed
            if self.phase_t >= self.T_LIFT:
                self.phase, self.phase_t = "transfer", 0

        elif self.phase == "transfer":
            # Move the base so the carried source bin ends up over the target.
            stand, yaw = self._standoff_pose(tgt[:2], src[:2])
            a[self.i_base] = self._base_delta_toward(state, stand, yaw)[
                : (self.i_base.stop - self.i_base.start)
            ]
            a[self.i_arm] = self._arm_delta_toward(state, self.CARRY)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_closed
            r = self.scene.robot
            close = False
            if r is not None:
                bx = _safe_get(state, r, "pos_base_x", 0.0)
                by = _safe_get(state, r, "pos_base_y", 0.0)
                close = float(np.hypot(stand[0] - bx, stand[1] - by)) < 0.06
            if close or self.phase_t >= self.T_TRANSFER:
                self.phase, self.phase_t = "pour", 0

        elif self.phase == "pour":
            # Rotate the wrist to tip the contents out over the target.
            a[self.i_arm] = self._arm_delta_toward(state, self.TIP)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_closed
            if self.phase_t >= self.T_POUR:
                self.phase, self.phase_t = "release", 0

        elif self.phase == "release":
            a[self.i_arm] = self._arm_delta_toward(state, self.CARRY)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_open
            if self.phase_t >= self.T_RELEASE:
                self.phase, self.phase_t = "sweep", 0

        else:  # "sweep" -- fallback / mop-up
            # Drive the base slowly back and forth along the source->target line
            # with the arm extended low, nudging stragglers toward the target.
            axis = tgt[:2] - src[:2]
            n = float(np.linalg.norm(axis))
            if n < 1e-6:
                axis = np.array([0.0, 1.0])
                n = 1.0
            u = axis / n
            perp = np.array([-u[1], u[0]])
            # Oscillate the goal point between just-behind-source and past-target.
            period = 240
            ph = (self.phase_t % period) / float(period)
            s = -0.35 + 1.5 * ph  # parametric position along the axis
            goal = src[:2] + u * (s * n) + perp * 0.55
            yaw = float(np.arctan2(-perp[1], -perp[0]))
            a[self.i_base] = self._base_delta_toward(state, goal, yaw)[
                : (self.i_base.stop - self.i_base.start)
            ]
            a[self.i_arm] = self._arm_delta_toward(state, self.REACH_DOWN)[
                : (self.i_arm.stop - self.i_arm.start)
            ]
            if self.i_grip is not None:
                a[self.i_grip] = self.grip_closed

        return self._finish(a)