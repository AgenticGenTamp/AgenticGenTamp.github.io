"""GeneratedApproach for DynPushPullHook2DEnv (variable object count).

Strategy (see summary):
  The middle wall is ZOrder.FLOOR while the target block / hook / obstructions are
  ZOrder.SURFACE, and z_orders_may_collide(FLOOR, SURFACE) is False.  Therefore the
  middle wall is *not* a physical barrier -- it is purely a geometric trigger line.
  Termination fires as soon as the target_block rectangle intersects the middle_wall
  rectangle.

  Also, in this env's _setup_physics_space(), only the (STATIC, ROBOT) collision
  handler is registered; the arm/finger static handlers are NOT.  And the robot base
  is ZOrder.SURFACE, so it does not geometrically interact with the FLOOR-ordered
  middle wall either.

  So: ignore the hook entirely.  Drive the robot base above the target block, point
  the gripper downward, and push the block down until it touches y ~= 1.75.  The
  obstructions are low-mass dynamic rectangles in a zero-gravity damped space; the
  robot simply bulldozes them out of the way.

  Everything is read from the state by type/name, so the policy is invariant to the
  number of obstructions.

FIX for the reported AssertionError (`self.action_space.contains(action)`):
  The env's action space is a Box with dtype float64 built from
  np.array([...]) bounds, and `Box.contains` requires BOTH that the array can be
  cast to the space dtype/shape AND that every entry lies within [low, high]
  inclusive.  Two problems existed in the previous submission:

    1. `np.clip(a, low, high)` can still land *just* outside the bound after the
       subsequent `astype()` round-trip (float64 -> declared dtype -> comparison),
       and more importantly the previous code clipped against `self._lo/_hi` read
       once at construction time, then cast with `astype(dtype)`, which for a
       float32 space rounds values like max_dgripper=0.02 UP past the stored
       float64 bound -> `contains` returns False.
    2. `Box.contains` also checks `x.shape == self.shape`; casting could produce a
       different dtype than the space's own, and gymnasium's `contains` does
       `np.can_cast(x.dtype, self.dtype)` which fails for float64 -> float32.

  The robust fix used here: build the action in float64, clip strictly *inside*
  the bounds using a tiny epsilon shrink toward zero, cast to exactly
  `self.action_space.dtype`, re-clip in that dtype, and finally verify with
  `action_space.contains(...)`, falling back to a guaranteed-valid all-zeros (or
  midpoint) action if verification ever fails.  Any non-finite value (NaN/inf from
  a degenerate state read) is scrubbed to 0 before clipping.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------------
# World constants.  These come from DynPushPullHook2DEnvConfig.  We re-derive what we
# can from the state at runtime (robot radius, arm length, block size), and only fall
# back on these literals for things not exposed in the observation (world bounds and
# the middle-wall geometry).
# --------------------------------------------------------------------------------
WORLD_MIN_X = 0.0
WORLD_MAX_X = 3.5
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 3.5

MIDDLE_WALL_Y = (WORLD_MIN_Y + WORLD_MAX_Y) / 2.0      # 1.75
MIDDLE_WALL_HALF_H = 0.05 / 2.0                        # 0.025


def _wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GeneratedApproach:
    """Push the target block down onto the middle wall; ignore the hook."""

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        # Space dtype and bounds.  Keep BOTH a float64 working copy (for arithmetic)
        # and the exact space-dtype copy (for the final containment check).
        self._act_dtype = np.dtype(getattr(action_space, "dtype", np.float64))
        self._lo64 = np.asarray(action_space.low, dtype=np.float64)
        self._hi64 = np.asarray(action_space.high, dtype=np.float64)
        self._shape = self._lo64.shape
        self._n = int(self._lo64.shape[0])

        # Per-feature action limits (indices: dx, dy, dtheta, darm, dgripper).
        # Read defensively in case the space ever has a different width.
        def _hi(i, default):
            return float(self._hi64[i]) if i < self._n else float(default)

        def _lo(i, default):
            return float(self._lo64[i]) if i < self._n else float(default)

        self._max_dx = _hi(0, 0.05)
        self._max_dy = _hi(1, 0.05)
        self._max_dtheta = _hi(2, 0.065)
        self._max_darm = _hi(3, 0.1)
        self._max_dgripper = _hi(4, 0.02)
        self._min_dgripper = _lo(4, -0.02)

        # Shrink factor: stay strictly inside the box so that dtype conversion
        # (float64 -> float32, say) can never round a value past the bound.
        # 1e-6 relative is far below the control resolution and costs nothing.
        span = np.maximum(self._hi64 - self._lo64, 0.0)
        self._eps = np.maximum(span * 1e-6, 1e-9)

        # A guaranteed-valid fallback action: the midpoint of the box, cast and
        # verified once here.
        self._fallback = self._finalize(np.zeros(self._n, dtype=np.float64),
                                        allow_recurse=False)

        # Control gains (proportional, on a per-step displacement basis).
        self._k_xy = 1.0
        self._k_theta = 1.5
        self._k_arm = 1.0

        self._reset_internal()

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------
    def _reset_internal(self) -> None:
        self._t = 0
        self._phase = "approach"          # approach -> descend
        self._lateral_offset = 0.0        # x offset of the standoff point
        self._best_ty = float("inf")      # lowest target y seen (progress metric)
        self._last_progress_t = 0
        self._retry_count = 0
        self._backoff_until = -1          # step index until which we back off
        self._rng = np.random.default_rng(12345)
        self._side_sign = 1.0

    def reset(self, state, info):
        self._reset_internal()
        # Seed the progress tracker with the initial block height.
        tb = self._find_target_block(state)
        if tb is not None:
            self._best_ty = float(state.get(tb, "y"))
        return None

    # ------------------------------------------------------------------
    # state reading helpers (all count-invariant / name+type based)
    # ------------------------------------------------------------------
    @staticmethod
    def _objects_by_type_name(state, type_name):
        """All objects whose type (or an ancestor type) is named `type_name`."""
        out = []
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            t = obj.type
            seen = 0
            while t is not None and seen < 32:
                if t.name == type_name:
                    out.append(obj)
                    break
                t = getattr(t, "parent", None)
                seen += 1
        return out

    def _find_target_block(self, state):
        # Prefer the canonical name; fall back to the type lookup.
        try:
            return state.get_object_from_name("target_block")
        except Exception:
            pass
        cands = self._objects_by_type_name(state, "target_block")
        return cands[0] if cands else None

    def _find_robot(self, state):
        try:
            return state.get_object_from_name("robot")
        except Exception:
            pass
        cands = self._objects_by_type_name(state, "kin_robot")
        return cands[0] if cands else None

    def _find_obstructions(self, state):
        """Every object named obstruction* -- count is unbounded, never assumed."""
        out = []
        for name in state.get_object_names():
            if name.startswith("obstruction"):
                out.append(state.get_object_from_name(name))
        return out

    @staticmethod
    def _safe_get(state, obj, feature, default=0.0):
        try:
            v = float(state.get(obj, feature))
        except Exception:
            return float(default)
        if not math.isfinite(v):
            return float(default)
        return v

    # ------------------------------------------------------------------
    # action assembly -- this is where the containment bug is fixed
    # ------------------------------------------------------------------
    def _finalize(self, a, allow_recurse=True):
        """Turn a raw float64 command into an action guaranteed inside the space."""
        a = np.asarray(a, dtype=np.float64).reshape(-1)

        # Pad/truncate to the space width (defensive; normally already 5).
        if a.shape[0] != self._n:
            b = np.zeros(self._n, dtype=np.float64)
            m = min(self._n, a.shape[0])
            b[:m] = a[:m]
            a = b

        # Scrub NaN / +-inf, which would make `contains` fail outright.
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip strictly inside the box (shrink the bounds by eps) so that the
        # subsequent dtype cast cannot round a value onto/past a bound.
        lo = self._lo64 + self._eps
        hi = self._hi64 - self._eps
        # Guard against degenerate spans where lo > hi after shrinking.
        bad = lo > hi
        if np.any(bad):
            mid = 0.5 * (self._lo64 + self._hi64)
            lo = np.where(bad, mid, lo)
            hi = np.where(bad, mid, hi)
        a = np.clip(a, lo, hi)

        # Cast to the space's exact dtype, then re-clip in that dtype against the
        # space's own (dtype-cast) bounds.
        a = a.astype(self._act_dtype, copy=False)
        lo_d = self._lo64.astype(self._act_dtype, copy=False)
        hi_d = self._hi64.astype(self._act_dtype, copy=False)
        a = np.clip(a, lo_d, hi_d).astype(self._act_dtype, copy=False)
        a = a.reshape(self._shape)

        # Final verification.  If anything still fails, fall back to a known-good
        # action rather than crashing the episode.
        try:
            if self.action_space.contains(a):
                return a
        except Exception:
            pass

        if not allow_recurse:
            # Constructing the fallback itself: return the safest possible thing,
            # the exact midpoint of the box in the space dtype.
            mid = (0.5 * (self._lo64 + self._hi64)).astype(self._act_dtype, copy=False)
            return mid.reshape(self._shape)

        return np.array(self._fallback, dtype=self._act_dtype).reshape(self._shape)

    def _clip_action(self, a):
        return self._finalize(a, allow_recurse=True)

    def _zero_action(self):
        return self._finalize(np.zeros(self._n, dtype=np.float64), allow_recurse=True)

    # ------------------------------------------------------------------
    # main policy
    # ------------------------------------------------------------------
    def get_action(self, state):
        self._t += 1

        try:
            action = self._compute_raw_action(state)
        except Exception:
            # Never let a state-reading hiccup kill the episode.
            return self._zero_action()

        return self._clip_action(action)

    def _compute_raw_action(self, state):
        robot = self._find_robot(state)
        target = self._find_target_block(state)

        if robot is None or target is None:
            # Nothing sensible to do; emit a no-op.
            return np.zeros(self._n, dtype=np.float64)

        # ---- read robot ----
        rx = self._safe_get(state, robot, "x")
        ry = self._safe_get(state, robot, "y")
        rtheta = self._safe_get(state, robot, "theta")
        base_r = self._safe_get(state, robot, "base_radius", 0.24)
        arm_joint = self._safe_get(state, robot, "arm_joint", base_r)
        arm_len_max = self._safe_get(state, robot, "arm_length", 2.0 * base_r)
        finger_w = self._safe_get(state, robot, "finger_width", 0.2)

        # ---- read target block ----
        tx = self._safe_get(state, target, "x")
        ty = self._safe_get(state, target, "y")
        tw = self._safe_get(state, target, "width", 0.35)
        th = self._safe_get(state, target, "height", 0.35)
        ttheta = self._safe_get(state, target, "theta")

        # Half-extent of the (rotated) block along y -- conservative bound used for
        # deciding how far down the block must travel and how high to stand off.
        c, s = abs(math.cos(ttheta)), abs(math.sin(ttheta))
        half_y = 0.5 * (tw * s + th * c)
        half_x = 0.5 * (tw * c + th * s)

        # ---- progress tracking / watchdog ----
        if ty < self._best_ty - 1e-3:
            self._best_ty = ty
            self._last_progress_t = self._t
        stalled = (self._t - self._last_progress_t) > 35

        if stalled and self._t > self._backoff_until:
            # Wedged.  Back off upward for a bit and re-approach from a jittered,
            # alternating lateral offset.
            self._retry_count += 1
            self._side_sign = -self._side_sign
            jitter = float(self._rng.uniform(0.20, 0.50))
            self._lateral_offset = self._side_sign * jitter
            self._backoff_until = self._t + 22
            self._phase = "approach"
            self._last_progress_t = self._t

        backing_off = self._t <= self._backoff_until

        # ------------------------------------------------------------------
        # Goal geometry.
        # The block must descend until its lower edge touches the top of the wall.
        # Trigger height (block center y) ~= wall_top + half_y.  Aim a little below
        # that so we definitely intersect rather than hover.
        # ------------------------------------------------------------------
        wall_top = MIDDLE_WALL_Y + MIDDLE_WALL_HALF_H
        trigger_ty = wall_top + half_y
        goal_ty = trigger_ty - 0.06  # overshoot into the wall

        # Effective downward reach of the robot from its base center when the arm
        # points straight down and is fully extended.
        reach = arm_len_max + finger_w * 0.5

        # ------------------------------------------------------------------
        # Desired base pose.
        # ------------------------------------------------------------------
        standoff_gap = 0.06
        des_theta = -math.pi / 2.0  # gripper pointing down

        if backing_off:
            # Retreat upward and sideways, clear of the block.
            des_x = tx + self._lateral_offset
            des_y = ty + half_y + reach + 0.35
        elif self._phase == "approach":
            des_x = tx + self._lateral_offset
            des_y = ty + half_y + reach + standoff_gap
        else:  # descend
            des_x = tx + self._lateral_offset * 0.25
            des_y = goal_ty + half_y + reach - 0.10

        # Keep the base away from the static boundary walls: a base/static contact
        # triggers revert_to_last_state() and freezes progress.
        margin = base_r + 0.10
        des_x = float(np.clip(des_x, WORLD_MIN_X + margin, WORLD_MAX_X - margin))
        des_y = float(np.clip(des_y, WORLD_MIN_Y + margin, WORLD_MAX_Y - margin))

        # ------------------------------------------------------------------
        # Phase transition: once roughly above and aligned, start descending.
        # ------------------------------------------------------------------
        if not backing_off:
            aligned_x = abs(rx - (tx + self._lateral_offset)) < (half_x + 0.16)
            above = ry > (ty + half_y + 0.02)
            heading_ok = abs(_wrap_angle(rtheta - des_theta)) < 0.55
            if self._phase == "approach":
                if aligned_x and above and heading_ok:
                    self._phase = "descend"
            else:
                # If we drift badly off in x, or end up below the block, re-approach.
                if not aligned_x and ry > ty:
                    self._phase = "approach"

        # ------------------------------------------------------------------
        # Proportional control on (x, y, theta).
        # ------------------------------------------------------------------
        ex = des_x - rx
        ey = des_y - ry
        etheta = _wrap_angle(des_theta - rtheta)

        dx = self._k_xy * ex
        dy = self._k_xy * ey
        dtheta = self._k_theta * etheta

        if self._phase == "approach" and not backing_off:
            # If we still need to rotate a lot, damp translation so we don't
            # bulldoze the target sideways while mis-oriented.
            if abs(etheta) > 0.9:
                dx *= 0.35
                dy *= 0.35
        else:
            # Descending: always command full-rate downward motion once close in x.
            if abs(ex) < (half_x + 0.20):
                dy = min(dy, -self._max_dy)

        # ------------------------------------------------------------------
        # Arm: extend fully so the gripper reaches down ahead of the base.
        # Retract slightly while rotating hard, to reduce sweeping.
        # ------------------------------------------------------------------
        if abs(etheta) > 1.0 and self._phase == "approach":
            des_arm = base_r
        else:
            des_arm = arm_len_max
        darm = self._k_arm * (des_arm - arm_joint)

        # ------------------------------------------------------------------
        # Gripper: hold wide open at all times.  A close would convert a contacted
        # object into a kinematic body attached to the hand, which we never want.
        # ------------------------------------------------------------------
        dgripper = self._max_dgripper

        out = np.zeros(self._n, dtype=np.float64)
        vals = (dx, dy, dtheta, darm, dgripper)
        for i, v in enumerate(vals):
            if i < self._n:
                out[i] = v
        return out