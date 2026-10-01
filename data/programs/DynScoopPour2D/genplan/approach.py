"""GeneratedApproach for CountParameterizedDynScoopPour2DEnv.

Strategy (bulldozer / plow):
---------------------------
The termination condition is purely a position test: at least 50% of the small
objects must be on the right side of the middle wall.  Nothing requires the hook
to be used.  The middle wall is only half the world height, and the robot base is
kinematic (it is moved by prescribed increments, not by forces), so the robot can
travel over the top of the wall freely while the dynamic small objects cannot.

So we ignore the hook entirely and use the robot itself as a plow:

  1. Close the gripper and extend the arm to make a broad rigid paddle.
  2. Rotate the paddle so it points downward (theta ~ -pi/2), giving a wide,
     low face that reaches down to the objects' level.
  3. Fly (above the pile) to a spot to the LEFT of the objects that still need
     moving, then descend to their level.
  4. Sweep rightward at maximum dx, pushing the cluster toward the wall.
  5. Near the wall, keep pressing right while lifting, so the paddle rides up the
     wall face and flings the bunched objects over the top.
  6. Retract (lift high, fly back left) and repeat, re-reading the state each
     cycle, until enough objects are on the right.

Everything is recomputed from the state each cycle, so the number of objects,
the mix of circles and squares, and their names are all irrelevant: we simply
look at whatever `small_*` objects exist.

Numerical safety note (bug fix):
--------------------------------
The action space bounds are float64 while the returned action is float32.  A
value that exactly equals a bound in float64 (e.g. dtheta = 0.09817477...) can
round *up* when cast to float32 and then fall outside the float64 box, which the
environment rejects.  Every returned action is therefore built in float64,
clamped to the declared bounds shrunk by a small relative epsilon, cast to
float32, and then re-checked/nudged in float32 so the final float32 values are
provably inside the float64 box.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _clamp(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value < low:
        return low
    if value > high:
        return high
    return value


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _safe_get(state: Any, obj: Any, feature: str, default: float = 0.0) -> float:
    """Read a feature, tolerating types that lack it."""
    try:
        value = float(state.get(obj, feature))
    except Exception:  # pylint: disable=broad-except
        return default
    if not np.isfinite(value):
        return default
    return value


def _has_feature(state: Any, obj: Any, feature: str) -> bool:
    try:
        feats = state.type_features[obj.type]
    except Exception:  # pylint: disable=broad-except
        return False
    return feature in feats


class GeneratedApproach:
    """Plow the small objects over the middle wall with the robot body."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        default_low = np.array([-0.03, -0.03, -0.098, -0.08, -0.015], dtype=np.float64)
        default_high = np.array([0.03, 0.03, 0.098, 0.08, 0.015], dtype=np.float64)

        low = np.asarray(getattr(action_space, "low", default_low), dtype=np.float64)
        high = np.asarray(getattr(action_space, "high", default_high), dtype=np.float64)

        # Determine the action dimensionality from the space when possible.
        shape = getattr(action_space, "shape", None)
        if shape is not None and len(shape) >= 1:
            self._dim = int(shape[0])
        else:
            self._dim = int(min(len(low), len(high)))
        if self._dim <= 0:
            self._dim = 5

        # Pad/trim bounds to the action dimension.
        if len(low) < self._dim:
            low = np.concatenate([low, np.full(self._dim - len(low), -np.inf)])
        if len(high) < self._dim:
            high = np.concatenate([high, np.full(self._dim - len(high), np.inf)])
        self._low = low[: self._dim].astype(np.float64)
        self._high = high[: self._dim].astype(np.float64)

        # Shrink the usable box slightly so that a float64 -> float32 cast can
        # never push a saturated value outside the declared bounds.  The shrink
        # is relative (plus a tiny absolute floor) so it is negligible in
        # magnitude but always larger than one float32 ULP at the bound.
        eps = 1e-6
        span = np.abs(self._high - self._low)
        margin = np.maximum(eps * np.maximum(np.abs(self._low), np.abs(self._high)),
                            1e-9)
        margin = np.minimum(margin, 0.25 * np.where(np.isfinite(span), span, 1.0))
        self._safe_low = self._low + margin
        self._safe_high = self._high - margin
        # Guard against degenerate (zero-width) dimensions.
        bad = self._safe_low > self._safe_high
        if np.any(bad):
            mid = 0.5 * (self._low + self._high)
            self._safe_low = np.where(bad, mid, self._safe_low)
            self._safe_high = np.where(bad, mid, self._safe_high)

        def _limit(idx: int, fallback: float) -> float:
            if idx >= self._dim:
                return fallback
            lo = self._safe_low[idx]
            hi = self._safe_high[idx]
            if not np.isfinite(lo) or not np.isfinite(hi):
                return fallback
            return float(min(abs(lo), abs(hi)))

        self.max_dx = _limit(0, 0.03)
        self.max_dy = _limit(1, 0.03)
        self.max_dth = _limit(2, 0.098)
        self.max_darm = _limit(3, 0.08)
        self.max_dgrip = _limit(4, 0.015)

        # Episode-level cached geometry (set in reset()).
        self._wall_x: float = 2.0
        self._wall_top: float = 1.2
        self._world_top: float = 2.6
        self._floor_y: float = 0.0
        self._min_x: float = 0.0
        self._robot_arm_max: float = 0.4
        self._robot_base_radius: float = 0.2
        self._reach: float = 0.6

        # Runtime plan state.
        self._phase: str = "prep"
        self._phase_steps: int = 0
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._sweep_start_x: float = 0.0
        self._cycle: int = 0
        self._step_count: int = 0
        self._sweep_depth_idx: int = 0
        self._last_progress_step: int = 0
        self._last_right_count: int = -1

    # ------------------------------------------------------------------ #
    # State reading utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _small_objects(state: Any) -> List[Any]:
        """Every count-defining small object, whatever its concrete type."""
        out = []
        for name in state.get_object_names():
            if name.startswith("small"):
                out.append(state.get_object_from_name(name))
        return out

    @staticmethod
    def _robot(state: Any) -> Optional[Any]:
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            if obj.type.name == "kin_robot":
                return obj
        try:
            return state.get_object_from_name("robot")
        except Exception:  # pylint: disable=broad-except
            return None

    @staticmethod
    def _hooks(state: Any) -> List[Any]:
        out = []
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            if obj.type.name in ("hook", "lobject", "tobject"):
                out.append(obj)
        return out

    def _object_extent(self, state: Any, obj: Any) -> float:
        """Approximate half-extent of a small object."""
        if _has_feature(state, obj, "radius"):
            return _safe_get(state, obj, "radius", 0.08)
        if _has_feature(state, obj, "size"):
            return 0.5 * _safe_get(state, obj, "size", 0.16) * 1.4142
        return 0.08

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #
    def reset(self, state, info):
        self._phase = "prep"
        self._phase_steps = 0
        self._cycle = 0
        self._step_count = 0
        self._sweep_depth_idx = 0
        self._last_progress_step = 0
        self._last_right_count = -1

        robot = self._robot(state)
        if robot is not None:
            self._robot_base_radius = _safe_get(state, robot, "base_radius", 0.2)
            self._robot_arm_max = _safe_get(state, robot, "arm_length", 0.4)
            gb_h = _safe_get(state, robot, "gripper_base_height", 0.25)
            fh = _safe_get(state, robot, "finger_height", 0.04)
            # How far the tip of the (fully extended) tool reaches from base center.
            self._reach = (self._robot_base_radius + self._robot_arm_max
                           + gb_h + fh)
            robot_y = _safe_get(state, robot, "y", 2.2)
        else:
            self._reach = 0.6
            robot_y = 2.2

        smalls = self._small_objects(state)
        small_xs = [_safe_get(state, o, "x", 0.0) for o in smalls]
        small_ys = [_safe_get(state, o, "y", 0.0) for o in smalls]

        hooks = self._hooks(state)
        hook_xs = [_safe_get(state, h, "x", 3.0) for h in hooks]

        # --- infer the wall x ------------------------------------------------
        # Small objects start entirely on the left of the wall; the hook starts
        # on the right.  Put the wall between the rightmost small object and the
        # hook, closer to the pile so our sweep actually reaches the wall face.
        if small_xs:
            pile_right = max(small_xs)
            pile_left = min(small_xs)
        else:
            pile_right, pile_left = 1.6, 0.2

        if hook_xs:
            hook_left = min(hook_xs)
        else:
            hook_left = pile_right + 1.0

        if hook_left > pile_right:
            wall_x = pile_right + 0.45 * (hook_left - pile_right)
        else:
            wall_x = pile_right + 0.4
        self._wall_x = wall_x

        # --- infer vertical geometry ----------------------------------------
        # The robot starts above the wall; the wall is half the world height.
        self._world_top = max(2.4, robot_y + 0.4)
        self._wall_top = 0.5 * self._world_top
        if small_ys:
            self._floor_y = max(0.0, min(small_ys) - 0.1)
        else:
            self._floor_y = 0.0
        self._min_x = max(0.05, (pile_left - 0.6))

        return None

    # ------------------------------------------------------------------ #
    # Goal bookkeeping
    # ------------------------------------------------------------------ #
    def _counts(self, state: Any) -> Tuple[int, int, List[Any]]:
        """(#right, #total, list of small objects still on the left)."""
        smalls = self._small_objects(state)
        left: List[Any] = []
        right = 0
        for obj in smalls:
            x = _safe_get(state, obj, "x", 0.0)
            if x > self._wall_x:
                right += 1
            else:
                left.append(obj)
        return right, len(smalls), left

    # ------------------------------------------------------------------ #
    # Action construction
    # ------------------------------------------------------------------ #
    def _action(self, dx: float, dy: float, dtheta: float,
                darm: float, dgrip: float) -> np.ndarray:
        """Build a float32 action guaranteed to lie inside the float64 box."""
        raw = [dx, dy, dtheta, darm, dgrip]
        vals = np.zeros(self._dim, dtype=np.float64)
        for i in range(min(self._dim, len(raw))):
            vals[i] = raw[i]

        # Clamp in float64 against the shrunken (safe) bounds.
        for i in range(self._dim):
            lo = self._safe_low[i]
            hi = self._safe_high[i]
            if not np.isfinite(lo):
                lo = -np.inf
            if not np.isfinite(hi):
                hi = np.inf
            v = vals[i]
            if not np.isfinite(v):
                v = 0.0
            vals[i] = _clamp(float(v), float(lo), float(hi))

        out = vals.astype(np.float32)

        # Re-verify in float32: the cast can round outward.  Nudge any offending
        # component toward zero (in float32 steps) until it is inside the box.
        for i in range(self._dim):
            lo = float(self._low[i])
            hi = float(self._high[i])
            for _ in range(64):
                v = float(out[i])
                if v < lo:
                    out[i] = np.nextafter(out[i], np.float32(hi))
                elif v > hi:
                    out[i] = np.nextafter(out[i], np.float32(lo))
                else:
                    break
            # Last-resort fallback: a value strictly inside the box.
            v = float(out[i])
            if v < lo or v > hi:
                mid = 0.5 * (lo + hi)
                out[i] = np.float32(mid if np.isfinite(mid) else 0.0)
        return out

    def _drive_theta(self, state: Any, robot: Any, desired: float) -> float:
        """Proportional angular step toward a desired absolute heading."""
        theta = _safe_get(state, robot, "theta", 0.0)
        err = _wrap_angle(desired - theta)
        return _clamp(err, -self.max_dth, self.max_dth)

    # ------------------------------------------------------------------ #
    # Main policy
    # ------------------------------------------------------------------ #
    def get_action(self, state):
        self._step_count += 1
        robot = self._robot(state)
        if robot is None:
            return self._action(0.0, 0.0, 0.0, 0.0, 0.0)

        rx = _safe_get(state, robot, "x", 0.0)
        ry = _safe_get(state, robot, "y", 0.0)
        arm = _safe_get(state, robot, "arm_joint", 0.0)
        grip = _safe_get(state, robot, "finger_gap", 0.12)

        right_count, total, left_objs = self._counts(state)

        # Track progress so we can vary the sweep if we stall.
        if right_count != self._last_right_count:
            self._last_right_count = right_count
            self._last_progress_step = self._step_count

        # If (somehow) everything is already across, just hold still.
        if total > 0 and right_count * 2 >= total:
            return self._action(0.0, 0.0, 0.0, 0.0, 0.0)

        # Paddle configuration: arm fully out, gripper fully closed, tool
        # pointing straight down so it presents a broad low face.
        darm = _clamp(self._robot_arm_max - arm, -self.max_darm, self.max_darm)
        dgrip = -self.max_dgrip if grip > 1e-4 else 0.0
        down = -math.pi / 2.0

        self._phase_steps += 1

        # ---------------------------------------------------------------- #
        # PHASE: prep -- get the paddle formed while still high up.
        # ---------------------------------------------------------------- #
        if self._phase == "prep":
            dth = self._drive_theta(state, robot, down)
            ready = (abs(self._robot_arm_max - arm) < 0.02
                     and grip <= 1e-3
                     and abs(dth) < 1e-3)
            target_high = self._wall_top + 0.45
            dy = _clamp(target_high - ry, -self.max_dy, self.max_dy)
            if ready or self._phase_steps > 60:
                self._begin_cycle(state, left_objs)
            return self._action(0.0, dy, dth, darm, dgrip)

        # ---------------------------------------------------------------- #
        # PHASE: goto_start -- fly (high) to the left of the remaining pile.
        # ---------------------------------------------------------------- #
        if self._phase == "goto_start":
            dth = self._drive_theta(state, robot, down)
            travel_y = self._wall_top + 0.45
            dy = _clamp(travel_y - ry, -self.max_dy, self.max_dy)
            dx = _clamp(self._sweep_start_x - rx, -self.max_dx, self.max_dx)
            # Only move horizontally once we are safely above the wall, so we
            # never clip through it on the way back left.
            if ry < self._wall_top + 0.25 and rx > self._wall_x - 0.4:
                dx = 0.0
            arrived = (abs(self._sweep_start_x - rx) < 0.03
                       and abs(travel_y - ry) < 0.05)
            if arrived or self._phase_steps > 300:
                self._phase = "descend"
                self._phase_steps = 0
            return self._action(dx, dy, dth, darm, dgrip)

        # ---------------------------------------------------------------- #
        # PHASE: descend -- drop the paddle down to the objects' level.
        # ---------------------------------------------------------------- #
        if self._phase == "descend":
            dth = self._drive_theta(state, robot, down)
            dy = _clamp(self._target_y - ry, -self.max_dy, self.max_dy)
            dx = _clamp(self._sweep_start_x - rx, -self.max_dx, self.max_dx)
            arrived = abs(self._target_y - ry) < 0.03
            if arrived or self._phase_steps > 200:
                self._phase = "sweep"
                self._phase_steps = 0
            return self._action(dx, dy, dth, darm, dgrip)

        # ---------------------------------------------------------------- #
        # PHASE: sweep -- push right along the floor toward the wall.
        # ---------------------------------------------------------------- #
        if self._phase == "sweep":
            dth = self._drive_theta(state, robot, down)
            dx = self.max_dx
            dy = _clamp(self._target_y - ry, -self.max_dy, self.max_dy)
            near_wall = rx > self._wall_x - (0.5 * self._reach + 0.10)
            if near_wall or self._phase_steps > 400:
                self._phase = "lift"
                self._phase_steps = 0
            return self._action(dx, dy, dth, darm, dgrip)

        # ---------------------------------------------------------------- #
        # PHASE: lift -- keep pressing right while riding up the wall face,
        # flinging the bunched objects over the top.
        # ---------------------------------------------------------------- #
        if self._phase == "lift":
            dth = self._drive_theta(state, robot, down)
            dx = self.max_dx
            dy = self.max_dy
            # Retract the arm a little as we rise so the tool tip sweeps up and
            # over the wall crest rather than jamming into its side.
            tip_y = ry - self._reach
            if tip_y > self._wall_top - 0.05:
                darm_l = -self.max_darm
            else:
                darm_l = darm
            done = (ry > self._wall_top + self._reach + 0.15
                    or rx > self._wall_x + 0.35)
            if done or self._phase_steps > 250:
                self._phase = "carry_over"
                self._phase_steps = 0
            return self._action(dx, dy, dth, darm_l, dgrip)

        # ---------------------------------------------------------------- #
        # PHASE: carry_over -- nudge just past the wall so anything riding the
        # paddle is deposited on the right side, then go back for more.
        # ---------------------------------------------------------------- #
        if self._phase == "carry_over":
            dth = self._drive_theta(state, robot, down)
            target_x = self._wall_x + 0.45
            dx = _clamp(target_x - rx, -self.max_dx, self.max_dx)
            travel_y = self._wall_top + 0.45
            dy = _clamp(travel_y - ry, -self.max_dy, self.max_dy)
            done = abs(target_x - rx) < 0.05
            if done or self._phase_steps > 150:
                self._cycle += 1
                self._begin_cycle(state, left_objs)
            return self._action(dx, dy, dth, darm, dgrip)

        # Unknown phase -> restart a cycle.
        self._begin_cycle(state, left_objs)
        return self._action(0.0, 0.0, 0.0, darm, dgrip)

    # ------------------------------------------------------------------ #
    # Cycle planning
    # ------------------------------------------------------------------ #
    def _begin_cycle(self, state: Any, left_objs: List[Any]) -> None:
        """Choose where to start the next sweep, based on what is still left."""
        self._phase = "goto_start"
        self._phase_steps = 0

        if left_objs:
            xs = [_safe_get(state, o, "x", 0.0) for o in left_objs]
            ys = [_safe_get(state, o, "y", 0.0) for o in left_objs]
            extents = [self._object_extent(state, o) for o in left_objs]
            leftmost = min(xs)
            typical_extent = float(np.median(extents)) if extents else 0.08
            # Sweep height: keep the paddle tip just above the floor so it
            # catches the objects rather than riding over them.
            base_tip_y = min(ys) - 0.5 * typical_extent
        else:
            leftmost = self._min_x + 0.2
            typical_extent = 0.08
            base_tip_y = self._floor_y + 0.05

        # Start comfortably to the left of the leftmost remaining object so the
        # paddle gets behind the cluster before pushing.
        start_x = leftmost - (0.5 * self._reach + 3.0 * typical_extent + 0.10)
        self._sweep_start_x = max(self._min_x, start_x)

        # Alternate the sweep depth between cycles: a low pass catches things
        # resting on the floor, a slightly higher pass catches anything that
        # ended up stacked.  If we have stalled, vary it more aggressively.
        stalled = (self._step_count - self._last_progress_step) > 400
        depths = [0.02, 0.10, 0.00, 0.16]
        if stalled:
            self._sweep_depth_idx += 1
        idx = (self._cycle + self._sweep_depth_idx) % len(depths)
        tip_y = max(self._floor_y + 0.01, base_tip_y + depths[idx])

        # Convert a desired tool-tip height into a base height.
        self._target_y = tip_y + self._reach
        # Never ask the base to go below its own radius (it would clip the floor).
        self._target_y = max(self._target_y,
                             self._floor_y + self._robot_base_radius + 0.02)
        self._target_x = self._sweep_start_x