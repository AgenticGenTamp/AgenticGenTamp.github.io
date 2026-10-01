"""Dynamo3D policy: push each obstacle chair into the goal region.

Correction from the previous attempt
------------------------------------
The prior policy assumed the goal was "robot reaches a spot on the floor" and
parked the base at the origin. The seed-0 rollout shows exactly that: the base
ended at (0.006, -0.007) with all velocities ~1e-17 (fully settled, commanding
zero), while the three chairs sat within a fraction of a millimetre of their
spawn poses. Reward was -0.01 * 1000 = -1000, i.e. the per-timestep penalty and
nothing else -- no +1.0 placement bonuses, no 10cm shaping bonus, ever.

That rules out the "robot reaches goal" reading. ``_check_goals`` evaluates
``on``/``in`` predicates, and the reward description is literal: +1.0 per object
placed within 5cm of its target, partial credit at 10cm. The count-defining
objects are the things that must be moved. So the task is: push every chair into
the goal region.

Where is the goal region? It is not exposed as an object in the observation
(the state holds only chairs plus the robot). What we do know:
  * chairs spawn clustered out at x ~ 2.4-3.9, y ~ 2.5-4.0 (seed 0), and at
    x ~ 1.0, y ~ 0.18 (seed 1, single chair);
  * the robot spawns near the origin-ish region and the ground regions in this
    family are laid out about the origin;
  * ``robot_ground_clearance`` keeps the base clear of chairs at spawn.
The consistent reading is that the chairs start scattered on the ground and must
be gathered to a target region near the world origin. So: drive each chair
toward the origin by pushing it from the far side.

Strategy (count-agnostic)
-------------------------
Repeat until every chair is close enough to the target:
  1. Pick the chair currently farthest from its target (greedy, deterministic).
  2. Compute the "staging" pose: a point behind the chair on the line from the
     target through the chair, offset by the chair half-width plus base radius.
  3. Drive the base to that staging pose while keeping clear of the chair
     (approach via an arc so we do not bulldoze it sideways on the way in).
  4. Once staged, drive forward through the chair toward the target, which
     pushes it.
  5. When that chair is placed, back off and move to the next one.

Nothing here indexes a fixed chair count: the chair set is enumerated from the
state every step by duck-typing on the movable-object features, and the loop
works for 1 chair or 12 or any held-out count.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Wrap an angle into [-pi, pi]."""
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _safe_get(state: Any, obj: Any, feature: str, default: float = 0.0) -> float:
    try:
        return float(state.get(obj, feature))
    except Exception:
        return float(default)


def _has_features(state: Any, obj: Any, features: Sequence[str]) -> bool:
    for f in features:
        try:
            state.get(obj, f)
        except Exception:
            return False
    return True


class GeneratedApproach:
    """Push every movable chair into the goal region near the world origin."""

    _ROBOT_BASE_FEATURES = ("pos_base_x", "pos_base_y", "pos_base_rot")
    _MOVABLE_FEATURES = ("x", "y", "z", "bb_x", "bb_y", "bb_z")

    # Target for the chairs. The ground goal region in this family is centred
    # on the world origin; 5cm tolerance is the scoring radius.
    _TARGET = (0.0, 0.0)
    _PLACE_TOL = 0.04          # aim inside the 5cm scoring radius
    _RELEASE_TOL = 0.045       # stop pushing once this close

    # Geometry. Chair bb is 0.5 x 0.5 x 0.9; the TidyBot base is ~0.35 radius.
    _BASE_RADIUS = 0.33
    _STAGE_MARGIN = 0.10       # extra gap behind the chair when staging
    _CLEARANCE = 0.62          # keep this far from a chair centre when transiting

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        self._act_dim = self._infer_action_dim(action_space)
        self._low, self._high = self._action_bounds(action_space, self._act_dim)

        # Per-step delta clamps (10 Hz control, deltas).
        self._max_lin_step = 0.055
        self._max_yaw_step = 0.12

        self._k_lin = 1.0
        self._k_yaw = 1.0

        # Episode state.
        self._robot_name: Optional[str] = None
        self._step = 0
        self._phase = "approach"       # "approach" | "push" | "backoff"
        self._target_chair: Optional[str] = None
        self._phase_ticks = 0
        self._backoff_ticks = 0

        # Stall handling.
        self._last_xy: Optional[Tuple[float, float]] = None
        self._stall = 0
        self._nudge_ticks = 0
        self._nudge_sign = 1.0

        # Progress watchdog per chair, so we abandon a hopeless chair.
        self._chair_effort: Dict[str, int] = {}
        self._chair_best: Dict[str, float] = {}
        self._abandoned: set = set()

    # ------------------------------------------------------------------
    # Action-space introspection
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_action_dim(action_space) -> int:
        shape = getattr(action_space, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0]:
            return int(shape[0])
        low = getattr(action_space, "low", None)
        if low is not None:
            return int(np.asarray(low).ravel().shape[0])
        return 11

    @staticmethod
    def _action_bounds(action_space, dim: int):
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is None or high is None:
            return (
                np.full(dim, -np.inf, dtype=np.float64),
                np.full(dim, np.inf, dtype=np.float64),
            )
        low = np.asarray(low, dtype=np.float64).ravel()
        high = np.asarray(high, dtype=np.float64).ravel()
        if low.shape[0] != dim:
            low = np.resize(low, dim)
            high = np.resize(high, dim)
        return low, high

    # ------------------------------------------------------------------
    # State parsing (count-agnostic)
    # ------------------------------------------------------------------

    def _find_robot(self, state) -> Optional[Any]:
        if self._robot_name is not None:
            try:
                obj = state.get_object_from_name(self._robot_name)
                if _has_features(state, obj, self._ROBOT_BASE_FEATURES):
                    return obj
            except Exception:
                pass

        for type_name in ("mujoco_tidybot_robot", "mujoco_fr3_robot"):
            try:
                typ = self.observation_space.get_type(type_name)
                objs = list(state.get_objects(typ))
                if objs:
                    self._robot_name = objs[0].name
                    return objs[0]
            except Exception:
                pass

        try:
            names = list(state.get_object_names())
        except Exception:
            names = []
        for name in names:
            try:
                obj = state.get_object_from_name(name)
            except Exception:
                continue
            if _has_features(state, obj, self._ROBOT_BASE_FEATURES):
                self._robot_name = name
                return obj
        return None

    def _robot_pose(self, state) -> Optional[Tuple[float, float, float]]:
        obj = self._find_robot(state)
        if obj is None:
            return None
        return (
            _safe_get(state, obj, "pos_base_x"),
            _safe_get(state, obj, "pos_base_y"),
            _safe_get(state, obj, "pos_base_rot"),
        )

    def _chairs(self, state) -> Dict[str, Tuple[float, float, float, float]]:
        """name -> (x, y, half_x, half_y) for every movable object present.

        Enumerated fresh each step; no fixed count assumed anywhere.
        """
        out: Dict[str, Tuple[float, float, float, float]] = {}
        try:
            names = list(state.get_object_names())
        except Exception:
            return out
        for name in names:
            try:
                obj = state.get_object_from_name(name)
            except Exception:
                continue
            if _has_features(state, obj, self._ROBOT_BASE_FEATURES):
                continue
            if not _has_features(state, obj, self._MOVABLE_FEATURES):
                continue
            out[name] = (
                _safe_get(state, obj, "x"),
                _safe_get(state, obj, "y"),
                0.5 * abs(_safe_get(state, obj, "bb_x", 0.5)),
                0.5 * abs(_safe_get(state, obj, "bb_y", 0.5)),
            )
        return out

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(ax - bx, ay - by)

    def _pick_chair(self, chairs: Dict[str, Tuple[float, float, float, float]]
                    ) -> Optional[str]:
        """Farthest unplaced, unabandoned chair (deterministic tie-break)."""
        tx, ty = self._TARGET
        best_name = None
        best_d = -1.0
        for name in sorted(chairs):
            if name in self._abandoned:
                continue
            cx, cy, _, _ = chairs[name]
            d = self._dist(cx, cy, tx, ty)
            if d <= self._RELEASE_TOL:
                continue
            if d > best_d:
                best_d = d
                best_name = name

        if best_name is not None:
            return best_name

        # Everything abandoned but not everything placed: retry the closest.
        candidates = [
            (self._dist(chairs[n][0], chairs[n][1], tx, ty), n)
            for n in sorted(chairs)
            if self._dist(chairs[n][0], chairs[n][1], tx, ty) > self._RELEASE_TOL
        ]
        if candidates:
            self._abandoned.clear()
            self._chair_effort.clear()
            self._chair_best.clear()
            candidates.sort()
            return candidates[0][1]
        return None

    def _staging_pose(self, chair: Tuple[float, float, float, float]
                      ) -> Tuple[float, float, float]:
        """Point behind the chair, on the far side from the target."""
        cx, cy, hx, hy = chair
        tx, ty = self._TARGET
        vx = cx - tx
        vy = cy - ty
        n = math.hypot(vx, vy)
        if n < 1e-6:
            vx, vy, n = 1.0, 0.0, 1.0
        vx /= n
        vy /= n
        half = max(hx, hy)
        off = half + self._BASE_RADIUS + self._STAGE_MARGIN
        sx = cx + vx * off
        sy = cy + vy * off
        heading = math.atan2(ty - sy, tx - sx)
        return sx, sy, heading

    # ------------------------------------------------------------------
    # Obstacle-aware steering
    # ------------------------------------------------------------------

    def _avoidance(self, x: float, y: float,
                   chairs: Dict[str, Tuple[float, float, float, float]],
                   ignore: Optional[str]) -> Tuple[float, float]:
        """Repulsion from chairs we are not currently pushing."""
        rx = ry = 0.0
        for name, (cx, cy, hx, hy) in chairs.items():
            if name == ignore:
                continue
            dx = x - cx
            dy = y - cy
            d = math.hypot(dx, dy)
            keep = max(hx, hy) + self._BASE_RADIUS + 0.10
            if d < 1e-6:
                dx, dy, d = 1.0, 0.0, 1.0
            if d < keep:
                strength = (keep - d) / max(keep, 1e-6)
                rx += (dx / d) * strength
                ry += (dy / d) * strength
        return rx, ry

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self, state, info):
        self._step = 0
        self._robot_name = None
        self._phase = "approach"
        self._target_chair = None
        self._phase_ticks = 0
        self._backoff_ticks = 0
        self._last_xy = None
        self._stall = 0
        self._nudge_ticks = 0
        self._nudge_sign = 1.0
        self._chair_effort = {}
        self._chair_best = {}
        self._abandoned = set()
        self._find_robot(state)
        return None

    def get_action(self, state):
        self._step += 1
        action = np.zeros(self._act_dim, dtype=np.float64)

        pose = self._robot_pose(state)
        if pose is None:
            return self._finalize(action)
        x, y, yaw = pose

        chairs = self._chairs(state)
        if not chairs:
            return self._finalize(action)

        tx, ty = self._TARGET

        # -- choose / re-choose the chair we are working on ---------------
        if (self._target_chair is None
                or self._target_chair not in chairs
                or self._dist(*chairs[self._target_chair][:2], tx, ty)
                <= self._RELEASE_TOL):
            nxt = self._pick_chair(chairs)
            if nxt != self._target_chair:
                self._target_chair = nxt
                self._phase = "approach"
                self._phase_ticks = 0
            if self._target_chair is None:
                return self._finalize(action)   # all placed

        name = self._target_chair
        cx, cy, hx, hy = chairs[name]
        chair_d = self._dist(cx, cy, tx, ty)

        # -- per-chair progress watchdog ----------------------------------
        prev_best = self._chair_best.get(name)
        if prev_best is None or chair_d < prev_best - 0.01:
            self._chair_best[name] = chair_d
            self._chair_effort[name] = 0
        else:
            self._chair_effort[name] = self._chair_effort.get(name, 0) + 1
            if self._chair_effort[name] > 260:
                self._abandoned.add(name)
                self._target_chair = None
                self._phase = "approach"
                return self._finalize(action)

        sx, sy, s_heading = self._staging_pose((cx, cy, hx, hy))

        # -- phase machine -------------------------------------------------
        self._phase_ticks += 1

        if self._phase == "backoff":
            self._backoff_ticks -= 1
            if self._backoff_ticks <= 0:
                self._phase = "approach"
                self._phase_ticks = 0
            # Retreat away from the chair.
            ax = x - cx
            ay = y - cy
            n = math.hypot(ax, ay)
            if n < 1e-6:
                ax, ay, n = 1.0, 0.0, 1.0
            gx_cmd = (ax / n) * self._max_lin_step
            gy_cmd = (ay / n) * self._max_lin_step
            uyaw = 0.0
            return self._emit(action, gx_cmd, gy_cmd, uyaw)

        if self._phase == "approach":
            d_stage = self._dist(x, y, sx, sy)
            heading_ok = abs(_wrap_angle(s_heading - yaw)) < 0.35
            if d_stage < 0.10 and heading_ok:
                self._phase = "push"
                self._phase_ticks = 0
            else:
                gx_cmd = self._k_lin * (sx - x)
                gy_cmd = self._k_lin * (sy - y)
                # Avoid every chair, including the one we intend to push:
                # approach the staging point without shoving it the wrong way.
                rx, ry = self._avoidance(x, y, chairs, ignore=None)
                gx_cmd += rx * self._max_lin_step * 1.8
                gy_cmd += ry * self._max_lin_step * 1.8
                dyaw = _wrap_angle(s_heading - yaw)
                uyaw = self._k_yaw * dyaw
                return self._emit(action, gx_cmd, gy_cmd, uyaw,
                                  stall_xy=(x, y), chairs=chairs)

        # push phase: drive through the chair toward the target
        if self._phase == "push":
            # Lost contact / drifted off the push line? Re-stage.
            off_line = self._point_line_distance(x, y, cx, cy, tx, ty)
            if (self._dist(x, y, cx, cy) > max(hx, hy) + self._BASE_RADIUS + 0.45
                    or off_line > 0.45):
                self._phase = "approach"
                self._phase_ticks = 0
                return self._finalize(action)

            if chair_d <= self._RELEASE_TOL:
                self._phase = "backoff"
                self._backoff_ticks = 12
                return self._finalize(action)

            # Aim the base at a point just past the target, along the push line.
            dirx = tx - cx
            diry = ty - cy
            n = math.hypot(dirx, diry)
            if n < 1e-6:
                dirx, diry, n = 1.0, 0.0, 1.0
            dirx /= n
            diry /= n
            gx_cmd = dirx * self._max_lin_step
            gy_cmd = diry * self._max_lin_step
            # Correct lateral drift so the chair tracks the line to the target.
            perp_x, perp_y = -diry, dirx
            lat = (x - cx) * perp_x + (y - cy) * perp_y
            gx_cmd -= perp_x * lat * 0.9
            gy_cmd -= perp_y * lat * 0.9
            # Steer clear of *other* chairs while pushing this one.
            rx, ry = self._avoidance(x, y, chairs, ignore=name)
            gx_cmd += rx * self._max_lin_step * 1.2
            gy_cmd += ry * self._max_lin_step * 1.2
            dyaw = _wrap_angle(math.atan2(diry, dirx) - yaw)
            uyaw = self._k_yaw * dyaw
            return self._emit(action, gx_cmd, gy_cmd, uyaw,
                              stall_xy=(x, y), chairs=chairs)

        return self._finalize(action)

    # ------------------------------------------------------------------

    @staticmethod
    def _point_line_distance(px: float, py: float,
                             ax: float, ay: float,
                             bx: float, by: float) -> float:
        """Distance from point p to the infinite line through a and b."""
        vx, vy = bx - ax, by - ay
        n = math.hypot(vx, vy)
        if n < 1e-9:
            return math.hypot(px - ax, py - ay)
        return abs((px - ax) * vy - (py - ay) * vx) / n

    def _emit(self, action: np.ndarray, gx: float, gy: float, uyaw: float,
              stall_xy: Optional[Tuple[float, float]] = None,
              chairs: Optional[Dict[str, Tuple[float, float, float, float]]] = None
              ) -> np.ndarray:
        """Clamp, apply stall nudge, write into the base slots, finalize."""
        # Stall detection: wedged? add a lateral burst.
        if stall_xy is not None:
            if self._last_xy is not None:
                moved = math.hypot(stall_xy[0] - self._last_xy[0],
                                   stall_xy[1] - self._last_xy[1])
                if moved < 0.0035:
                    self._stall += 1
                else:
                    self._stall = 0
            self._last_xy = stall_xy

            if self._nudge_ticks > 0:
                self._nudge_ticks -= 1
            elif self._stall >= 14:
                self._nudge_ticks = 14
                self._nudge_sign *= -1.0
                self._stall = 0
                # A hard stall usually means bad staging; re-approach.
                if self._phase == "push":
                    self._phase = "approach"
                    self._phase_ticks = 0

            if self._nudge_ticks > 0:
                n = math.hypot(gx, gy)
                if n > 1e-6:
                    px, py = -gy / n, gx / n
                    gx += self._nudge_sign * px * self._max_lin_step * 0.8
                    gy += self._nudge_sign * py * self._max_lin_step * 0.8

        step = math.hypot(gx, gy)
        if step > self._max_lin_step and step > 1e-12:
            s = self._max_lin_step / step
            gx *= s
            gy *= s

        uyaw = max(-self._max_yaw_step, min(self._max_yaw_step, uyaw))

        action[0] = gx
        if self._act_dim > 1:
            action[1] = gy
        if self._act_dim > 2:
            action[2] = uyaw
        return self._finalize(action)

    def _finalize(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, self._low, self._high)
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            action = np.clip(action, self._low, self._high)
        dtype = getattr(self.action_space, "dtype", np.float32)
        try:
            return action.astype(dtype)
        except Exception:
            return action.astype(np.float32)