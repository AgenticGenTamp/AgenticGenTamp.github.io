"""Motion2D waypoint-following policy.

Strategy
--------
The scene is a left-to-right sequence of thin vertical walls, each made of a
bottom piece (rising from the floor) and a top piece (hanging from the ceiling),
leaving a narrow horizontal-slot doorway between them.  The robot base is a
circle, the target is a rectangle on the right.

So:

1. Group the ``obstacle*`` rectangles by their x coordinate -> one group per wall.
2. For each wall, the doorway spans from ``max(y + height)`` over the pieces that
   start at the floor, up to ``min(y)`` over the remaining (ceiling) pieces.
   Aim at the middle of that span.
3. Emit, per wall in increasing x: an *approach* waypoint just left of the wall at
   the gap's y, and an *exit* waypoint just right of the wall at the same y.
   Finish with the target-region center.
4. Follow waypoints with axis-sequenced motion: align y first (while safely
   between walls), then translate purely horizontally through the slot.  Pure
   horizontal motion through a centered gap always fits; diagonal motion wedges.
5. Because a colliding action is a no-op, detect "position did not change" and
   run a small recovery (back off in -x, re-align y, retry).

Everything is derived from the state at ``reset`` time, so any number of walls
(including zero) is handled.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _feature(state, obj, name, default=None):
    """Read a feature, returning ``default`` if the type lacks it."""
    try:
        return float(state.get(obj, name))
    except Exception:
        return default


def _type_name(obj):
    try:
        return obj.type.name
    except Exception:
        return ""


def _is_robot(state, obj):
    """A robot is the object carrying base_radius / arm_joint features."""
    if _type_name(obj) == "crv_robot":
        return True
    return _feature(state, obj, "base_radius") is not None and \
        _feature(state, obj, "arm_joint") is not None


def _is_target(state, obj):
    if _type_name(obj) == "target_region":
        return True
    # Fallback: object literally named like a target region.
    return "target" in obj.name.lower()


def _rect_bounds(state, obj):
    """Axis-aligned (x_lo, x_hi, y_lo, y_hi) for a rectangle object.

    Kinematic2D rectangles are given by lower-left corner + width/height + theta.
    In this environment theta is always 0, but rotate defensively anyway so a
    rotated rectangle still yields a correct bounding box.
    """
    x = _feature(state, obj, "x", 0.0)
    y = _feature(state, obj, "y", 0.0)
    w = _feature(state, obj, "width", 0.0)
    h = _feature(state, obj, "height", 0.0)
    th = _feature(state, obj, "theta", 0.0) or 0.0
    if abs(th) < 1e-9:
        return (x, x + w, y, y + h)
    c, s = np.cos(th), np.sin(th)
    corners = [
        (x, y),
        (x + w * c, y + w * s),
        (x + w * c - h * s, y + w * s + h * c),
        (x - h * s, y + h * c),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), max(xs), min(ys), max(ys))


# --------------------------------------------------------------------------- #
# The approach
# --------------------------------------------------------------------------- #

class GeneratedApproach:
    """Waypoint follower that threads every narrow passage's gap center."""

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        self._low = low
        self._high = high
        # Per-step translation limits (use the smaller magnitude, symmetric use).
        self._max_dx = float(min(abs(low[0]), abs(high[0])))
        self._max_dy = float(min(abs(low[1]), abs(high[1])))

        # Tunables.
        self._pos_tol = 0.015           # waypoint arrival tolerance
        self._align_tol = 0.008         # y-alignment tolerance before entering a gap
        self._stuck_eps = 1e-4          # movement below this counts as "no motion"
        self._stuck_patience = 3        # consecutive no-motion steps before recovery
        self._recovery_len = 6          # steps spent in a recovery maneuver

        self._reset_episode_vars()

    # -- episode bookkeeping ------------------------------------------------ #

    def _reset_episode_vars(self):
        self._waypoints = []
        self._wp_idx = 0
        self._prev_xy = None
        self._stuck_count = 0
        self._recovery_steps = 0
        self._recovery_dir = 1.0
        self._step_count = 0
        self._base_radius = 0.1
        self._clearance = 0.15
        self._gap_x_spans = []   # (x_lo, x_hi) of each wall, for "inside gap" checks

    # -- plan construction -------------------------------------------------- #

    def _build_plan(self, state):
        """Compute the waypoint list from an initial (or current) state."""
        robot = None
        target = None
        obstacles = []

        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            if _is_robot(state, obj):
                if robot is None:
                    robot = obj
                continue
            if _is_target(state, obj):
                if target is None:
                    target = obj
                continue
            if name.startswith("obstacle"):
                obstacles.append(obj)

        # Robot geometry.
        if robot is not None:
            br = _feature(state, robot, "base_radius", 0.1) or 0.1
            self._base_radius = float(br)
        # Clearance must also cover the (retracted) gripper sticking out front.
        arm = 0.0
        if robot is not None:
            arm = _feature(state, robot, "arm_joint", 0.0) or 0.0
            gw = _feature(state, robot, "gripper_width", 0.0) or 0.0
            arm = arm + gw
        self._clearance = max(self._base_radius, arm) + 0.05

        # Final goal: center of the target region.
        goal = None
        if target is not None:
            tx_lo, tx_hi, ty_lo, ty_hi = _rect_bounds(state, target)
            goal = (0.5 * (tx_lo + tx_hi), 0.5 * (ty_lo + ty_hi))

        # Group obstacle rectangles into walls by shared x.
        walls = self._group_walls(state, obstacles)

        waypoints = []
        self._gap_x_spans = []
        rx = _feature(state, robot, "x", 0.0) if robot is not None else 0.0

        for (x_lo, x_hi, gap_y) in walls:
            # Only walls that actually lie ahead of the robot matter; a wall
            # already behind us should not drag us backwards.
            if x_hi < rx - self._clearance:
                continue
            approach = (x_lo - self._clearance, gap_y)
            exit_wp = (x_hi + self._clearance, gap_y)
            waypoints.append(approach)
            waypoints.append(exit_wp)
            self._gap_x_spans.append((x_lo - self._clearance * 0.5,
                                      x_hi + self._clearance * 0.5))

        if goal is not None:
            waypoints.append(goal)

        self._waypoints = waypoints
        self._wp_idx = 0

    def _group_walls(self, state, obstacles):
        """Return sorted list of (wall_x_lo, wall_x_hi, gap_center_y) per wall."""
        if not obstacles:
            return []

        # Bounds for each obstacle, plus the global vertical extent (used to
        # decide which pieces are "floor" pieces and which are "ceiling" pieces).
        recs = []
        for obj in obstacles:
            x_lo, x_hi, y_lo, y_hi = _rect_bounds(state, obj)
            recs.append((x_lo, x_hi, y_lo, y_hi))

        world_y_lo = min(r[2] for r in recs)
        world_y_hi = max(r[3] for r in recs)

        # Cluster by x with a tolerance comparable to the (very thin) wall width.
        widths = [r[1] - r[0] for r in recs]
        med_w = float(np.median(widths)) if widths else 0.01
        tol = max(med_w * 2.0, 1e-3)

        recs_sorted = sorted(recs, key=lambda r: r[0])
        groups = []
        current = [recs_sorted[0]]
        for r in recs_sorted[1:]:
            if abs(r[0] - current[-1][0]) <= tol:
                current.append(r)
            else:
                groups.append(current)
                current = [r]
        groups.append(current)

        walls = []
        for grp in groups:
            x_lo = min(r[0] for r in grp)
            x_hi = max(r[1] for r in grp)

            # Floor pieces: bottom sits at (or near) the lowest extent.
            floor_tops = [r[3] for r in grp if r[2] <= world_y_lo + 1e-6]
            ceil_bots = [r[2] for r in grp if r[3] >= world_y_hi - 1e-6]

            if floor_tops and ceil_bots:
                gap_lo = max(floor_tops)
                gap_hi = min(ceil_bots)
            elif floor_tops:
                # Only a bottom piece: pass above it.
                gap_lo = max(floor_tops)
                gap_hi = world_y_hi
            elif ceil_bots:
                # Only a top piece: pass below it.
                gap_lo = world_y_lo
                gap_hi = min(ceil_bots)
            else:
                # Free-floating blocks: use the biggest vertical opening between
                # the stacked pieces (generic fallback).
                gap_lo, gap_hi = self._largest_opening(grp, world_y_lo, world_y_hi)

            if gap_hi <= gap_lo:
                # Degenerate; fall back to the largest opening in the group.
                gap_lo, gap_hi = self._largest_opening(grp, world_y_lo, world_y_hi)

            gap_y = 0.5 * (gap_lo + gap_hi)
            walls.append((x_lo, x_hi, gap_y))

        walls.sort(key=lambda w: w[0])
        return walls

    @staticmethod
    def _largest_opening(grp, world_y_lo, world_y_hi):
        """Largest vertical free interval among the pieces of one wall group."""
        spans = sorted(((r[2], r[3]) for r in grp), key=lambda s: s[0])
        best = (world_y_lo, world_y_hi)
        best_size = -1.0
        cursor = world_y_lo
        for (lo, hi) in spans:
            if lo - cursor > best_size:
                best_size = lo - cursor
                best = (cursor, lo)
            cursor = max(cursor, hi)
        if world_y_hi - cursor > best_size:
            best = (cursor, world_y_hi)
        return best

    # -- gym-facing API ----------------------------------------------------- #

    def reset(self, state, info):
        self._reset_episode_vars()
        try:
            self._build_plan(state)
        except Exception:
            self._waypoints = []
            self._wp_idx = 0
        return None

    def get_action(self, state):
        self._step_count += 1

        robot = None
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            if _is_robot(state, obj):
                robot = obj
                break

        action = np.zeros(self._low.shape, dtype=np.float32)
        if robot is None:
            return self._finalize(action)

        rx = _feature(state, robot, "x", 0.0)
        ry = _feature(state, robot, "y", 0.0)

        # Lazily build a plan if reset was never called / plan is empty.
        if not self._waypoints:
            try:
                self._build_plan(state)
            except Exception:
                self._waypoints = []
            if not self._waypoints:
                return self._finalize(action)

        # -- stuck detection (a colliding action is a no-op) ----------------- #
        if self._prev_xy is not None:
            moved = abs(rx - self._prev_xy[0]) + abs(ry - self._prev_xy[1])
            if moved < self._stuck_eps:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
        self._prev_xy = (rx, ry)

        if self._recovery_steps <= 0 and self._stuck_count >= self._stuck_patience:
            self._recovery_steps = self._recovery_len
            self._stuck_count = 0
            # Alternate the recovery nudge direction in y each time it fires.
            self._recovery_dir *= -1.0

        # -- advance through waypoints -------------------------------------- #
        while self._wp_idx < len(self._waypoints) - 1:
            wx, wy = self._waypoints[self._wp_idx]
            if abs(wx - rx) <= self._pos_tol and abs(wy - ry) <= self._pos_tol:
                self._wp_idx += 1
            else:
                break

        wx, wy = self._waypoints[min(self._wp_idx, len(self._waypoints) - 1)]

        # -- recovery maneuver ---------------------------------------------- #
        if self._recovery_steps > 0:
            self._recovery_steps -= 1
            # Back away from the wall (in -x) and jiggle in y to unwedge.
            action[0] = np.float32(-self._max_dx * 0.6)
            action[1] = np.float32(self._recovery_dir * self._max_dy * 0.6)
            return self._finalize(action)

        # -- normal axis-sequenced motion ----------------------------------- #
        dx_need = wx - rx
        dy_need = wy - ry

        inside_gap = any(x_lo <= rx <= x_hi for (x_lo, x_hi) in self._gap_x_spans)

        if not inside_gap and abs(dy_need) > self._align_tol:
            # Align y first, in the safe corridor between walls.
            dx_cmd = 0.0
            dy_cmd = dy_need
        else:
            # Aligned (or already committed inside a slot): drive in x, with a
            # small y correction to hold the center line.
            dx_cmd = dx_need
            dy_cmd = np.clip(dy_need, -0.25 * self._max_dy, 0.25 * self._max_dy)

        action[0] = np.float32(np.clip(dx_cmd, -self._max_dx, self._max_dx))
        action[1] = np.float32(np.clip(dy_cmd, -self._max_dy, self._max_dy))
        # theta / arm / vacuum are irrelevant for this task: leave them at zero.
        return self._finalize(action)

    # -- action sanitation --------------------------------------------------- #

    def _finalize(self, action):
        """Clip into the action space and match its dtype/shape exactly."""
        a = np.asarray(action, dtype=np.float64).reshape(self._low.shape)
        a = np.clip(a, self._low, self._high)
        dtype = getattr(self.action_space, "dtype", np.float32)
        out = a.astype(dtype)
        # Guard against float32 rounding pushing a value a hair out of bounds.
        out = np.clip(out, self._low.astype(dtype), self._high.astype(dtype))
        return out