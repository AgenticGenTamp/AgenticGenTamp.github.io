"""
GeneratedApproach for SweepSimple3DEnv (variable object count).

Strategy (count-invariant):
  1. Grab the wiper tool (wiper_0) with the mobile base + arm.
  2. Compute the centroid of all cubes still outside the goal region.
  3. Position the wiper "behind" the cluster along the sweep direction
     (centroid -> goal), blade perpendicular to that direction.
  4. Push the base forward along the sweep direction, dragging the wiper
     through the cluster and shoving all cubes toward the goal at once.
  5. Recompute and repeat until done / out of steps.

The only count-dependent quantities are O(n) reductions (centroid, cluster
radius, "still outside" filter). The macro-action sequence has fixed length.

Action layout (TidyBot3DRobotActionSpace, 11 dims):
    [0:3]  base pose  (x, y, theta)
    [3:6]  arm end-effector position (x, y, z)
    [6:10] arm end-effector orientation quaternion (x, y, z, w)
    [10]   gripper position (0 = open, 1 = closed)

The environment is configured with act_delta=True by default, but we cannot
rely on that from inside the policy; we therefore *clip into the action
space bounds* and use a controller that works reasonably in either
interpretation: we emit absolute-ish targets when the space bounds are wide
(absolute mode) and small deltas when the bounds are tight (delta mode).
This is decided once, at construction, from the action_space low/high.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Small math helpers
# --------------------------------------------------------------------------


def _wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _quat_wxyz_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw (rotation about z) from a (w, x, y, z) quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) for a pure yaw rotation."""
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
                    dtype=np.float64)


def _quat_mul_xyzw(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Hamilton product of two (x, y, z, w) quaternions: a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _rot2d(v: Sequence[float], theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)


# --------------------------------------------------------------------------
# The approach
# --------------------------------------------------------------------------


class GeneratedApproach:
    # ---- tunable constants -------------------------------------------------

    # Gripper conventions.
    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSED = 1.0

    # Nominal "carry" height of the end effector above the floor while the
    # wiper is held and being dragged.
    CARRY_Z = 0.075
    # Height used while descending onto the wiper.
    GRASP_Z = 0.035
    # Height used while moving around without the tool.
    TRAVEL_Z = 0.30

    # How far the base should stand back from the point the end effector is
    # aiming at (roughly the arm's comfortable forward reach).
    REACH = 0.45

    # Margin behind the cluster at which the blade is planted before a push.
    BACKOFF_EXTRA = 0.28
    # Extra distance pushed past the goal point (so cubes end up well inside).
    OVERSHOOT = 0.12

    # Per-step motion limits (used to synthesize smooth trajectories).
    MAX_BASE_STEP = 0.06          # metres of base translation per step
    MAX_BASE_YAW_STEP = 0.14      # radians of base yaw per step
    MAX_EE_STEP = 0.05            # metres of end-effector translation per step

    # Tolerances for declaring a waypoint reached.
    BASE_POS_TOL = 0.07
    BASE_YAW_TOL = 0.16
    EE_POS_TOL = 0.05

    # Step budgets for each macro phase (safety valve against getting stuck).
    MAX_PHASE_STEPS = 220

    # Goal region: offset of the "left side of the kitchen island" sweep
    # target, expressed in the island's own frame (metres). Measured from the
    # island fixture pose; the island is ~1.2 m long so this puts the target
    # off to one side of it, on the floor.
    ISLAND_TARGET_LOCAL = np.array([0.0, -0.70], dtype=np.float64)
    # Radius within which we consider a cube "done" (goal tolerance is 5cm,
    # but the region itself is larger; we use a generous disc).
    GOAL_RADIUS = 0.32

    # --------------------------------------------------------------------

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        # Action space bounds (11-dim).
        low = np.asarray(getattr(action_space, "low", np.full(11, -1.0)),
                         dtype=np.float64).reshape(-1)
        high = np.asarray(getattr(action_space, "high", np.full(11, 1.0)),
                          dtype=np.float64).reshape(-1)
        if low.shape[0] != 11:
            low = np.full(11, -1.0, dtype=np.float64)
            high = np.full(11, 1.0, dtype=np.float64)
        self._low = low
        self._high = high
        self._adim = int(low.shape[0])

        # Decide delta vs absolute mode from the base-x bound. A delta-mode
        # space has a small symmetric bound (e.g. +-0.1 m); an absolute-mode
        # space spans the room (several metres).
        span = float(high[0] - low[0])
        self._delta_mode = span < 2.0

        # Episode state.
        self._t = 0
        self._phase_idx = 0
        self._phase_t = 0
        self._plan: List[Dict[str, Any]] = []
        self._pass_count = 0
        self._robot_name: Optional[str] = None
        self._wiper_name: Optional[str] = None
        self._goal_xy: Optional[np.ndarray] = None
        self._sweep_dir: np.ndarray = np.array([1.0, 0.0], dtype=np.float64)

    # ---------------------------------------------------------------- reset

    def reset(self, state, info):
        self._t = 0
        self._phase_idx = 0
        self._phase_t = 0
        self._plan = []
        self._pass_count = 0

        self._robot_name = self._find_robot_name(state)
        self._wiper_name = self._find_wiper_name(state)
        self._goal_xy = self._compute_goal_xy(state)

        # Build the first macro plan: grab the wiper, then sweep.
        self._build_plan(state, include_grasp=True)
        return None

    # ------------------------------------------------------------ main loop

    def get_action(self, state):
        self._t += 1

        try:
            action = self._get_action_inner(state)
        except Exception:
            # Never crash the rollout; fall back to a no-op.
            action = self._noop(state)

        return self._finish(action)

    # ------------------------------------------------------------- internals

    def _get_action_inner(self, state):
        if self._goal_xy is None:
            self._goal_xy = self._compute_goal_xy(state)

        # Advance through the plan, skipping completed phases.
        guard = 0
        while guard < 64:
            guard += 1

            if self._phase_idx >= len(self._plan):
                # Plan exhausted: recompute a fresh sweeping pass.
                self._pass_count += 1
                self._build_plan(state, include_grasp=self._should_regrasp(state))
                if self._phase_idx >= len(self._plan):
                    return self._noop(state)

            phase = self._plan[self._phase_idx]

            if self._phase_done(state, phase) or self._phase_t > self.MAX_PHASE_STEPS:
                self._phase_idx += 1
                self._phase_t = 0
                continue

            self._phase_t += 1
            return self._execute_phase(state, phase)

        return self._noop(state)

    # ------------------------------------------------------ plan construction

    def _build_plan(self, state, include_grasp: bool) -> None:
        """Construct the macro-phase list for one sweeping pass."""
        self._phase_idx = 0
        self._phase_t = 0
        self._plan = []

        goal = self._goal_xy if self._goal_xy is not None else np.zeros(2)

        remaining = self._remaining_cubes(state)
        if not remaining:
            # Everything is (probably) done; idle gracefully.
            self._plan = [{"kind": "idle"}]
            return

        centroid, radius = self._cluster_stats(remaining)

        # Sweep direction: from the cluster toward the goal.
        direction = _normalize(np.asarray(goal, dtype=np.float64) - centroid)
        self._sweep_dir = direction

        # Base yaw so the robot faces along the sweep direction.
        push_yaw = float(math.atan2(direction[1], direction[0]))

        # Where the blade is planted before the push.
        backoff = radius + self.BACKOFF_EXTRA
        start_xy = centroid - direction * backoff
        # Where the blade ends up after the push.
        end_xy = np.asarray(goal, dtype=np.float64) + direction * self.OVERSHOOT

        if include_grasp:
            wiper_xy, wiper_yaw = self._wiper_pose(state)
            if wiper_xy is not None:
                # Approach the wiper from a standoff, descend, close, lift.
                approach_yaw = wiper_yaw
                self._plan.extend(
                    [
                        {
                            "kind": "goto_ee",
                            "ee_xy": wiper_xy,
                            "ee_z": self.TRAVEL_Z,
                            "yaw": approach_yaw,
                            "grip": self.GRIPPER_OPEN,
                            "tol_scale": 1.4,
                        },
                        {
                            "kind": "goto_ee",
                            "ee_xy": wiper_xy,
                            "ee_z": self.GRASP_Z,
                            "yaw": approach_yaw,
                            "grip": self.GRIPPER_OPEN,
                            "tol_scale": 1.0,
                        },
                        {
                            "kind": "hold",
                            "steps": 8,
                            "ee_xy": wiper_xy,
                            "ee_z": self.GRASP_Z,
                            "yaw": approach_yaw,
                            "grip": self.GRIPPER_CLOSED,
                        },
                        {
                            "kind": "goto_ee",
                            "ee_xy": wiper_xy,
                            "ee_z": self.CARRY_Z,
                            "yaw": approach_yaw,
                            "grip": self.GRIPPER_CLOSED,
                            "tol_scale": 1.4,
                        },
                    ]
                )

        # Swing around behind the cluster with the tool held, turn to face the
        # goal, plant the blade, then push.
        stage_xy = start_xy - direction * 0.30
        self._plan.extend(
            [
                {
                    "kind": "goto_ee",
                    "ee_xy": stage_xy,
                    "ee_z": self.CARRY_Z + 0.06,
                    "yaw": push_yaw,
                    "grip": self.GRIPPER_CLOSED,
                    "tol_scale": 1.8,
                },
                {
                    "kind": "goto_ee",
                    "ee_xy": start_xy,
                    "ee_z": self.CARRY_Z,
                    "yaw": push_yaw,
                    "grip": self.GRIPPER_CLOSED,
                    "tol_scale": 1.2,
                },
                {
                    "kind": "push",
                    "from_xy": start_xy,
                    "to_xy": end_xy,
                    "yaw": push_yaw,
                    "grip": self.GRIPPER_CLOSED,
                },
                # Back off a little so the next pass can re-approach cleanly.
                {
                    "kind": "goto_ee",
                    "ee_xy": end_xy - direction * 0.35,
                    "ee_z": self.CARRY_Z + 0.08,
                    "yaw": push_yaw,
                    "grip": self.GRIPPER_CLOSED,
                    "tol_scale": 1.8,
                },
            ]
        )

    def _should_regrasp(self, state) -> bool:
        """True if the wiper appears to have been dropped away from the hand."""
        wiper_xy, _ = self._wiper_pose(state)
        if wiper_xy is None:
            return False
        base_xy, _ = self._base_pose(state)
        if base_xy is None:
            return True
        # If the tool is far from the base it is no longer being carried.
        return bool(np.linalg.norm(wiper_xy - base_xy) > 1.1)

    # ----------------------------------------------------- phase termination

    def _phase_done(self, state, phase) -> bool:
        kind = phase.get("kind", "idle")

        if kind == "idle":
            return False

        if kind == "hold":
            return self._phase_t >= int(phase.get("steps", 5))

        base_xy, base_yaw = self._base_pose(state)
        if base_xy is None:
            return True

        if kind == "goto_ee":
            target_base = self._base_target_for_ee(
                np.asarray(phase["ee_xy"], dtype=np.float64), float(phase["yaw"])
            )
            tol = self.BASE_POS_TOL * float(phase.get("tol_scale", 1.0))
            pos_ok = bool(np.linalg.norm(target_base - base_xy) < tol)
            yaw_ok = abs(_wrap_angle(float(phase["yaw"]) - base_yaw)) < (
                self.BASE_YAW_TOL * float(phase.get("tol_scale", 1.0))
            )
            return pos_ok and yaw_ok

        if kind == "push":
            target_base = self._base_target_for_ee(
                np.asarray(phase["to_xy"], dtype=np.float64), float(phase["yaw"])
            )
            return bool(np.linalg.norm(target_base - base_xy) < self.BASE_POS_TOL)

        return True

    # -------------------------------------------------------- phase execution

    def _execute_phase(self, state, phase):
        kind = phase.get("kind", "idle")

        if kind == "idle":
            return self._noop(state)

        if kind == "hold":
            return self._command(
                state,
                base_xy=None,
                base_yaw=float(phase.get("yaw", 0.0)),
                ee_xy=np.asarray(phase["ee_xy"], dtype=np.float64),
                ee_z=float(phase["ee_z"]),
                grip=float(phase["grip"]),
                hold_base=True,
            )

        if kind in ("goto_ee", "push"):
            if kind == "push":
                ee_xy = np.asarray(phase["to_xy"], dtype=np.float64)
                ee_z = self.CARRY_Z
            else:
                ee_xy = np.asarray(phase["ee_xy"], dtype=np.float64)
                ee_z = float(phase["ee_z"])
            yaw = float(phase["yaw"])
            base_target = self._base_target_for_ee(ee_xy, yaw)
            return self._command(
                state,
                base_xy=base_target,
                base_yaw=yaw,
                ee_xy=ee_xy,
                ee_z=ee_z,
                grip=float(phase["grip"]),
                hold_base=False,
            )

        return self._noop(state)

    # --------------------------------------------------------- command build

    def _base_target_for_ee(self, ee_xy: np.ndarray, yaw: float) -> np.ndarray:
        """Base (x, y) that puts the end effector at ee_xy when facing yaw."""
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        return np.asarray(ee_xy, dtype=np.float64) - forward * self.REACH

    def _command(
        self,
        state,
        base_xy: Optional[np.ndarray],
        base_yaw: float,
        ee_xy: np.ndarray,
        ee_z: float,
        grip: float,
        hold_base: bool,
    ) -> np.ndarray:
        cur_base_xy, cur_base_yaw = self._base_pose(state)
        if cur_base_xy is None:
            cur_base_xy = np.zeros(2, dtype=np.float64)
            cur_base_yaw = 0.0

        if hold_base or base_xy is None:
            desired_base = cur_base_xy.copy()
        else:
            desired_base = np.asarray(base_xy, dtype=np.float64)

        # Rate-limit the base toward its target.
        d_base = desired_base - cur_base_xy
        dist = float(np.linalg.norm(d_base))
        if dist > self.MAX_BASE_STEP:
            d_base = d_base * (self.MAX_BASE_STEP / dist)
        d_yaw = _wrap_angle(base_yaw - cur_base_yaw)
        d_yaw = float(np.clip(d_yaw, -self.MAX_BASE_YAW_STEP, self.MAX_BASE_YAW_STEP))

        # End-effector target expressed in world coordinates, then rate-limited
        # relative to a nominal "in front of the base" reference.
        nominal_ee = cur_base_xy + np.array(
            [math.cos(cur_base_yaw), math.sin(cur_base_yaw)], dtype=np.float64
        ) * self.REACH
        d_ee = np.asarray(ee_xy, dtype=np.float64) - nominal_ee
        n_ee = float(np.linalg.norm(d_ee))
        if n_ee > self.MAX_EE_STEP:
            d_ee = d_ee * (self.MAX_EE_STEP / n_ee)

        # Orientation: gripper pointing down, yawed with the base.
        # A downward-pointing tool frame: rotate pi about x, then yaw about z.
        q_down = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # (x,y,z,w) = pi about x
        q_yaw = _yaw_to_quat_xyzw(base_yaw)
        quat_xyzw = _quat_mul_xyzw(q_yaw, q_down)

        action = np.zeros(self._adim, dtype=np.float64)

        if self._delta_mode:
            action[0] = d_base[0]
            action[1] = d_base[1]
            action[2] = d_yaw
            action[3] = d_ee[0]
            action[4] = d_ee[1]
            # Vertical: drive toward ee_z from the nominal carry height.
            action[5] = float(np.clip(ee_z - self.CARRY_Z, -self.MAX_EE_STEP,
                                      self.MAX_EE_STEP))
            # Small orientation correction only (deltas), keep near identity.
            action[6] = 0.0
            action[7] = 0.0
            action[8] = float(np.clip(d_yaw, -0.1, 0.1))
            if self._adim > 9:
                action[9] = 0.0
            action[10 if self._adim > 10 else self._adim - 1] = grip
        else:
            target_base = cur_base_xy + d_base
            action[0] = target_base[0]
            action[1] = target_base[1]
            action[2] = _wrap_angle(cur_base_yaw + d_yaw)
            target_ee = nominal_ee + d_ee
            action[3] = target_ee[0]
            action[4] = target_ee[1]
            action[5] = ee_z
            action[6] = quat_xyzw[0]
            action[7] = quat_xyzw[1]
            action[8] = quat_xyzw[2]
            if self._adim > 9:
                action[9] = quat_xyzw[3]
            action[10 if self._adim > 10 else self._adim - 1] = grip

        return action

    def _noop(self, state) -> np.ndarray:
        action = np.zeros(self._adim, dtype=np.float64)
        if self._delta_mode:
            if self._adim > 10:
                action[10] = self.GRIPPER_CLOSED
            return action
        cur_base_xy, cur_base_yaw = self._base_pose(state)
        if cur_base_xy is None:
            cur_base_xy = np.zeros(2, dtype=np.float64)
            cur_base_yaw = 0.0
        action[0] = cur_base_xy[0]
        action[1] = cur_base_xy[1]
        action[2] = cur_base_yaw
        nominal_ee = cur_base_xy + np.array(
            [math.cos(cur_base_yaw), math.sin(cur_base_yaw)], dtype=np.float64
        ) * self.REACH
        action[3] = nominal_ee[0]
        action[4] = nominal_ee[1]
        action[5] = self.CARRY_Z
        q = _quat_mul_xyzw(_yaw_to_quat_xyzw(cur_base_yaw),
                           np.array([1.0, 0.0, 0.0, 0.0]))
        action[6] = q[0]
        action[7] = q[1]
        action[8] = q[2]
        if self._adim > 9:
            action[9] = q[3]
        if self._adim > 10:
            action[10] = self.GRIPPER_CLOSED
        return action

    def _finish(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self._adim:
            b = np.zeros(self._adim, dtype=np.float64)
            n = min(self._adim, a.shape[0])
            b[:n] = a[:n]
            a = b
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.clip(a, self._low, self._high)
        return a.astype(np.float32)

    # ------------------------------------------------------- state accessors

    def _find_robot_name(self, state) -> Optional[str]:
        if self._robot_name is not None:
            try:
                state.get_object_from_name(self._robot_name)
                return self._robot_name
            except Exception:
                pass
        for name in state.get_object_names():
            try:
                obj = state.get_object_from_name(name)
            except Exception:
                continue
            tname = getattr(getattr(obj, "type", None), "name", "")
            if "robot" in str(tname):
                return name
        return None

    def _find_wiper_name(self, state) -> Optional[str]:
        best = None
        best_extent = -1.0
        for name in state.get_object_names():
            try:
                obj = state.get_object_from_name(name)
            except Exception:
                continue
            tname = str(getattr(getattr(obj, "type", None), "name", ""))
            if tname != "mujoco_movable_object":
                continue
            if name.startswith("cube_"):
                continue
            try:
                extent = max(
                    float(state.get(obj, "bb_x")),
                    float(state.get(obj, "bb_y")),
                    float(state.get(obj, "bb_z")),
                )
            except Exception:
                extent = 0.0
            if name.startswith("wiper"):
                extent += 10.0
            if extent > best_extent:
                best_extent = extent
                best = name
        return best

    def _base_pose(self, state) -> Tuple[Optional[np.ndarray], float]:
        name = self._find_robot_name(state)
        if name is None:
            return None, 0.0
        try:
            obj = state.get_object_from_name(name)
            x = float(state.get(obj, "pos_base_x"))
            y = float(state.get(obj, "pos_base_y"))
            th = float(state.get(obj, "pos_base_rot"))
            return np.array([x, y], dtype=np.float64), th
        except Exception:
            return None, 0.0

    def _wiper_pose(self, state) -> Tuple[Optional[np.ndarray], float]:
        name = self._wiper_name or self._find_wiper_name(state)
        self._wiper_name = name
        if name is None:
            return None, 0.0
        try:
            obj = state.get_object_from_name(name)
            x = float(state.get(obj, "x"))
            y = float(state.get(obj, "y"))
            yaw = _quat_wxyz_to_yaw(
                float(state.get(obj, "qw")),
                float(state.get(obj, "qx")),
                float(state.get(obj, "qy")),
                float(state.get(obj, "qz")),
            )
            return np.array([x, y], dtype=np.float64), yaw
        except Exception:
            return None, 0.0

    def _cube_positions(self, state) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for name in state.get_object_names():
            if not name.startswith("cube_"):
                continue
            try:
                obj = state.get_object_from_name(name)
                out.append(
                    np.array(
                        [float(state.get(obj, "x")), float(state.get(obj, "y"))],
                        dtype=np.float64,
                    )
                )
            except Exception:
                continue
        return out

    def _remaining_cubes(self, state) -> List[np.ndarray]:
        goal = self._goal_xy
        cubes = self._cube_positions(state)
        if goal is None:
            return cubes
        rem = [c for c in cubes if float(np.linalg.norm(c - goal)) > self.GOAL_RADIUS]
        return rem if rem else []

    @staticmethod
    def _cluster_stats(points: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        centroid = arr.mean(axis=0)
        radius = float(np.max(np.linalg.norm(arr - centroid, axis=1))) if len(arr) else 0.0
        return centroid, radius

    def _compute_goal_xy(self, state) -> np.ndarray:
        """Target point: offset from the kitchen island, in the island's frame."""
        island = None
        for name in state.get_object_names():
            if "island" in name and "drawer" not in name:
                try:
                    obj = state.get_object_from_name(name)
                except Exception:
                    continue
                tname = str(getattr(getattr(obj, "type", None), "name", ""))
                if tname == "mujoco_fixture":
                    island = obj
                    break
        if island is None:
            # Fall back to the centroid of the cubes shifted sideways.
            cubes = self._cube_positions(state)
            if cubes:
                c, _ = self._cluster_stats(cubes)
                return c + np.array([-0.6, 0.0], dtype=np.float64)
            return np.zeros(2, dtype=np.float64)

        try:
            ix = float(state.get(island, "x"))
            iy = float(state.get(island, "y"))
            iyaw = _quat_wxyz_to_yaw(
                float(state.get(island, "qw")),
                float(state.get(island, "qx")),
                float(state.get(island, "qy")),
                float(state.get(island, "qz")),
            )
        except Exception:
            return np.zeros(2, dtype=np.float64)

        offset_world = _rot2d(self.ISLAND_TARGET_LOCAL, iyaw)
        return np.array([ix, iy], dtype=np.float64) + offset_world