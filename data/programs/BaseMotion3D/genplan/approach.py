"""Policy for kinder/BaseMotion3D-v0.

Strategy
--------
The goal is purely a base-navigation problem: the episode terminates when the
Euclidean distance between the robot base (obs[0:2]) and the target
(obs[19:21]) drops below `target_radius = 0.05`.  Nothing about the arm,
fingers, or grasp state matters.

The base delta in the action is applied in the WORLD frame (the env does
`current_base_pose + SE2Pose(dx, dy, drot)`), so we can simply drive straight
at the target with a proportional controller whose step is clipped to the
action-space magnitude limit (0.4 by default).  On the final approach the
un-clipped error is used directly, so we land exactly on the target instead of
oscillating around the tight 0.05 radius.

The only complication is that the scene loads a realistic room mesh as a
collision body and `check_base_collisions=True`; when a motion collides the
*entire* step is reverted (the robot does not move at all).  So we add a cheap
stall-recovery: if a commanded motion produced (almost) no displacement, we
sidestep perpendicular to the goal direction for a few steps before resuming
the straight-line drive, alternating the sidestep sign and growing the detour
length if it keeps failing.
"""

from __future__ import annotations

import numpy as np


# Observation layout (22-dim).
_OBS_BASE_X = 0
_OBS_BASE_Y = 1
_OBS_BASE_ROT = 2
_OBS_TARGET_X = 19
_OBS_TARGET_Y = 20

# Action layout (11-dim).
_ACT_DIM = 11
_ACT_BASE_X = 0
_ACT_BASE_Y = 1
_ACT_BASE_ROT = 2

# Env constant: goal_reached() uses dist < target_radius.
_TARGET_RADIUS = 0.05
# Aim a bit inside the radius so float32 round-tripping of the observation
# cannot leave us marginally outside.
_GOAL_TOL = 0.6 * _TARGET_RADIUS

# If a commanded step of magnitude >= _STALL_CMD produced a displacement below
# _STALL_MOVE, treat it as a collision-blocked (reverted) step.
_STALL_CMD = 1e-3
_STALL_MOVE = 1e-4

# Sidestep schedule for stall recovery.
_MIN_DETOUR_STEPS = 2
_MAX_DETOUR_STEPS = 12


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale `vec` so its L2 norm is at most `max_norm`."""
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm <= 0.0:
        return vec
    return vec * (max_norm / norm)


class GeneratedApproach:
    """Clipped proportional base controller with collision stall recovery."""

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        # Action bounds: index 0/1 are the base x/y deltas.  Use the tightest
        # symmetric bound available so the action is always inside the space.
        low = np.asarray(getattr(action_space, "low", -0.4 * np.ones(_ACT_DIM)),
                         dtype=np.float64)
        high = np.asarray(getattr(action_space, "high", 0.4 * np.ones(_ACT_DIM)),
                          dtype=np.float64)
        self._low = low
        self._high = high

        n = int(np.asarray(getattr(action_space, "shape", (_ACT_DIM,))).ravel()[0]) \
            if getattr(action_space, "shape", None) else _ACT_DIM
        self._act_dim = n if n > 0 else _ACT_DIM

        # Max per-step base translation magnitude (env clips componentwise to
        # +/- max_action_mag; we additionally cap the L2 norm by the same value
        # which is conservative and always legal).
        bx = min(abs(float(low[_ACT_BASE_X])), abs(float(high[_ACT_BASE_X]))) \
            if len(low) > _ACT_BASE_X else 0.4
        by = min(abs(float(low[_ACT_BASE_Y])), abs(float(high[_ACT_BASE_Y]))) \
            if len(low) > _ACT_BASE_Y else 0.4
        self._max_step = max(1e-6, min(bx, by))

        self._dtype = getattr(action_space, "dtype", np.float32)

        self._reset_internal()

    # ------------------------------------------------------------------ #
    # Episode bookkeeping
    # ------------------------------------------------------------------ #
    def _reset_internal(self) -> None:
        self._prev_pos = None          # base (x, y) at previous get_action call
        self._prev_cmd = None          # commanded (dx, dy) at previous call
        self._detour_left = 0          # remaining sidestep steps
        self._detour_dir = None        # unit sidestep direction (world frame)
        self._detour_sign = 1.0        # alternates between recovery attempts
        self._detour_len = _MIN_DETOUR_STEPS
        self._consecutive_stalls = 0

    def reset(self, state, info):
        """Called at the start of each episode with the initial observation."""
        self._reset_internal()
        return None

    # ------------------------------------------------------------------ #
    # Policy
    # ------------------------------------------------------------------ #
    def get_action(self, state):
        obs = np.asarray(state, dtype=np.float64).ravel()

        pos = np.array([obs[_OBS_BASE_X], obs[_OBS_BASE_Y]])
        target = np.array([obs[_OBS_TARGET_X], obs[_OBS_TARGET_Y]])

        # --- detect a stall (collision-reverted step) --------------------- #
        if self._prev_pos is not None and self._prev_cmd is not None:
            commanded = float(np.linalg.norm(self._prev_cmd))
            moved = float(np.linalg.norm(pos - self._prev_pos))
            if commanded >= _STALL_CMD and moved < _STALL_MOVE:
                self._on_stall(pos, target)
            elif moved >= _STALL_MOVE:
                # Genuine progress: decay the stall counter.
                self._consecutive_stalls = 0

        # --- choose the base delta ---------------------------------------- #
        error = target - pos
        dist = float(np.linalg.norm(error))

        if dist <= _GOAL_TOL:
            # Already at the goal (termination happens in the env); emit a
            # zero base command so we do not drift back out.
            delta = np.zeros(2)
            self._detour_left = 0
        elif self._detour_left > 0 and self._detour_dir is not None:
            # Sidestep around whatever blocked us.
            self._detour_left -= 1
            delta = _clip_norm(self._detour_dir * self._max_step, self._max_step)
        else:
            # Straight-line clipped proportional drive.  When within one step
            # of the target the un-clipped error lands us exactly on it.
            delta = _clip_norm(error, self._max_step)

        # --- assemble the full action ------------------------------------- #
        action = np.zeros(self._act_dim, dtype=np.float64)
        if self._act_dim > _ACT_BASE_X:
            action[_ACT_BASE_X] = delta[0]
        if self._act_dim > _ACT_BASE_Y:
            action[_ACT_BASE_Y] = delta[1]
        # Rotation, arm joints, and gripper stay at zero: none of them affect
        # goal_reached(), and holding the arm still avoids self-collisions.
        if self._act_dim > _ACT_BASE_ROT:
            action[_ACT_BASE_ROT] = 0.0

        # Clip componentwise into the action space to be safe.
        if len(self._low) == self._act_dim and len(self._high) == self._act_dim:
            action = np.clip(action, self._low, self._high)

        self._prev_pos = pos
        self._prev_cmd = np.array([action[_ACT_BASE_X], action[_ACT_BASE_Y]]) \
            if self._act_dim > _ACT_BASE_Y else np.zeros(2)

        return action.astype(self._dtype, copy=False)

    # ------------------------------------------------------------------ #
    # Stall recovery
    # ------------------------------------------------------------------ #
    def _on_stall(self, pos: np.ndarray, target: np.ndarray) -> None:
        """Blocked by geometry: plan a perpendicular sidestep."""
        self._consecutive_stalls += 1

        error = target - pos
        norm = float(np.linalg.norm(error))
        if norm < 1e-9:
            goal_dir = np.array([1.0, 0.0])
        else:
            goal_dir = error / norm

        # Perpendicular to the goal direction, sign alternating between
        # successive recovery attempts so we try both ways around an obstacle.
        perp = np.array([-goal_dir[1], goal_dir[0]]) * self._detour_sign

        # Blend in a little forward motion so we hug the obstacle rather than
        # sliding purely sideways.
        mix = perp + 0.25 * goal_dir
        mix_norm = float(np.linalg.norm(mix))
        self._detour_dir = mix / mix_norm if mix_norm > 1e-9 else perp

        self._detour_left = self._detour_len

        # Escalate: next failure tries the other side and a longer detour.
        self._detour_sign *= -1.0
        if self._consecutive_stalls % 2 == 0:
            self._detour_len = min(_MAX_DETOUR_STEPS, self._detour_len + 2)