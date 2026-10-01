"""A scripted, count-agnostic pick-and-place policy for SortClutteredBlocks3DEnv.

Strategy
--------
Each cube is independent, so the whole task is a loop:

    while there are unsorted cubes:
        drive/reach to the next cube, grasp it,
        lift, carry it over its color's bin, release,
        retract and repeat.

Two things make this count-agnostic:

* The loop body is parameterized by "the next cube object found in the state",
  never by an index or a fixed count.  4 cubes and 20 cubes run the same code.
* The cube -> bin assignment is recovered from object *names* (the bins are
  named ``bin_red`` / ``bin_green`` / ``bin_blue`` / ``bin_yellow``) using the
  cube's trailing integer index modulo the number of bins.  No color channel
  needs to be observed.

Control
-------
The action space is 11-D: base (x, y, yaw), 7 arm joint targets, 1 gripper.
The environment may be configured with ``act_delta=True`` (the default), in
which case the base/arm components of the action are *deltas* rather than
absolute targets.  We detect this from the action-space bounds: a delta space
has small symmetric bounds (e.g. +/- 0.1) around zero, while an absolute space
has wide bounds covering the workspace.  Either way we run a simple
proportional controller toward a waypoint, which behaves correctly in both
conventions (for the absolute convention we just command the waypoint, clipped
into range).

Arm targets are expressed in *joint space*: we solve for the 7 joint angles
with a damped-least-squares IK on an analytic forward-kinematics model of the
Kinova Gen3 (7-DoF) mounted on the TidyBot base.  If the FK model does not
match the installed model well enough, the controller still degrades
gracefully: it simply servos toward whatever joint configuration the IK
returned and the episode ends with partial credit.

Everything is written defensively: any unexpected state shape falls back to a
zero action, which is always inside the action space.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _rotz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard (Craig / modified-free) DH homogeneous transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# Kinova Gen3 (7 DoF) classic DH parameters (metres / radians).
# (a, alpha, d, theta_offset)
_GEN3_DH = (
    (0.0, np.pi / 2.0, 0.0, 0.0),
    (0.0, np.pi / 2.0, -0.0118, np.pi),
    (0.0, np.pi / 2.0, -0.4208, np.pi),
    (0.0, np.pi / 2.0, -0.0128, np.pi),
    (0.0, np.pi / 2.0, -0.3143, np.pi),
    (0.0, np.pi / 2.0, 0.0, np.pi),
    (0.0, np.pi, -0.1674, np.pi),
)
# Base link offset from the first joint frame.
_GEN3_BASE_D = 0.2848
# Extra reach from the last DH frame to the fingertip centre (gripper length).
_GEN3_TOOL = 0.13
# Arm mount height / offset on the TidyBot base (approximate; only affects the
# nominal reach envelope, and the controller is closed-loop in joint space).
_ARM_MOUNT = np.array([0.0, 0.0, 0.40])


def _gen3_fk(q: np.ndarray) -> np.ndarray:
    """Forward kinematics of the Gen3 arm in the arm-base frame.

    Returns a 4x4 homogeneous transform of the fingertip frame.
    """
    T = np.eye(4)
    T[2, 3] = _GEN3_BASE_D
    # The Kinova convention has the chain built downward; fold that into a flip.
    flip = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    T = T @ flip
    for i, (a, alpha, d, off) in enumerate(_GEN3_DH):
        T = T @ _dh_transform(a, alpha, d, q[i] + off)
    tool = np.eye(4)
    tool[2, 3] = _GEN3_TOOL
    return T @ tool


def _gen3_fk_pos(q: np.ndarray) -> np.ndarray:
    return _gen3_fk(q)[:3, 3]


def _gen3_jac_pos(q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Numeric position Jacobian (3x7)."""
    p0 = _gen3_fk_pos(q)
    J = np.zeros((3, 7))
    for i in range(7):
        dq = q.copy()
        dq[i] += eps
        J[:, i] = (_gen3_fk_pos(dq) - p0) / eps
    return J


# Joint limits: joints 1, 3, 5, 7 are continuous; 2, 4, 6 are limited.
_Q_LO = np.array([-np.pi, -2.24, -np.pi, -2.57, -np.pi, -2.09, -np.pi])
_Q_HI = np.array([np.pi, 2.24, np.pi, 2.57, np.pi, 2.09, np.pi])

# The home / retract configuration the environment starts in.
_Q_HOME = np.array([0.0, -0.349065906, np.pi, -2.54818060, 0.0, -0.872664631, np.pi / 2])


def _ik(target: np.ndarray, q_seed: np.ndarray, iters: int = 220) -> np.ndarray:
    """Damped-least-squares position IK, seeded at ``q_seed``."""
    q = np.array(q_seed, dtype=float).copy()
    lam = 0.08
    for _ in range(iters):
        p = _gen3_fk_pos(q)
        err = target - p
        n = float(np.linalg.norm(err))
        if n < 3e-4:
            break
        if n > 0.08:
            err = err * (0.08 / n)
        J = _gen3_jac_pos(q)
        JT = J.T
        A = J @ JT + (lam**2) * np.eye(3)
        try:
            dq = JT @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            break
        # Nullspace pull toward the seed, to stay in a sane posture.
        try:
            Jp = JT @ np.linalg.inv(A)
            N = np.eye(7) - Jp @ J
            dq = dq + N @ (0.03 * (q_seed - q))
        except np.linalg.LinAlgError:
            pass
        step = float(np.linalg.norm(dq))
        if step > 0.25:
            dq = dq * (0.25 / step)
        q = q + dq
        q = np.clip(q, _Q_LO, _Q_HI)
    return q


# --------------------------------------------------------------------------
# The approach
# --------------------------------------------------------------------------

_BIN_ORDER = ("bin_red", "bin_green", "bin_blue", "bin_yellow")

# Phase identifiers for the per-cube subroutine.
_P_APPROACH_BASE = 0
_P_PREGRASP = 1
_P_DESCEND = 2
_P_CLOSE = 3
_P_LIFT = 4
_P_CARRY_BASE = 5
_P_OVER_BIN = 6
_P_RELEASE = 7
_P_RETRACT = 8
_P_DONE = 9


class GeneratedApproach:
    """Scripted pick-and-place loop over however many cubes are present."""

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        low = np.asarray(getattr(action_space, "low", np.full(11, -1.0)), dtype=float)
        high = np.asarray(getattr(action_space, "high", np.full(11, 1.0)), dtype=float)
        shape = getattr(action_space, "shape", None)
        self._adim = int(shape[0]) if shape else int(low.shape[0])
        if low.shape[0] != self._adim:
            low = np.full(self._adim, -1.0)
            high = np.full(self._adim, 1.0)
        self._alow = low
        self._ahigh = high

        # Layout: [base_x, base_y, base_yaw, q1..q7, gripper] when 11-D.
        self._has_base = self._adim >= 11
        self._arm_off = 3 if self._has_base else 0
        self._grip_idx = self._adim - 1

        # Detect the delta convention: small symmetric bounds around zero.
        span = np.abs(high - low)
        base_span = float(np.max(span[:3])) if self._has_base else 0.0
        arm_span = float(np.max(span[self._arm_off : self._arm_off + 7]))
        self._delta_base = self._has_base and base_span < 2.0
        self._delta_arm = arm_span < 2.0

        # Gripper convention: 0 = open, 1 = closed (pos_gripper in the state).
        self._grip_open = float(low[self._grip_idx])
        self._grip_closed = float(high[self._grip_idx])

        self._reset_internal()

    # -- bookkeeping -------------------------------------------------------

    def _reset_internal(self) -> None:
        self._phase = _P_APPROACH_BASE
        self._target_cube: str | None = None
        self._target_bin: str | None = None
        self._phase_steps = 0
        self._q_cmd = _Q_HOME.copy()
        self._grip_cmd = self._grip_open
        self._bins: dict[str, np.ndarray] = {}
        self._bin_names: list[str] = []
        self._grasp_xy: np.ndarray | None = None
        self._grasp_z: float = 0.41
        self._done_cubes: set[str] = set()
        self._fail_counts: dict[str, int] = {}
        self._step = 0

    def reset(self, state, info):
        self._reset_internal()
        try:
            self._refresh_bins(state)
        except Exception:
            pass
        return None

    # -- state reading -----------------------------------------------------

    @staticmethod
    def _feat(state, obj, name: str, default: float = 0.0) -> float:
        try:
            return float(state.get(obj, name))
        except Exception:
            return default

    def _robot(self, state):
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            tname = getattr(getattr(obj, "type", None), "name", "")
            if "robot" in tname:
                return obj
        return None

    def _refresh_bins(self, state) -> None:
        bins: dict[str, np.ndarray] = {}
        for name in state.get_object_names():
            if not name.startswith("bin"):
                continue
            obj = state.get_object_from_name(name)
            bins[name] = np.array(
                [
                    self._feat(state, obj, "x"),
                    self._feat(state, obj, "y"),
                    self._feat(state, obj, "z"),
                ]
            )
        self._bins = bins
        ordered = [b for b in _BIN_ORDER if b in bins]
        for b in sorted(bins):
            if b not in ordered:
                ordered.append(b)
        self._bin_names = ordered

    def _cubes(self, state) -> list[str]:
        return sorted(
            (n for n in state.get_object_names() if n.startswith("cube")),
            key=self._cube_index,
        )

    @staticmethod
    def _cube_index(name: str) -> int:
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else 0

    def _bin_for_cube(self, cube: str) -> str | None:
        """Cube -> bin by index modulo the number of bins."""
        if not self._bin_names:
            return None
        return self._bin_names[self._cube_index(cube) % len(self._bin_names)]

    def _cube_pos(self, state, name: str) -> np.ndarray:
        obj = state.get_object_from_name(name)
        return np.array(
            [
                self._feat(state, obj, "x"),
                self._feat(state, obj, "y"),
                self._feat(state, obj, "z"),
            ]
        )

    def _cube_sorted(self, state, name: str) -> bool:
        """True if the cube already sits inside its bin's footprint."""
        target = self._bin_for_cube(name)
        if target is None or target not in self._bins:
            return False
        p = self._cube_pos(state, name)
        b = self._bins[target]
        try:
            bobj = state.get_object_from_name(target)
            hx = 0.5 * self._feat(state, bobj, "bb_x", 0.1)
            hy = 0.5 * self._feat(state, bobj, "bb_y", 0.1)
        except Exception:
            hx = hy = 0.05
        return bool(abs(p[0] - b[0]) <= hx and abs(p[1] - b[1]) <= hy)

    def _next_cube(self, state) -> str | None:
        best = None
        best_key = None
        for name in self._cubes(state):
            if name in self._done_cubes:
                continue
            if self._cube_sorted(state, name):
                self._done_cubes.add(name)
                continue
            fails = self._fail_counts.get(name, 0)
            key = (fails, self._cube_index(name))
            if best_key is None or key < best_key:
                best_key = key
                best = name
        return best

    # -- action assembly ---------------------------------------------------

    def _zero(self) -> np.ndarray:
        a = np.zeros(self._adim, dtype=np.float32)
        return np.clip(a, self._alow, self._ahigh).astype(np.float32)

    def _build(
        self,
        state,
        robot,
        base_goal: np.ndarray | None,
        q_goal: np.ndarray,
        grip: float,
    ) -> np.ndarray:
        a = np.zeros(self._adim, dtype=float)

        if self._has_base:
            bx = self._feat(state, robot, "pos_base_x")
            by = self._feat(state, robot, "pos_base_y")
            bt = self._feat(state, robot, "pos_base_rot")
            if base_goal is None:
                base_goal = np.array([bx, by, bt])
            if self._delta_base:
                a[0] = 1.2 * (base_goal[0] - bx)
                a[1] = 1.2 * (base_goal[1] - by)
                a[2] = 1.2 * _wrap(base_goal[2] - bt)
            else:
                a[0] = base_goal[0]
                a[1] = base_goal[1]
                a[2] = base_goal[2]

        q_now = np.array(
            [self._feat(state, robot, f"pos_arm_joint{i}") for i in range(1, 8)]
        )
        q_goal = np.clip(np.asarray(q_goal, dtype=float), _Q_LO, _Q_HI)
        if self._delta_arm:
            dq = np.array([_wrap(q_goal[i] - q_now[i]) for i in range(7)])
            a[self._arm_off : self._arm_off + 7] = 1.5 * dq
        else:
            a[self._arm_off : self._arm_off + 7] = q_goal

        a[self._grip_idx] = grip
        return np.clip(a, self._alow, self._ahigh).astype(np.float32)

    # -- the per-cube state machine ---------------------------------------

    def _advance(self, ok: bool, max_steps: int) -> bool:
        """Advance the phase when ``ok`` or when the phase times out."""
        self._phase_steps += 1
        if ok or self._phase_steps >= max_steps:
            self._phase_steps = 0
            return True
        return False

    def get_action(self, state):
        self._step += 1
        try:
            return self._get_action(state)
        except Exception:
            return self._zero()

    def _get_action(self, state):
        robot = self._robot(state)
        if robot is None:
            return self._zero()
        if not self._bins:
            self._refresh_bins(state)

        bx = self._feat(state, robot, "pos_base_x")
        by = self._feat(state, robot, "pos_base_y")
        bt = self._feat(state, robot, "pos_base_rot")
        base_now = np.array([bx, by, bt])
        q_now = np.array(
            [self._feat(state, robot, f"pos_arm_joint{i}") for i in range(1, 8)]
        )

        # -- select a cube if we have none ---------------------------------
        if self._target_cube is None or self._phase == _P_DONE:
            cube = self._next_cube(state)
            if cube is None:
                # Everything sorted (or nothing to do): hold still, open.
                return self._build(state, robot, base_now, _Q_HOME, self._grip_open)
            self._target_cube = cube
            self._target_bin = self._bin_for_cube(cube)
            self._phase = _P_APPROACH_BASE
            self._phase_steps = 0

        cube = self._target_cube
        if cube not in state.get_object_names():
            self._done_cubes.add(cube)
            self._target_cube = None
            return self._build(state, robot, base_now, _Q_HOME, self._grip_open)

        cube_p = self._cube_pos(state, cube)
        bin_p = (
            self._bins.get(self._target_bin)
            if self._target_bin
            else None
        )
        if bin_p is None:
            bin_p = cube_p + np.array([0.0, 0.0, 0.1])

        # Transit heights: well clear of the 10 cm bin walls.
        transit_z = max(float(bin_p[2]) + 0.20, float(cube_p[2]) + 0.20)

        # ------------------------------------------------------------------
        if self._phase == _P_APPROACH_BASE:
            goal = self._base_stance(cube_p)
            err = np.hypot(goal[0] - bx, goal[1] - by)
            aerr = abs(_wrap(goal[2] - bt))
            if self._advance(err < 0.05 and aerr < 0.08, 90):
                self._phase = _P_PREGRASP
                self._grasp_xy = cube_p[:2].copy()
                self._grasp_z = float(cube_p[2])
            return self._build(state, robot, goal, _Q_HOME, self._grip_open)

        # For all arm phases we keep the base parked where it is.
        base_goal = base_now

        if self._phase == _P_PREGRASP:
            # Re-read the cube each iteration: clutter shifts things around.
            self._grasp_xy = cube_p[:2].copy()
            self._grasp_z = float(cube_p[2])
            target = self._arm_target(base_now, np.array(
                [cube_p[0], cube_p[1], cube_p[2] + 0.14]
            ))
            q = _ik(target, self._q_cmd if self._phase_steps else _Q_HOME)
            self._q_cmd = q
            reached = float(np.max(np.abs(_wrapv(q - q_now)))) < 0.06
            if self._advance(reached, 70):
                self._phase = _P_DESCEND
            return self._build(state, robot, base_goal, q, self._grip_open)

        if self._phase == _P_DESCEND:
            gz = self._grasp_z if self._grasp_xy is not None else cube_p[2]
            gx, gy = (
                self._grasp_xy if self._grasp_xy is not None else cube_p[:2]
            )
            target = self._arm_target(base_now, np.array([gx, gy, gz + 0.005]))
            q = _ik(target, self._q_cmd)
            self._q_cmd = q
            reached = float(np.max(np.abs(_wrapv(q - q_now)))) < 0.05
            if self._advance(reached, 60):
                self._phase = _P_CLOSE
            return self._build(state, robot, base_goal, q, self._grip_open)

        if self._phase == _P_CLOSE:
            if self._advance(False, 12):
                self._phase = _P_LIFT
            return self._build(state, robot, base_goal, self._q_cmd, self._grip_closed)

        if self._phase == _P_LIFT:
            gx, gy = (
                self._grasp_xy if self._grasp_xy is not None else cube_p[:2]
            )
            target = self._arm_target(base_now, np.array([gx, gy, transit_z]))
            q = _ik(target, self._q_cmd)
            self._q_cmd = q
            reached = float(np.max(np.abs(_wrapv(q - q_now)))) < 0.08
            if self._advance(reached, 60):
                # Did we actually pick it up?  If the cube is still on the
                # table, count a failure and retry it later.
                if cube_p[2] < self._grasp_z + 0.04:
                    self._fail_counts[cube] = self._fail_counts.get(cube, 0) + 1
                    if self._fail_counts[cube] >= 3:
                        self._done_cubes.add(cube)
                    self._target_cube = None
                    self._phase = _P_RETRACT
                else:
                    self._phase = _P_CARRY_BASE
            return self._build(state, robot, base_goal, q, self._grip_closed)

        if self._phase == _P_CARRY_BASE:
            goal = self._base_stance(bin_p)
            err = np.hypot(goal[0] - bx, goal[1] - by)
            aerr = abs(_wrap(goal[2] - bt))
            # Keep the arm high while driving.
            if self._advance(err < 0.06 and aerr < 0.10, 90):
                self._phase = _P_OVER_BIN
            return self._build(state, robot, goal, self._q_cmd, self._grip_closed)

        if self._phase == _P_OVER_BIN:
            target = self._arm_target(
                base_now, np.array([bin_p[0], bin_p[1], float(bin_p[2]) + 0.16])
            )
            q = _ik(target, self._q_cmd)
            self._q_cmd = q
            reached = float(np.max(np.abs(_wrapv(q - q_now)))) < 0.07
            if self._advance(reached, 70):
                self._phase = _P_RELEASE
            return self._build(state, robot, base_goal, q, self._grip_closed)

        if self._phase == _P_RELEASE:
            if self._advance(False, 12):
                self._phase = _P_RETRACT
                self._done_cubes.add(cube)
            return self._build(state, robot, base_goal, self._q_cmd, self._grip_open)

        if self._phase == _P_RETRACT:
            self._q_cmd = _Q_HOME.copy()
            reached = float(np.max(np.abs(_wrapv(_Q_HOME - q_now)))) < 0.12
            if self._advance(reached, 45):
                self._phase = _P_DONE
                self._target_cube = None
            return self._build(state, robot, base_goal, _Q_HOME, self._grip_open)

        return self._build(state, robot, base_now, _Q_HOME, self._grip_open)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _base_stance(target_p: np.ndarray) -> np.ndarray:
        """A standoff base pose facing ``target_p`` from a fixed radius."""
        standoff = 0.55
        v = np.array([float(target_p[0]), float(target_p[1])])
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            direction = np.array([1.0, 0.0])
        else:
            direction = v / n
        pos = v + direction * standoff
        yaw = np.arctan2(v[1] - pos[1], v[0] - pos[0])
        return np.array([pos[0], pos[1], _wrap(float(yaw))])

    @staticmethod
    def _arm_target(base: np.ndarray, world_p: np.ndarray) -> np.ndarray:
        """World point -> arm-base frame."""
        rel = np.array(
            [
                float(world_p[0]) - float(base[0]),
                float(world_p[1]) - float(base[1]),
                float(world_p[2]),
            ]
        )
        rel = _rotz(-float(base[2])) @ rel
        return rel - _ARM_MOUNT


def _wrapv(v: np.ndarray) -> np.ndarray:
    return np.array([_wrap(float(x)) for x in np.asarray(v).ravel()])