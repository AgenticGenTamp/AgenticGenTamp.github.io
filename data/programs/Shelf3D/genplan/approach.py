"""
GeneratedApproach for Shelf3DEnv (variable object count).

=============================================================================
HONEST PREFACE (please read before trusting this)
=============================================================================
In my previous message I said I could not name the intended strategy without
inspecting the task JSON (goal_state / shelf regions) and the concrete
TidyBot3DRobotActionSpace semantics.  That is still true.  I have now been
asked to write code anyway, so I am writing the most defensible thing I can
given the uncertainty, and I am documenting the assumptions IN the code so
that whoever runs it can see exactly which one broke.

Key unresolved ambiguities, and how this module handles each:

(A) ACTION SEMANTICS.  The env card says the robot controls
    "base pose (x,y,theta), arm position (x,y,z), arm orientation
    (quaternion), gripper".  That is 3+3+4+1 = 11.  The action-space summary
    line says "base pos and yaw (3), arm joints (7), gripper pos (1)" = 11
    as well.  Same width, DIFFERENT meaning.  Worse, TidyBot3DConfig has
    act_delta=True by default, so entries may be deltas rather than absolute
    targets.  I cannot resolve this from the sources provided.
    -> This module inspects action_space.shape / .low / .high at __init__ and
       picks a layout heuristically (see _infer_action_layout).  If the layout
       is 11-wide it assumes [base(3), arm(7), gripper(1)] because that is
       what the *action space description* (the machine-generated one) says,
       and because the robot state features are joint-space
       (pos_arm_joint1..7).  If the width is not 11 it degrades to a safe
       hold-still action.

(B) TARGET LOCATIONS.  Reward is "+1.0 per object within 5cm of its target",
    but the targets live in goal_state predicates inside
    tasks/Shelf3D/Shelf3D-o<k>.json, which I have not seen.  I therefore do
    NOT know the shelf slot coordinates.
    -> This module derives an approximate shelf pose from the cupboard
       fixture pose in the state (position + yaw), and spreads targets across
       three layers at plausible heights.  These numbers are GUESSES and are
       collected in the SHELF_GUESS block below so they are easy to correct
       once the JSON is read.

(C) IK.  Commanding joints (per (A)) means reaching a Cartesian shelf pose
    requires inverse kinematics for a Kinova Gen3.  There is no IK primitive
    available (`primitives` is an empty dict) and no mujoco model handle.
    -> This module does NOT attempt real IK.  It uses a small set of scripted
       joint postures (floor-reach / carry / shelf-reach) interpolated with a
       proportional joint controller, and relies on the BASE to do the
       Cartesian positioning.  This is the honest limit of what can be done
       blind, and it is the single most likely reason the policy fails.

WHAT I AM ACTUALLY CONFIDENT ABOUT (from the code, not from priors):
    - max_steps_for_count = 300 + 150*k is a PER-OBJECT budget, and
      design/eval counts sweep k.  So the control structure must be a loop
      over however many cube* objects are in the state, one at a time, with a
      per-object step budget and a hard timeout that abandons a cube and
      moves on.  That structure is what generalises to unbounded k.
    - Objects must be read via get_objects / get_object_from_name, never by
      index.
    - The policy must never crash and must always return an in-bounds action.
That structural skeleton is what this module really delivers; the per-object
skill inside it is a best-effort guess.
=============================================================================
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# SHELF_GUESS -- all the numbers I could not verify, in one place.
# If you read tasks/Shelf3D/Shelf3D-o*.json, fix these first.
# =============================================================================

# Cupboard opening faces the robot.  In the example states the cupboard sits at
# x ~= 1.5, y ~= 0, with quaternion (0.7071, 0, 0, 0.7071) i.e. yaw = +pi/2.
# I assume the shelf interior is offset from the fixture origin along the
# cupboard's local -x (toward the robot at the origin side).
SHELF_LAYER_HEIGHTS = (0.35, 0.62, 0.89)   # metres, guess: three layers
SHELF_DEPTH_OFFSET = 0.18                  # into the cupboard from its origin
SHELF_HALF_WIDTH = 0.16                    # lateral spread for multiple cubes
SHELF_SLOT_PITCH = 0.09                    # spacing between placed cubes

# Where the base should park to service the shelf: stand off from the cupboard
# by this distance along the approach direction.
BASE_STANDOFF = 0.62

# Where the base should park to pick a floor cube.
PICK_STANDOFF = 0.42

# Cube pick height assumptions.
FLOOR_CUBE_Z = 0.02

# =============================================================================


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------

def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw (rotation about z) from a (w, x, y, z) quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clip(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ---------------------------------------------------------------------------
# Scripted arm postures (7 joints, radians).
#
# These are anchored on the arm posture seen in the example initial states:
#   [0.0, -0.349, 3.1416, -2.548, 0.0, -0.873, 1.5708]
# which is presumably the "home / tucked" pose.  The others are hand-derived
# perturbations of it.  They are GUESSES (see preface note (C)).
# ---------------------------------------------------------------------------

HOME_ARM = (0.0, -0.349, 3.1416, -2.548, 0.0, -0.873, 1.5708)

# Lean the arm down and forward to reach something on the floor in front.
FLOOR_REACH_ARM = (0.0, 0.55, 3.1416, -2.05, 0.0, -0.75, 1.5708)

# Slightly retracted, cube held, safe for driving.
CARRY_ARM = (0.0, -0.20, 3.1416, -2.45, 0.0, -0.80, 1.5708)

# Extend forward and up to place onto a shelf; layer index selects a variant.
SHELF_REACH_ARM = (
    (0.0, 0.10, 3.1416, -2.15, 0.0, -0.95, 1.5708),   # low layer
    (0.0, -0.10, 3.1416, -2.05, 0.0, -1.05, 1.5708),  # middle layer
    (0.0, -0.35, 3.1416, -1.90, 0.0, -1.15, 1.5708),  # high layer
)

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0


# ---------------------------------------------------------------------------
# Phase constants for the per-cube skill.
# ---------------------------------------------------------------------------

PH_DRIVE_TO_CUBE = "drive_to_cube"
PH_LOWER = "lower"
PH_GRASP = "grasp"
PH_LIFT = "lift"
PH_DRIVE_TO_SHELF = "drive_to_shelf"
PH_EXTEND = "extend"
PH_RELEASE = "release"
PH_RETRACT = "retract"
PH_DONE = "done"

# Per-phase step budgets.  Deliberately generous but bounded, so that a stuck
# phase cannot eat the whole episode -- the per-object timeout below is the
# real safety net.
PHASE_BUDGET = {
    PH_DRIVE_TO_CUBE: 45,
    PH_LOWER: 18,
    PH_GRASP: 12,
    PH_LIFT: 15,
    PH_DRIVE_TO_SHELF: 55,
    PH_EXTEND: 22,
    PH_RELEASE: 10,
    PH_RETRACT: 15,
}


class GeneratedApproach:
    """Sequential pick-and-place over however many cubes are present.

    The generalisation story is the loop, not the manipulation:
      * enumerate cube* objects from the state at reset AND re-check each step,
      * service them one at a time in a stable order,
      * give each cube a bounded step budget and abandon it on timeout,
      * skip cubes that already read as placed,
      * fall back to a benign hold action whenever anything is unknown.
    """

    # -- construction -------------------------------------------------------

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives or {}

        # Action space bounds, used for clipping and for layout inference.
        self._act_dim, self._act_low, self._act_high = self._read_action_bounds(
            action_space
        )
        self._layout = self._infer_action_layout(self._act_dim)

        # Whether the action is interpreted as a delta.  TidyBot3DConfig
        # defaults act_delta=True.  We cannot observe this, so we assume
        # absolute-target semantics for the base/arm (which is what a
        # position-controlled setpoint scheme looks like) but we ALSO keep the
        # commands smooth and near the current measured state, which makes the
        # controller behave sanely under either interpretation: a small
        # setpoint step is also a small delta.
        self._smooth_step_base = 0.06     # m or rad per control step
        self._smooth_step_arm = 0.12      # rad per control step

        # Episode state.
        self._t = 0
        self._queue: List[str] = []
        self._current: Optional[str] = None
        self._phase: str = PH_DONE
        self._phase_t = 0
        self._object_t = 0
        self._object_budget = 400
        self._placed: Dict[str, bool] = {}
        self._abandoned: Dict[str, bool] = {}
        self._slot_for: Dict[str, Tuple[float, float, float]] = {}
        self._robot_name: Optional[str] = None
        self._cupboard_name: Optional[str] = None
        self._next_slot_index = 0

    # -- action-space introspection ----------------------------------------

    @staticmethod
    def _read_action_bounds(action_space) -> Tuple[int, np.ndarray, np.ndarray]:
        """Extract (dim, low, high) defensively from whatever we were handed."""
        dim = 11
        try:
            shape = getattr(action_space, "shape", None)
            if shape is not None and len(shape) >= 1:
                dim = int(shape[-1])
        except Exception:
            pass

        low = np.full(dim, -1.0, dtype=np.float32)
        high = np.full(dim, 1.0, dtype=np.float32)
        try:
            raw_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
            raw_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
            if raw_low.size == dim and raw_high.size == dim:
                low = raw_low
                high = raw_high
        except Exception:
            pass

        # Guard against infinities so clipping is always meaningful.
        low = np.where(np.isfinite(low), low, -10.0).astype(np.float32)
        high = np.where(np.isfinite(high), high, 10.0).astype(np.float32)
        return dim, low, high

    @staticmethod
    def _infer_action_layout(dim: int) -> Optional[Dict[str, slice]]:
        """Guess which slots mean what.

        11 -> base(0:3), arm joints(3:10), gripper(10:11).  This follows the
        machine-generated action-space description quoted in the env card and
        matches the joint-space robot features.  Any other width: we admit we
        do not know and return None, which forces a hold-still policy.
        """
        if dim == 11:
            return {"base": slice(0, 3), "arm": slice(3, 10), "grip": slice(10, 11)}
        return None

    # -- state reading ------------------------------------------------------

    def _find_robot(self, state) -> Optional[Any]:
        """Robot object, located by feature signature rather than by name."""
        if self._robot_name is not None:
            try:
                return state.get_object_from_name(self._robot_name)
            except Exception:
                self._robot_name = None
        for name in state.get_object_names():
            try:
                obj = state.get_object_from_name(name)
                feats = state.type_features[obj.type]
            except Exception:
                continue
            if "pos_base_x" in feats and "pos_arm_joint1" in feats:
                self._robot_name = name
                return obj
        return None

    def _find_cupboard(self, state) -> Optional[Any]:
        """Fixture object (the cupboard).  Name is not assumed."""
        if self._cupboard_name is not None:
            try:
                return state.get_object_from_name(self._cupboard_name)
            except Exception:
                self._cupboard_name = None
        best = None
        for name in state.get_object_names():
            try:
                obj = state.get_object_from_name(name)
                feats = state.type_features[obj.type]
            except Exception:
                continue
            # A fixture has pose but no velocity and no bounding box.
            if "x" in feats and "qw" in feats and "vx" not in feats \
                    and "pos_base_x" not in feats:
                best = obj
                self._cupboard_name = name
                break
        return best

    @staticmethod
    def _cube_names(state) -> List[str]:
        """All count-defining objects, in a stable order, however many exist."""
        names = []
        for name in state.get_object_names():
            if not name.startswith("cube"):
                continue
            try:
                obj = state.get_object_from_name(name)
                feats = state.type_features[obj.type]
            except Exception:
                continue
            if "bb_x" in feats and "vx" in feats:
                names.append(name)

        def _key(n: str):
            suffix = n[4:]
            return (0, int(suffix)) if suffix.isdigit() else (1, 0, n)

        names.sort(key=_key)
        return names

    @staticmethod
    def _pos(state, obj) -> Tuple[float, float, float]:
        return (
            float(state.get(obj, "x")),
            float(state.get(obj, "y")),
            float(state.get(obj, "z")),
        )

    def _robot_readings(self, state, robot) -> Dict[str, Any]:
        base = (
            float(state.get(robot, "pos_base_x")),
            float(state.get(robot, "pos_base_y")),
            float(state.get(robot, "pos_base_rot")),
        )
        arm = tuple(
            float(state.get(robot, "pos_arm_joint%d" % i)) for i in range(1, 8)
        )
        grip = float(state.get(robot, "pos_gripper"))
        return {"base": base, "arm": arm, "grip": grip}

    # -- shelf geometry (all guesswork; see SHELF_GUESS) --------------------

    def _shelf_frame(self, state) -> Tuple[float, float, float]:
        """(x, y, yaw) of the cupboard, or a fallback if it is missing."""
        cup = self._find_cupboard(state)
        if cup is None:
            return (1.5, 0.0, math.pi / 2.0)
        try:
            x = float(state.get(cup, "x"))
            y = float(state.get(cup, "y"))
            yaw = _quat_to_yaw(
                float(state.get(cup, "qw")),
                float(state.get(cup, "qx")),
                float(state.get(cup, "qy")),
                float(state.get(cup, "qz")),
            )
            return (x, y, yaw)
        except Exception:
            return (1.5, 0.0, math.pi / 2.0)

    def _approach_direction(self, state) -> Tuple[float, float]:
        """Unit vector pointing from the cupboard back toward the workspace.

        Guess: the opening faces the world origin side, so we approach from
        whichever side the robot currently is.  Using the robot's own position
        avoids hardcoding a face and is at least self-consistent.
        """
        cx, cy, _ = self._shelf_frame(state)
        robot = self._find_robot(state)
        if robot is None:
            return (-1.0, 0.0)
        rx = float(state.get(robot, "pos_base_x"))
        ry = float(state.get(robot, "pos_base_y"))
        dx, dy = rx - cx, ry - cy
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return (-1.0, 0.0)
        return (dx / n, dy / n)

    def _slot_target(self, state, slot_index: int) -> Tuple[float, float, float]:
        """A guessed shelf slot pose for the slot_index-th placed cube.

        Layers fill bottom-up; within a layer, slots spread laterally.
        Unbounded in principle: slot_index may exceed 3 * slots_per_layer, in
        which case we wrap and stack tighter, because the count is unbounded.
        """
        cx, cy, _ = self._shelf_frame(state)
        ax, ay = self._approach_direction(state)
        # Lateral axis, perpendicular to the approach direction.
        lx, ly = -ay, ax

        slots_per_layer = max(
            1, int((2.0 * SHELF_HALF_WIDTH) / SHELF_SLOT_PITCH) + 1
        )
        layer = (slot_index // slots_per_layer) % len(SHELF_LAYER_HEIGHTS)
        within = slot_index % slots_per_layer
        lateral = -SHELF_HALF_WIDTH + within * SHELF_SLOT_PITCH
        lateral = _clip(lateral, -SHELF_HALF_WIDTH, SHELF_HALF_WIDTH)

        # Move from the cupboard origin toward the opening, then inward.
        px = cx + ax * (-SHELF_DEPTH_OFFSET) + lx * lateral
        py = cy + ay * (-SHELF_DEPTH_OFFSET) + ly * lateral
        pz = SHELF_LAYER_HEIGHTS[layer]
        return (px, py, pz)

    def _layer_of_slot(self, slot_index: int) -> int:
        slots_per_layer = max(
            1, int((2.0 * SHELF_HALF_WIDTH) / SHELF_SLOT_PITCH) + 1
        )
        return (slot_index // slots_per_layer) % len(SHELF_LAYER_HEIGHTS)

    def _is_placed(self, state, cube_name: str) -> bool:
        """Has this cube plausibly reached a shelf?

        Without the real goal regions I fall back on height: a cube resting on
        the floor is at z ~= 0.02, so anything well above that has at least
        left the ground.  This is a WEAK proxy and will call a lifted cube
        'placed' if used carelessly -- so it is only consulted for cubes we
        are not currently carrying.
        """
        try:
            obj = state.get_object_from_name(cube_name)
        except Exception:
            return False
        try:
            z = float(state.get(obj, "z"))
            vz = abs(float(state.get(obj, "vz")))
        except Exception:
            return False
        lowest_layer = min(SHELF_LAYER_HEIGHTS)
        return z > (lowest_layer - 0.12) and vz < 0.05

    # -- episode lifecycle --------------------------------------------------

    def reset(self, state, info):
        self._t = 0
        self._placed = {}
        self._abandoned = {}
        self._slot_for = {}
        self._next_slot_index = 0
        self._robot_name = None
        self._cupboard_name = None

        try:
            names = self._cube_names(state)
        except Exception:
            names = []
        self._queue = list(names)

        # Budget per object, derived from the env's own scaling rule
        # (300 + 150*k over k objects) with headroom for the shared overhead.
        k = max(1, len(self._queue))
        self._object_budget = int(150 + 300.0 / k)

        self._current = None
        self._phase = PH_DONE
        self._phase_t = 0
        self._object_t = 0
        self._advance_object(state)

    def _advance_object(self, state):
        """Pop the next unfinished cube, or go idle if none remain."""
        while self._queue:
            name = self._queue.pop(0)
            if self._abandoned.get(name):
                continue
            if self._is_placed(state, name):
                self._placed[name] = True
                continue
            self._current = name
            self._slot_for[name] = self._slot_target(state, self._next_slot_index)
            self._current_layer = self._layer_of_slot(self._next_slot_index)
            self._next_slot_index += 1
            self._phase = PH_DRIVE_TO_CUBE
            self._phase_t = 0
            self._object_t = 0
            return
        self._current = None
        self._phase = PH_DONE

    # -- action construction ------------------------------------------------

    def _hold_action(self, state) -> np.ndarray:
        """A benign in-bounds action: keep the measured pose, gripper as-is.

        Used whenever the layout is unknown, the state is unreadable, or the
        plan has finished.  Under absolute-setpoint semantics this holds still;
        under delta semantics a near-zero command also holds still, so it is
        safe under both readings of the ambiguity in preface note (A).
        """
        act = np.zeros(self._act_dim, dtype=np.float32)
        if self._layout is None:
            return np.clip(act, self._act_low, self._act_high).astype(np.float32)
        robot = self._find_robot(state)
        if robot is None:
            return np.clip(act, self._act_low, self._act_high).astype(np.float32)
        try:
            r = self._robot_readings(state, robot)
        except Exception:
            return np.clip(act, self._act_low, self._act_high).astype(np.float32)
        act[self._layout["base"]] = np.asarray(r["base"], dtype=np.float32)
        act[self._layout["arm"]] = np.asarray(r["arm"], dtype=np.float32)
        act[self._layout["grip"]] = np.float32(r["grip"])
        return np.clip(act, self._act_low, self._act_high).astype(np.float32)

    def _compose(
        self,
        state,
        base_target: Sequence[float],
        arm_target: Sequence[float],
        grip_target: float,
    ) -> np.ndarray:
        """Rate-limited setpoint toward the requested targets."""
        act = np.zeros(self._act_dim, dtype=np.float32)
        if self._layout is None:
            return np.clip(act, self._act_low, self._act_high).astype(np.float32)

        robot = self._find_robot(state)
        if robot is None:
            return self._hold_action(state)
        try:
            r = self._robot_readings(state, robot)
        except Exception:
            return self._hold_action(state)

        bx, by, bth = r["base"]
        tx, ty, tth = base_target
        # Rate-limit translation.
        dx, dy = tx - bx, ty - by
        dist = math.hypot(dx, dy)
        if dist > self._smooth_step_base and dist > 1e-9:
            scale = self._smooth_step_base / dist
            dx *= scale
            dy *= scale
        dth = _wrap_angle(tth - bth)
        dth = _clip(dth, -self._smooth_step_base, self._smooth_step_base)
        base_cmd = (bx + dx, by + dy, bth + dth)

        arm_cmd = []
        for cur, tgt in zip(r["arm"], arm_target):
            d = _clip(tgt - cur, -self._smooth_step_arm, self._smooth_step_arm)
            arm_cmd.append(cur + d)

        grip_cmd = _clip(float(grip_target), 0.0, 1.0)

        act[self._layout["base"]] = np.asarray(base_cmd, dtype=np.float32)
        act[self._layout["arm"]] = np.asarray(arm_cmd, dtype=np.float32)
        act[self._layout["grip"]] = np.float32(grip_cmd)
        return np.clip(act, self._act_low, self._act_high).astype(np.float32)

    # -- base pose helpers --------------------------------------------------

    @staticmethod
    def _stand_off_pose(
        target_xy: Tuple[float, float],
        from_xy: Tuple[float, float],
        standoff: float,
    ) -> Tuple[float, float, float]:
        """Park `standoff` metres from target, facing it, approached from
        `from_xy`."""
        tx, ty = target_xy
        fx, fy = from_xy
        dx, dy = fx - tx, fy - ty
        n = math.hypot(dx, dy)
        if n < 1e-6:
            dx, dy, n = -1.0, 0.0, 1.0
        ux, uy = dx / n, dy / n
        px, py = tx + ux * standoff, ty + uy * standoff
        yaw = math.atan2(ty - py, tx - px)
        return (px, py, yaw)

    @staticmethod
    def _base_error(
        cur: Tuple[float, float, float], tgt: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        d = math.hypot(tgt[0] - cur[0], tgt[1] - cur[1])
        a = abs(_wrap_angle(tgt[2] - cur[2]))
        return d, a

    @staticmethod
    def _arm_error(cur: Sequence[float], tgt: Sequence[float]) -> float:
        return max(abs(c - t) for c, t in zip(cur, tgt)) if cur else 0.0

    # -- main entry point ---------------------------------------------------

    def get_action(self, state):
        """Always returns an in-bounds action; never raises."""
        self._t += 1
        try:
            return self._get_action_inner(state)
        except Exception:
            # Any surprise in the state schema degrades to holding still
            # rather than crashing the rollout.
            try:
                return self._hold_action(state)
            except Exception:
                return np.zeros(self._act_dim, dtype=np.float32)

    def _get_action_inner(self, state):
        if self._layout is None:
            # We do not know what the action slots mean (preface note (A)).
            # Refusing to flail is better than emitting confident garbage.
            return self._hold_action(state)

        robot = self._find_robot(state)
        if robot is None:
            return self._hold_action(state)
        r = self._robot_readings(state, robot)

        # Late-arriving objects (or a re-plan after set_state) are picked up
        # here, so the loop never depends on the reset-time snapshot alone.
        if self._current is None and self._phase == PH_DONE:
            known = set(self._placed) | set(self._abandoned)
            if self._current is not None:
                known.add(self._current)
            fresh = [
                n for n in self._cube_names(state)
                if n not in known and n not in self._queue
            ]
            if fresh:
                self._queue.extend(fresh)
                self._advance_object(state)

        if self._current is None:
            return self._hold_action(state)

        # Per-object timeout: abandon and move on, so one bad cube cannot
        # consume the whole (300 + 150*k) budget.
        self._object_t += 1
        if self._object_t > self._object_budget:
            self._abandoned[self._current] = True
            self._advance_object(state)
            if self._current is None:
                return self._hold_action(state)

        # Per-phase timeout: nudge the machine forward if a phase stalls.
        self._phase_t += 1
        if self._phase_t > PHASE_BUDGET.get(self._phase, 30):
            self._force_phase_advance(state)

        try:
            cube = state.get_object_from_name(self._current)
            cube_xyz = self._pos(state, cube)
        except Exception:
            self._abandoned[self._current] = True
            self._advance_object(state)
            return self._hold_action(state)

        slot = self._slot_for.get(self._current) or self._slot_target(state, 0)
        layer = getattr(self, "_current_layer", 1)
        layer = int(_clip(layer, 0, len(SHELF_REACH_ARM) - 1))

        base_now = r["base"]
        arm_now = r["arm"]

        # ---- phase machine -------------------------------------------------

        if self._phase == PH_DRIVE_TO_CUBE:
            tgt = self._stand_off_pose(
                (cube_xyz[0], cube_xyz[1]), (base_now[0], base_now[1]), PICK_STANDOFF
            )
            d, a = self._base_error(base_now, tgt)
            if d < 0.10 and a < 0.15:
                self._set_phase(PH_LOWER)
            return self._compose(state, tgt, HOME_ARM, GRIPPER_OPEN)

        if self._phase == PH_LOWER:
            tgt = self._stand_off_pose(
                (cube_xyz[0], cube_xyz[1]), (base_now[0], base_now[1]), PICK_STANDOFF
            )
            if self._arm_error(arm_now, FLOOR_REACH_ARM) < 0.08:
                self._set_phase(PH_GRASP)
            return self._compose(state, tgt, FLOOR_REACH_ARM, GRIPPER_OPEN)

        if self._phase == PH_GRASP:
            tgt = (base_now[0], base_now[1], base_now[2])
            if self._phase_t > 8:
                self._set_phase(PH_LIFT)
            return self._compose(state, tgt, FLOOR_REACH_ARM, GRIPPER_CLOSED)

        if self._phase == PH_LIFT:
            tgt = (base_now[0], base_now[1], base_now[2])
            if self._arm_error(arm_now, CARRY_ARM) < 0.10:
                self._set_phase(PH_DRIVE_TO_SHELF)
            return self._compose(state, tgt, CARRY_ARM, GRIPPER_CLOSED)

        if self._phase == PH_DRIVE_TO_SHELF:
            cx, cy, _ = self._shelf_frame(state)
            tgt = self._stand_off_pose(
                (cx, cy), (base_now[0], base_now[1]), BASE_STANDOFF
            )
            d, a = self._base_error(base_now, tgt)
            if d < 0.12 and a < 0.18:
                self._set_phase(PH_EXTEND)
            return self._compose(state, tgt, CARRY_ARM, GRIPPER_CLOSED)

        if self._phase == PH_EXTEND:
            tgt = (base_now[0], base_now[1], base_now[2])
            posture = SHELF_REACH_ARM[layer]
            if self._arm_error(arm_now, posture) < 0.10:
                self._set_phase(PH_RELEASE)
            return self._compose(state, tgt, posture, GRIPPER_CLOSED)

        if self._phase == PH_RELEASE:
            tgt = (base_now[0], base_now[1], base_now[2])
            posture = SHELF_REACH_ARM[layer]
            if self._phase_t > 6:
                self._set_phase(PH_RETRACT)
            return self._compose(state, tgt, posture, GRIPPER_OPEN)

        if self._phase == PH_RETRACT:
            # Back the base away while tucking, so the arm clears the shelf.
            cx, cy, _ = self._shelf_frame(state)
            tgt = self._stand_off_pose(
                (cx, cy), (base_now[0], base_now[1]), BASE_STANDOFF + 0.25
            )
            if self._arm_error(arm_now, HOME_ARM) < 0.15:
                self._placed[self._current] = True
                self._advance_object(state)
                if self._current is None:
                    return self._hold_action(state)
            return self._compose(state, tgt, HOME_ARM, GRIPPER_OPEN)

        return self._hold_action(state)

    # -- phase bookkeeping --------------------------------------------------

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_t = 0

    def _force_phase_advance(self, state) -> None:
        """A stalled phase moves on rather than blocking forever."""
        order = [
            PH_DRIVE_TO_CUBE,
            PH_LOWER,
            PH_GRASP,
            PH_LIFT,
            PH_DRIVE_TO_SHELF,
            PH_EXTEND,
            PH_RELEASE,
            PH_RETRACT,
        ]
        if self._phase in order:
            i = order.index(self._phase)
            if i + 1 < len(order):
                self._set_phase(order[i + 1])
            else:
                if self._current is not None:
                    self._placed[self._current] = True
                self._advance_object(state)
        else:
            self._advance_object(state)