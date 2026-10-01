# ============================================================================
# GeneratedApproach for kinder/BalanceBeam3D-o3-v0
#
# IMPORTANT CAVEAT (please read before using this as a "solution"):
#
# The interface I am given here is:
#
#     reset(state, info) -> None
#     get_action(state)  -> action in action_space
#
# I am handed observations and must return ACTIONS. I am NOT handed the env
# object, so I cannot call env.set_state(...). The "teleport the state"
# strategy discussed earlier is therefore not expressible through this
# interface -- it requires mutating the simulator's ground truth, and all I
# can emit here is an 11-dim action vector that goes through MuJoCo physics.
#
# I deliberately do not attempt to reach around this interface (e.g. by
# walking the gc object graph or sys._getframe to find the live env and
# calling set_state on it). That would be a covert modification of the
# evaluation rather than a policy, and it would produce a score that does
# not correspond to the task being measured. If a state-teleport baseline
# is actually wanted, it should be run through a harness that passes the
# env in explicitly, so that what happened is legible in the results.
#
# So this module is an honest attempt at the real task: a scripted
# navigate -> grasp -> transport -> place controller, run three times, with
# along-beam offsets chosen so the net torque on the hinge cancels.
#
# Expected performance: modest. Scripted IK-free joint-space manipulation
# against contact dynamics is genuinely hard, and I expect partial credit
# (blocks near but often not on the beam) more often than full success.
# That is a real number about a real attempt, which is more useful than a
# fabricated 3.0.
#
# Design notes:
#   - Action space is 11-dim: base (x, y, yaw), 7 arm joints, 1 gripper.
#     The TidyBot config uses act_delta=True by default, but robocode may
#     configure otherwise, so at reset() we probe the action_space bounds
#     to guess whether actions are absolute targets or deltas, and we drive
#     the base/arm with a proportional controller toward setpoints either
#     way (a P-controller on error works for both conventions: for absolute
#     targets we clip the target itself; for deltas we emit the clipped
#     error).
#   - Arm control is joint-space with a small analytic reach heuristic
#     rather than full IK, since no IK primitive is provided (primitives is
#     an empty dict). We use a handful of tuned joint configurations for
#     "home", "pre-grasp (low reach)", "grasp", "lift", and "place", and
#     rely on positioning the BASE so the target lands under the gripper.
#   - Seesaw geometry is read from the observation, not hardcoded, except
#     for the generated-seesaw defaults that are not exposed in the obs
#     (pivot_height, beam_clearance, beam_thickness). Those come from
#     GeneratedSeesaw's documented defaults; bb_z gives us a cross-check.
# ============================================================================

import numpy as np


# ---------------------------------------------------------------------------
# Observation layout (from the environment's documented table).
# ---------------------------------------------------------------------------

IDX = {
    "large_block": 0,
    "robot": 16,
    "seesaw_1": 38,
    "small_block_1": 54,
    "small_block_2": 70,
}

# Movable-object feature offsets within its 16-wide block.
O_X, O_Y, O_Z = 0, 1, 2
O_QW, O_QX, O_QY, O_QZ = 3, 4, 5, 6
O_VX, O_VY, O_VZ = 7, 8, 9
O_WX, O_WY, O_WZ = 10, 11, 12
O_BBX, O_BBY, O_BBZ = 13, 14, 15

# Robot feature offsets within its 22-wide block (starting at index 16).
R_BASE_X, R_BASE_Y, R_BASE_ROT = 0, 1, 2
R_ARM_J1 = 3           # joints 1..7 at offsets 3..9
R_GRIPPER = 10
R_VEL_BASE_X, R_VEL_BASE_Y, R_VEL_BASE_ROT = 11, 12, 13
R_VEL_ARM_J1 = 14      # joint vels 1..7 at offsets 14..20
R_VEL_GRIPPER = 21

# Action layout: [base_x, base_y, base_yaw, j1..j7, gripper]
A_BASE_X, A_BASE_Y, A_BASE_YAW = 0, 1, 2
A_ARM_J1 = 3
A_GRIPPER = 10
ACTION_DIM = 11

# GeneratedSeesaw defaults that the observation does not expose.
SEESAW_PIVOT_HEIGHT = 0.04
SEESAW_BEAM_CLEARANCE = 0.002
SEESAW_BEAM_THICKNESS = 0.01
SEESAW_BEAM_LENGTH_DEFAULT = 0.40


# ---------------------------------------------------------------------------
# Small math helpers.
# ---------------------------------------------------------------------------

def _wrap_angle(a):
    """Wrap an angle to [-pi, pi]."""
    return (float(a) + np.pi) % (2.0 * np.pi) - np.pi


def _quat_to_yaw(qw, qx, qy, qz):
    """Yaw (rotation about world z) from a MuJoCo (w, x, y, z) quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _obj(state, name):
    """Return the 16-wide feature block for a movable object."""
    i = IDX[name]
    return np.asarray(state, dtype=np.float64)[i:i + 16]


def _robot(state):
    """Return the 22-wide robot feature block."""
    i = IDX["robot"]
    return np.asarray(state, dtype=np.float64)[i:i + 22]


def _xy(v):
    return np.array([v[O_X], v[O_Y]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Scripted arm configurations.
#
# These are joint-space waypoints for the Kinova Gen3 style 7-DoF arm. The
# initial state observed at reset is approximately:
#   j = [0, -0.349, 3.1416, -2.548, 0, -0.873, 1.5708]
# which is the stowed/home pose. We derive reach poses from it by opening
# the shoulder/elbow so the wrist descends in front of the base.
#
# Without IK these are necessarily approximate; the base is what does the
# fine XY positioning.
# ---------------------------------------------------------------------------

ARM_HOME = np.array([0.0, -0.349, 3.1416, -2.548, 0.0, -0.873, 1.5708])

# Reach forward and down toward the floor in front of the base.
ARM_REACH_FLOOR = np.array([0.0, 0.52, 3.1416, -1.55, 0.0, -1.05, 1.5708])

# Slightly higher: hovering above a floor object before descent.
ARM_HOVER_FLOOR = np.array([0.0, 0.30, 3.1416, -1.85, 0.0, -0.95, 1.5708])

# Lifted carry pose: object held clear of the ground.
ARM_CARRY = np.array([0.0, -0.15, 3.1416, -2.15, 0.0, -0.95, 1.5708])

# Place pose: beam surface sits ~5-6 cm above ground, so we descend less
# far than for a floor grasp.
ARM_HOVER_BEAM = np.array([0.0, 0.18, 3.1416, -1.95, 0.0, -0.95, 1.5708])
ARM_PLACE_BEAM = np.array([0.0, 0.40, 3.1416, -1.70, 0.0, -1.00, 1.5708])

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0

# How far in front of the base (in base-frame +x) the gripper lands when
# the arm is in a reach pose. Tuned by eye against the TidyBot geometry.
REACH_OFFSET = 0.52


class _Waypoint:
    """One step of the scripted plan."""

    __slots__ = ("kind", "base_xy", "base_yaw", "arm", "gripper", "settle")

    def __init__(self, kind, base_xy=None, base_yaw=None, arm=None,
                 gripper=None, settle=8):
        self.kind = kind
        self.base_xy = None if base_xy is None else np.asarray(
            base_xy, dtype=np.float64)
        self.base_yaw = base_yaw
        self.arm = None if arm is None else np.asarray(arm, dtype=np.float64)
        self.gripper = gripper
        self.settle = int(settle)


class GeneratedApproach:
    """Scripted pick-and-place policy for BalanceBeam3D-o3.

    Plan: for each of the three cubes, drive the base to a standoff pose
    facing the cube, reach down, close the gripper, lift, drive to a
    standoff pose facing the seesaw beam at the chosen along-beam offset,
    lower, open, and retract. Offsets are symmetric so the hinge torques
    cancel.
    """

    # Along-beam local-x offsets (metres) for each cube. The large block
    # goes over the pivot (zero moment arm); the two identical small blocks
    # are placed symmetrically so their torques cancel exactly.
    PLACE_OFFSETS = {
        "large_block": 0.00,
        "small_block_1": +0.095,
        "small_block_2": -0.095,
    }

    # Order matters: put the heavy block on first (over the pivot, so it
    # cannot tip anything), then the symmetric pair.
    PLACE_ORDER = ["large_block", "small_block_1", "small_block_2"]

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives if primitives is not None else {}

        self._low = np.asarray(
            getattr(action_space, "low", -np.ones(ACTION_DIM)),
            dtype=np.float64).reshape(-1)
        self._high = np.asarray(
            getattr(action_space, "high", np.ones(ACTION_DIM)),
            dtype=np.float64).reshape(-1)
        if self._low.shape[0] != ACTION_DIM:
            self._low = np.full(ACTION_DIM, -1.0)
            self._high = np.full(ACTION_DIM, 1.0)

        # Heuristic: if the base-x bound is small (order 0.1 m) the action is
        # a per-step delta; if it is large (order 1 m or more) it is more
        # likely an absolute target. Either way the P-controller below works,
        # but the gain and clipping differ.
        span = float(self._high[A_BASE_X] - self._low[A_BASE_X])
        self._delta_mode = span < 1.0

        self._plan = []
        self._pc = 0          # program counter into self._plan
        self._hold = 0        # remaining settle ticks on current waypoint
        self._steps = 0
        self._max_steps = 100000

    # -- geometry -----------------------------------------------------------

    def _beam_frame(self, state):
        """Return (origin_xyz, yaw, half_length, surface_z) for the seesaw."""
        s = _obj(state, "seesaw_1")
        origin = np.array([s[O_X], s[O_Y], s[O_Z]], dtype=np.float64)
        yaw = _quat_to_yaw(s[O_QW], s[O_QX], s[O_QY], s[O_QZ])

        # bb_x is the full beam length for GeneratedSeesaw.
        beam_len = float(s[O_BBX]) if s[O_BBX] > 1e-6 else \
            SEESAW_BEAM_LENGTH_DEFAULT
        half_len = 0.5 * beam_len

        # Top surface of the beam, in world z.
        surface_z = (origin[2] + SEESAW_PIVOT_HEIGHT + SEESAW_BEAM_CLEARANCE
                     + SEESAW_BEAM_THICKNESS)
        return origin, yaw, half_len, surface_z

    def _place_target(self, state, name):
        """World-frame resting position for `name` on the beam."""
        origin, yaw, half_len, surface_z = self._beam_frame(state)

        off = self.PLACE_OFFSETS[name]
        # Keep well inside the beam so contact is unambiguous.
        off = float(np.clip(off, -0.75 * half_len, 0.75 * half_len))

        c, s_ = np.cos(yaw), np.sin(yaw)
        wx = origin[0] + c * off
        wy = origin[1] + s_ * off

        half_cube = 0.5 * float(_obj(state, name)[O_BBZ])
        wz = surface_z + half_cube
        return np.array([wx, wy, wz], dtype=np.float64), yaw

    @staticmethod
    def _standoff(target_xy, from_xy):
        """Base pose that puts the gripper over `target_xy`.

        The base parks REACH_OFFSET away along the line from `from_xy` to
        the target, facing the target.
        """
        d = np.asarray(target_xy, dtype=np.float64) - np.asarray(
            from_xy, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            d = np.array([1.0, 0.0])
            n = 1.0
        u = d / n
        base_xy = np.asarray(target_xy, dtype=np.float64) - REACH_OFFSET * u
        yaw = float(np.arctan2(u[1], u[0]))
        return base_xy, yaw

    # -- planning -----------------------------------------------------------

    def _build_plan(self, state):
        plan = []
        rb = _robot(state)
        cursor = np.array([rb[R_BASE_X], rb[R_BASE_Y]], dtype=np.float64)

        for name in self.PLACE_ORDER:
            cube = _obj(state, name)
            cube_xy = _xy(cube)

            # --- approach and grasp ---
            gb_xy, gb_yaw = self._standoff(cube_xy, cursor)
            plan.append(_Waypoint("drive", base_xy=gb_xy, base_yaw=gb_yaw,
                                  arm=ARM_HOME, gripper=GRIPPER_OPEN,
                                  settle=10))
            plan.append(_Waypoint("arm", base_xy=gb_xy, base_yaw=gb_yaw,
                                  arm=ARM_HOVER_FLOOR, gripper=GRIPPER_OPEN,
                                  settle=8))
            plan.append(_Waypoint("arm", base_xy=gb_xy, base_yaw=gb_yaw,
                                  arm=ARM_REACH_FLOOR, gripper=GRIPPER_OPEN,
                                  settle=10))
            plan.append(_Waypoint("grip", base_xy=gb_xy, base_yaw=gb_yaw,
                                  arm=ARM_REACH_FLOOR, gripper=GRIPPER_CLOSED,
                                  settle=12))
            plan.append(_Waypoint("arm", base_xy=gb_xy, base_yaw=gb_yaw,
                                  arm=ARM_CARRY, gripper=GRIPPER_CLOSED,
                                  settle=10))

            # --- transport and place ---
            tgt_xyz, _ = self._place_target(state, name)
            tgt_xy = tgt_xyz[:2]
            pb_xy, pb_yaw = self._standoff(tgt_xy, gb_xy)
            plan.append(_Waypoint("drive", base_xy=pb_xy, base_yaw=pb_yaw,
                                  arm=ARM_CARRY, gripper=GRIPPER_CLOSED,
                                  settle=12))
            plan.append(_Waypoint("arm", base_xy=pb_xy, base_yaw=pb_yaw,
                                  arm=ARM_HOVER_BEAM, gripper=GRIPPER_CLOSED,
                                  settle=8))
            plan.append(_Waypoint("arm", base_xy=pb_xy, base_yaw=pb_yaw,
                                  arm=ARM_PLACE_BEAM, gripper=GRIPPER_CLOSED,
                                  settle=10))
            plan.append(_Waypoint("grip", base_xy=pb_xy, base_yaw=pb_yaw,
                                  arm=ARM_PLACE_BEAM, gripper=GRIPPER_OPEN,
                                  settle=10))
            # Retract gently so we do not drag the cube off the beam.
            plan.append(_Waypoint("arm", base_xy=pb_xy, base_yaw=pb_yaw,
                                  arm=ARM_HOVER_BEAM, gripper=GRIPPER_OPEN,
                                  settle=8))
            plan.append(_Waypoint("arm", base_xy=pb_xy, base_yaw=pb_yaw,
                                  arm=ARM_HOME, gripper=GRIPPER_OPEN,
                                  settle=8))

            cursor = pb_xy

        # Park clear of the seesaw and hold still, so the beam can settle
        # and the goal checker sees a static, balanced scene.
        origin, _, _, _ = self._beam_frame(state)
        park = origin[:2] + np.array([-0.9, -0.9])
        plan.append(_Waypoint("drive", base_xy=park, base_yaw=0.0,
                              arm=ARM_HOME, gripper=GRIPPER_OPEN, settle=40))
        return plan

    # -- control ------------------------------------------------------------

    def _emit(self, state, wp):
        """Proportional controller toward a waypoint's setpoints."""
        rb = _robot(state)
        a = np.zeros(ACTION_DIM, dtype=np.float64)

        # Base.
        if wp.base_xy is not None:
            ex = wp.base_xy[0] - rb[R_BASE_X]
            ey = wp.base_xy[1] - rb[R_BASE_Y]
        else:
            ex = ey = 0.0
        if wp.base_yaw is not None:
            eyaw = _wrap_angle(wp.base_yaw - rb[R_BASE_ROT])
        else:
            eyaw = 0.0

        # Arm.
        if wp.arm is not None:
            cur = rb[R_ARM_J1:R_ARM_J1 + 7]
            earm = np.array([_wrap_angle(t - c)
                             for t, c in zip(wp.arm, cur)], dtype=np.float64)
        else:
            earm = np.zeros(7, dtype=np.float64)

        if self._delta_mode:
            kb, ka = 0.9, 0.9
            a[A_BASE_X] = kb * ex
            a[A_BASE_Y] = kb * ey
            a[A_BASE_YAW] = kb * eyaw
            a[A_ARM_J1:A_ARM_J1 + 7] = ka * earm
        else:
            # Absolute targets: command the setpoint directly.
            a[A_BASE_X] = wp.base_xy[0] if wp.base_xy is not None \
                else rb[R_BASE_X]
            a[A_BASE_Y] = wp.base_xy[1] if wp.base_xy is not None \
                else rb[R_BASE_Y]
            a[A_BASE_YAW] = wp.base_yaw if wp.base_yaw is not None \
                else rb[R_BASE_ROT]
            a[A_ARM_J1:A_ARM_J1 + 7] = wp.arm if wp.arm is not None else \
                rb[R_ARM_J1:R_ARM_J1 + 7]

        a[A_GRIPPER] = wp.gripper if wp.gripper is not None \
            else rb[R_GRIPPER]

        return np.clip(a, self._low, self._high).astype(np.float32)

    def _reached(self, state, wp):
        """Have we converged to this waypoint's setpoints?"""
        rb = _robot(state)
        ok = True
        if wp.base_xy is not None:
            d = np.hypot(wp.base_xy[0] - rb[R_BASE_X],
                         wp.base_xy[1] - rb[R_BASE_Y])
            ok = ok and d < 0.06
        if wp.base_yaw is not None:
            ok = ok and abs(_wrap_angle(wp.base_yaw - rb[R_BASE_ROT])) < 0.12
        if wp.arm is not None:
            cur = rb[R_ARM_J1:R_ARM_J1 + 7]
            err = max(abs(_wrap_angle(t - c)) for t, c in zip(wp.arm, cur))
            ok = ok and err < 0.10
        return bool(ok)

    # -- API ----------------------------------------------------------------

    def reset(self, state, info):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        self._steps = 0
        self._pc = 0
        self._hold = 0
        try:
            self._plan = self._build_plan(state)
        except Exception:
            # If anything about the observation layout surprises us, fall
            # back to doing nothing rather than flailing the arm around.
            self._plan = []
        if self._plan:
            self._hold = self._plan[0].settle
        return None

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        self._steps += 1

        if not self._plan or self._pc >= len(self._plan):
            # Plan exhausted: hold still. A zero delta (or a command equal to
            # the current pose in absolute mode) keeps the scene static so the
            # beam can settle.
            return self._emit(state, _Waypoint(
                "hold", base_xy=None, base_yaw=None, arm=None, gripper=None))

        wp = self._plan[self._pc]
        action = self._emit(state, wp)

        # Advance when converged, or when the settle budget for this
        # waypoint runs out (prevents deadlock on an unreachable setpoint).
        if self._hold > 0:
            self._hold -= 1
        if self._reached(state, wp) or self._hold <= 0:
            self._pc += 1
            if self._pc < len(self._plan):
                self._hold = self._plan[self._pc].settle

        return action