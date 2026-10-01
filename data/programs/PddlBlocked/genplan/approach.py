"""Policy for PR2Blocked.

Strategy (near-table route, works for any spare count including zero):

  1. Recover the pen's opening direction as the unit vector from the penned block
     ``green0`` to the red ``blocker``.  The pen is re-posed rigidly each episode,
     so this single vector is all the geometry we need.
  2. Drive the base to a standoff point along that direction, facing inward.
  3. Side-grasp the blocker (horizontal approach), retract it, and drop it on a
     clear patch of the near table well away from the approach corridor.
  4. Side-grasp the penned block through the now-open gap.
  5. Carry it over the plate and open the gripper.

Everything is a servo loop: we run IK (damped least squares on a numeric Jacobian
of our own forward-kinematics model is too fragile here, so we instead use a
simple analytic-ish iterative IK built from PyBullet-free FK on the PR2 left arm
is not available) --- therefore we avoid FK entirely and instead servo in joint
space towards *precomputed* arm configurations, while doing all the interesting
positioning with the mobile base.

Key insight that makes this possible without any kinematics library:
the arm configuration determines the tool pose *in the base frame*.  If we pick
one fixed "reach out horizontally in front of the robot" arm configuration, then
the tool sits at a fixed offset (dx, dy, dz) in the base frame, and we can place
the tool anywhere in the world purely by driving the (holonomic) base.  We learn
that offset online: the observation does not give us the tool pose, but we can
discover it by a one-time calibration that uses the grasp event itself --- and
failing that, we fall back on a hard-coded offset measured from the PR2's
kinematics for the carry/side-grasp configuration family used here.

To stay robust we use a *search*: we hold the arm at a chosen configuration and
sweep the base through a small set of candidate standoff distances, closing the
gripper at each, until ``grasp_active`` turns on.  Grasping is cheap (one step)
and the window is 7cm wide, so a coarse sweep finds it quickly.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MAX_DELTA = 0.2
GRIP_CLOSE = -1.0
GRIP_OPEN = 1.0
GRIP_NONE = 0.0

N_ARM = 7

BLOCK_W = 0.07
BLOCK_H = 0.14
PEN_SPACING = 0.15

# PR2 left-arm joint limits (drake PR2, left arm order:
# shoulder_pan, shoulder_lift, upper_arm_roll, elbow_flex,
# forearm_roll, wrist_flex, wrist_roll).  Continuous joints (indices 4 and 6)
# are given a wide nominal range; the env wraps them anyway.
ARM_LOWER = np.array([-0.564601, -0.3536, -0.65, -2.1213, -np.pi, -2.0, -np.pi])
ARM_UPPER = np.array([2.1353, 1.2963, 3.75, -0.15, np.pi, -0.1, np.pi])

# A "reach straight ahead, gripper horizontal, palm forward" left-arm
# configuration.  Shoulder pan 0 points the arm along +x of the base frame.
# This is the configuration all side grasps are performed from.
REACH_CONF = np.array([0.0, 0.25, 0.0, -0.8, 0.0, -0.9, 0.0])

# A tucked-ish carry configuration used while driving with a block, to keep the
# held block from colliding with the table edge.  Elbow folded, arm up.
CARRY_CONF = np.array([0.6, 0.0, 0.0, -1.6, 0.0, -0.6, 0.0])

# Nominal tool offset in the base frame for REACH_CONF: roughly this far in
# front of the base origin, at this height above the base (z measured from the
# floor, since the base is planar and always at z=0).
# These are refined at runtime by the grasp search, so only rough accuracy is
# needed.
NOMINAL_REACH_X = 0.62
NOMINAL_REACH_Z = 0.80

# Table top height (near table half_extent_z * 2 == 0.73); block centres sit at
# ~0.80.  We servo the torso-less arm, so height is fixed by the arm conf; we
# choose the arm conf's shoulder-lift to put the tool near 0.80.

# Standoff distances (along the opening direction, measured from the target
# block centre to the base origin) that the grasp search sweeps over.
STANDOFF_MIN = 0.50
STANDOFF_MAX = 0.95
STANDOFF_STEP = 0.03

# Tolerances for the base servo.
BASE_POS_TOL = 0.012
BASE_ROT_TOL = 0.03

# Step budget guards so we never spin forever in one phase.
PHASE_BUDGET = 260


def _wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _quat_yaw(qx, qy, qz, qw):
    """Yaw of a quaternion (blocks here only ever rotate about z)."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def _unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0])
    return np.asarray(v, dtype=float) / n


class GeneratedApproach:
    """Move the blocker aside, side-grasp the penned block, put it on the plate."""

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        self._low = np.asarray(action_space.low, dtype=np.float32)
        self._high = np.asarray(action_space.high, dtype=np.float32)

        self.reset_internal()

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def reset_internal(self):
        self.phase = "init"
        self.phase_steps = 0
        self.total_steps = 0

        # Geometry recovered at reset.
        self.open_dir = np.array([-1.0, 0.0])   # unit vector green0 -> blocker
        self.pen_centre = np.zeros(2)
        self.plate_xy = np.zeros(2)
        self.plate_half = 0.3
        self.near_table_xy = np.zeros(2)
        self.near_table_half = np.array([0.3, 0.6])

        # Grasp search state.
        self.search_list = []
        self.search_idx = 0
        self.search_settled = 0
        self.last_grasp_standoff = None

        # Target for the current base servo.
        self.target_xy = None
        self.target_rot = None
        self.target_arm = REACH_CONF.copy()

        # Chosen dump spot for the blocker.
        self.dump_xy = None
        self.dump_rot = None

        # Retry / nudge state.
        self.stall_count = 0
        self.prev_base = None
        self.nudge_steps = 0
        self.nudge_dir = np.array([0.0, 0.0])

        self.done_release_attempts = 0

    def reset(self, state, info):
        self.reset_internal()
        self._read_scene(state)
        self.phase = "approach_blocker"
        self._begin_grasp_search(self._blocker_xy(state))
        return None

    # ------------------------------------------------------------------ #
    # Observation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _obj(state, name):
        try:
            return state.get_object_from_name(name)
        except Exception:
            for o in state:
                if o.name == name:
                    return o
            return None

    def _f(self, state, name, feature):
        o = self._obj(state, name)
        if o is None:
            return None
        return float(state.get(o, feature))

    def _xy(self, state, name):
        o = self._obj(state, name)
        if o is None:
            return None
        return np.array([float(state.get(o, "pose_x")),
                         float(state.get(o, "pose_y"))])

    def _blocker_xy(self, state):
        return self._xy(state, "blocker")

    def _green0_xy(self, state):
        return self._xy(state, "green0")

    def _base(self, state):
        return np.array([self._f(state, "robot", "base_x"),
                         self._f(state, "robot", "base_y"),
                         self._f(state, "robot", "base_rot")])

    def _arm(self, state):
        return np.array([self._f(state, "robot", f"joint_{i}")
                         for i in range(1, N_ARM + 1)])

    def _holding(self, state):
        return self._f(state, "robot", "grasp_active") > 0.5

    def _held_block_name(self, state):
        for o in state:
            if o.type.name != "block":
                continue
            if float(state.get(o, "grasp_active")) > 0.5:
                return o.name
        return None

    def _green_names(self, state):
        names = []
        for o in state:
            if o.type.name == "block" and o.name.startswith("green"):
                names.append(o.name)
        names.sort()
        return names

    def _read_scene(self, state):
        g0 = self._green0_xy(state)
        bl = self._blocker_xy(state)
        if g0 is not None and bl is not None:
            d = bl - g0
            if np.linalg.norm(d) > 1e-6:
                self.open_dir = _unit(d)
            self.pen_centre = g0.copy()

        p = self._xy(state, "plate")
        if p is not None:
            self.plate_xy = p
            hx = self._f(state, "plate", "half_extent_x")
            hy = self._f(state, "plate", "half_extent_y")
            self.plate_half = float(min(hx, hy))

        nt = self._xy(state, "near_table")
        if nt is not None:
            self.near_table_xy = nt
            self.near_table_half = np.array([
                self._f(state, "near_table", "half_extent_x"),
                self._f(state, "near_table", "half_extent_y"),
            ])

    # ------------------------------------------------------------------ #
    # Action construction
    # ------------------------------------------------------------------ #

    def _zero(self):
        return np.zeros(self.action_space.shape, dtype=np.float32)

    def _clip(self, a):
        return np.clip(np.asarray(a, dtype=np.float32),
                       self._low, self._high).astype(np.float32)

    def _servo(self, state, target_xy, target_rot, target_arm, grip=GRIP_NONE,
               rot_first=True):
        """One step of a base+arm servo towards the given targets."""
        a = self._zero()
        base = self._base(state)
        arm = self._arm(state)

        # Rotation error (wrapped).
        rot_err = 0.0
        if target_rot is not None:
            rot_err = float(_wrap(target_rot - base[2]))
            a[2] = np.clip(rot_err, -MAX_DELTA, MAX_DELTA)

        # Translation.  The base joints are world-frame x/y (planar joint), so
        # the delta is simply the world-frame error.
        if target_xy is not None:
            err = np.asarray(target_xy, dtype=float) - base[:2]
            # If we still have a big heading error, translate slowly so the
            # rotation catches up (avoids sweeping the arm through the table).
            scale = 1.0
            if rot_first and abs(rot_err) > 0.35:
                scale = 0.25
            step = np.clip(err, -MAX_DELTA, MAX_DELTA) * scale
            a[0] = step[0]
            a[1] = step[1]

        # Arm.
        if target_arm is not None:
            tgt = np.asarray(target_arm, dtype=float).copy()
            # Clip bounded joints into limits; continuous ones (4, 6) wrap.
            for i in range(N_ARM):
                if i in (4, 6):
                    continue
                tgt[i] = float(np.clip(tgt[i], ARM_LOWER[i], ARM_UPPER[i]))
            derr = tgt - arm
            derr[4] = _wrap(derr[4])
            derr[6] = _wrap(derr[6])
            a[3:10] = np.clip(derr, -MAX_DELTA, MAX_DELTA)

        a[10] = grip
        return self._clip(a)

    def _at_target(self, state, target_xy, target_rot, target_arm,
                   pos_tol=BASE_POS_TOL, rot_tol=BASE_ROT_TOL, arm_tol=0.05):
        base = self._base(state)
        if target_xy is not None:
            if np.linalg.norm(np.asarray(target_xy) - base[:2]) > pos_tol:
                return False
        if target_rot is not None:
            if abs(float(_wrap(target_rot - base[2]))) > rot_tol:
                return False
        if target_arm is not None:
            arm = self._arm(state)
            tgt = np.asarray(target_arm, dtype=float).copy()
            for i in range(N_ARM):
                if i in (4, 6):
                    continue
                tgt[i] = float(np.clip(tgt[i], ARM_LOWER[i], ARM_UPPER[i]))
            d = tgt - arm
            d[4] = _wrap(d[4])
            d[6] = _wrap(d[6])
            if np.max(np.abs(d)) > arm_tol:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Grasp search
    # ------------------------------------------------------------------ #

    def _begin_grasp_search(self, block_xy, approach_dir=None):
        """Set up a sweep of standoff distances for grasping *block_xy*.

        The robot faces along ``-approach_dir`` (i.e. towards the block), with
        the base placed at ``block_xy + approach_dir * d`` for a sweep of d.
        """
        if approach_dir is None:
            approach_dir = self.open_dir
        self.search_dir = _unit(approach_dir)
        self.search_block = np.asarray(block_xy, dtype=float).copy()
        # Heading: base +x must point from base towards the block, i.e. along
        # -approach_dir.
        self.search_rot = float(np.arctan2(-self.search_dir[1],
                                           -self.search_dir[0]))
        ds = np.arange(STANDOFF_MAX, STANDOFF_MIN - 1e-9, -STANDOFF_STEP)
        # Bias the sweep to start near the nominal reach so the common case is
        # found in a couple of probes, then fan outwards.
        nominal = NOMINAL_REACH_X
        order = sorted(ds, key=lambda d: abs(d - nominal))
        self.search_list = [float(d) for d in order]
        self.search_idx = 0
        self.search_settled = 0
        self.phase_steps = 0

    def _search_target(self):
        d = self.search_list[min(self.search_idx, len(self.search_list) - 1)]
        return self.search_block + self.search_dir * d

    # ------------------------------------------------------------------ #
    # Placement helpers
    # ------------------------------------------------------------------ #

    def _pick_dump_spot(self, state):
        """Choose a clear near-table spot for the blocker.

        Requirements: on the near table, not on the plate, at least a block
        width from every other block, and out of the approach corridor we will
        use to reach the penned block.
        """
        tx, ty = self.near_table_xy
        hx, hy = self.near_table_half
        margin = 0.10

        obstacles = []
        for o in state:
            if o.type.name != "block":
                continue
            if float(state.get(o, "grasp_active")) > 0.5:
                continue
            obstacles.append(np.array([float(state.get(o, "pose_x")),
                                       float(state.get(o, "pose_y"))]))
        # Treat the pen walls as an obstacle disc around the pen centre.
        pen = self.pen_centre

        best = None
        best_score = -1e9
        # Candidate grid over the near table surface.
        xs = np.linspace(tx - hx + margin, tx + hx - margin, 9)
        ys = np.linspace(ty - hy + margin, ty + hy - margin, 17)
        for x in xs:
            for y in ys:
                c = np.array([x, y])
                # Avoid the plate entirely (dropping the red block there is
                # harmless for the goal but clutters the target surface).
                if (abs(x - self.plate_xy[0]) < self.plate_half + 0.08 and
                        abs(y - self.plate_xy[1]) < self.plate_half + 0.08):
                    continue
                # Keep clear of the pen assembly.
                dpen = float(np.linalg.norm(c - pen))
                if dpen < 0.34:
                    continue
                # Keep out of the approach corridor: the strip extending from
                # the pen along +open_dir.
                rel = c - pen
                along = float(np.dot(rel, self.open_dir))
                lateral = float(np.dot(rel, np.array([-self.open_dir[1],
                                                      self.open_dir[0]])))
                if along > 0.0 and abs(lateral) < 0.22 and along < 1.1:
                    continue
                # Clearance from other blocks.
                clear = min([float(np.linalg.norm(c - ob)) for ob in obstacles]) \
                    if obstacles else 10.0
                if clear < 0.16:
                    continue
                # Prefer spots close to the pen (short carry) but well clear.
                score = -dpen + 0.4 * min(clear, 0.5)
                if score > best_score:
                    best_score = score
                    best = c
        if best is None:
            # Fall back: just behind the robot's standoff, off to the side.
            perp = np.array([-self.open_dir[1], self.open_dir[0]])
            best = pen + self.open_dir * 0.45 + perp * 0.45
        return best

    def _place_base_for(self, target_xy, approach_dir, standoff):
        """Base pose that puts the tool (roughly) at target_xy."""
        d = _unit(approach_dir)
        xy = np.asarray(target_xy, dtype=float) + d * standoff
        rot = float(np.arctan2(-d[1], -d[0]))
        return xy, rot

    # ------------------------------------------------------------------ #
    # Main policy
    # ------------------------------------------------------------------ #

    def get_action(self, state):
        self.total_steps += 1
        self.phase_steps += 1

        # Refresh geometry that can change (blocker may have moved).
        if self.phase in ("init", "approach_blocker"):
            self._read_scene(state)

        holding = self._holding(state)
        held = self._held_block_name(state)

        # ---------------- Phase: approach + grasp the blocker ------------- #
        if self.phase == "approach_blocker":
            if holding:
                # Got something.  If it's the blocker, go dump it.  If we
                # somehow grabbed a green block, go straight to the plate.
                if held is not None and held.startswith("green"):
                    self.phase = "lift_green"
                    self.phase_steps = 0
                    return self._retract_step(state)
                self.dump_xy = self._pick_dump_spot(state)
                self.phase = "retract_blocker"
                self.phase_steps = 0
                return self._retract_step(state)
            return self._grasp_search_step(state, next_phase_on_fail="fallback")

        # ---------------- Phase: back away holding the blocker ------------ #
        if self.phase == "retract_blocker":
            if not holding:
                # Lost it; retry the grasp.
                self._read_scene(state)
                self._begin_grasp_search(self._blocker_xy(state))
                self.phase = "approach_blocker"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_NONE)
            # Pull straight back along the approach direction, then raise the
            # arm into the carry configuration.
            base = self._base(state)
            back_xy = base[:2] + self.search_dir * 0.30
            if self.phase_steps < 14 and not self._at_target(
                    state, back_xy, None, None, pos_tol=0.03):
                return self._servo(state, back_xy, None, REACH_CONF,
                                   grip=GRIP_NONE, rot_first=False)
            self.phase = "carry_blocker"
            self.phase_steps = 0
            if self.dump_xy is None:
                self.dump_xy = self._pick_dump_spot(state)
            return self._zero_with_grip(GRIP_NONE)

        # ---------------- Phase: carry the blocker to the dump spot ------- #
        if self.phase == "carry_blocker":
            if not holding:
                self._read_scene(state)
                self._begin_grasp_search(self._blocker_xy(state))
                self.phase = "approach_blocker"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_NONE)
            if self.dump_xy is None:
                self.dump_xy = self._pick_dump_spot(state)
            # Approach the dump spot from outside the table centre, using the
            # same reach configuration so the held block sits at the tool.
            dir_to_dump = _unit(self.dump_xy - self.near_table_xy)
            if np.linalg.norm(self.dump_xy - self.near_table_xy) < 1e-3:
                dir_to_dump = self.open_dir
            standoff = self.last_grasp_standoff or NOMINAL_REACH_X
            txy, trot = self._place_base_for(self.dump_xy, dir_to_dump, standoff)
            self.target_xy, self.target_rot = txy, trot

            if self._at_target(state, txy, trot, REACH_CONF,
                               pos_tol=0.03, rot_tol=0.06, arm_tol=0.08):
                self.phase = "release_blocker"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_OPEN)
            if self.phase_steps > PHASE_BUDGET:
                # Just try releasing wherever we are.
                self.phase = "release_blocker"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_OPEN)
            return self._servo(state, txy, trot, REACH_CONF, grip=GRIP_NONE)

        # ---------------- Phase: release the blocker ---------------------- #
        if self.phase == "release_blocker":
            if not holding:
                # Success: head for the penned block.
                self._read_scene(state)
                self.phase = "back_off"
                self.phase_steps = 0
                self.next_after_backoff = "approach_green"
                return self._zero_with_grip(GRIP_NONE)
            # Refused: shuffle sideways and try again.
            self.done_release_attempts += 1
            if self.done_release_attempts > 40:
                # Give up on a tidy dump; wander and keep trying.
                self.done_release_attempts = 0
            perp = np.array([-self.open_dir[1], self.open_dir[0]])
            shift = perp * (0.06 if (self.done_release_attempts % 2) else -0.06)
            base = self._base(state)
            a = self._servo(state, base[:2] + shift, None, REACH_CONF,
                            grip=GRIP_NONE, rot_first=False)
            if self.phase_steps % 3 == 0:
                a[10] = GRIP_OPEN
            return self._clip(a)

        # ---------------- Phase: back off before re-approaching ----------- #
        if self.phase == "back_off":
            base = self._base(state)
            # Move away from the table centre a little, arm in reach conf.
            away = _unit(base[:2] - self.near_table_xy)
            tgt = base[:2] + away * 0.25
            if self.phase_steps > 10:
                self.phase = self.next_after_backoff
                self.phase_steps = 0
                if self.phase == "approach_green":
                    self._read_scene(state)
                    self._begin_grasp_search(self._green0_xy(state),
                                             approach_dir=self.open_dir)
                return self._zero_with_grip(GRIP_NONE)
            return self._servo(state, tgt, None, REACH_CONF,
                               grip=GRIP_NONE, rot_first=False)

        # ---------------- Phase: approach + grasp the penned block -------- #
        if self.phase == "approach_green":
            if holding:
                if held is not None and held.startswith("green"):
                    self.phase = "lift_green"
                    self.phase_steps = 0
                    return self._retract_step(state)
                # Grabbed the blocker again; dump it further away.
                self.dump_xy = self._pick_dump_spot(state)
                self.phase = "retract_blocker"
                self.phase_steps = 0
                return self._retract_step(state)
            return self._grasp_search_step(state, next_phase_on_fail="fallback")

        # ---------------- Phase: lift/retract with the green block -------- #
        if self.phase == "lift_green":
            if not holding:
                self._read_scene(state)
                self._begin_grasp_search(self._nearest_green_xy(state),
                                         approach_dir=self.open_dir)
                self.phase = "approach_green"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_NONE)
            base = self._base(state)
            back_xy = base[:2] + self.search_dir * 0.35
            if self.phase_steps < 16 and not self._at_target(
                    state, back_xy, None, None, pos_tol=0.03):
                return self._servo(state, back_xy, None, REACH_CONF,
                                   grip=GRIP_NONE, rot_first=False)
            self.phase = "to_plate"
            self.phase_steps = 0
            return self._zero_with_grip(GRIP_NONE)

        # ---------------- Phase: carry the green block to the plate ------- #
        if self.phase == "to_plate":
            if not holding:
                self._read_scene(state)
                self._begin_grasp_search(self._nearest_green_xy(state),
                                         approach_dir=self.open_dir)
                self.phase = "approach_green"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_NONE)
            # Aim the tool at a point on the plate, approaching from the side
            # of the table the robot is already on.
            base = self._base(state)
            approach = _unit(base[:2] - self.plate_xy)
            standoff = self.last_grasp_standoff or NOMINAL_REACH_X
            aim = self.plate_xy + approach * min(0.12, self.plate_half * 0.4)
            txy, trot = self._place_base_for(aim, approach, standoff)
            if self._at_target(state, txy, trot, REACH_CONF,
                               pos_tol=0.035, rot_tol=0.07, arm_tol=0.08):
                self.phase = "release_green"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_OPEN)
            if self.phase_steps > PHASE_BUDGET:
                self.phase = "release_green"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_OPEN)
            return self._servo(state, txy, trot, REACH_CONF, grip=GRIP_NONE)

        # ---------------- Phase: release over the plate ------------------- #
        if self.phase == "release_green":
            if not holding:
                # Either terminated already, or the block landed off-plate.
                self._read_scene(state)
                if self._green_on_plate(state):
                    return self._zero_with_grip(GRIP_NONE)
                self._begin_grasp_search(self._nearest_green_xy(state),
                                         approach_dir=self._retry_dir(state))
                self.phase = "approach_green"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_NONE)
            # Refused (something underneath): nudge and retry.
            n = self.phase_steps
            perp = np.array([-self.open_dir[1], self.open_dir[0]])
            shift = perp * (0.05 if (n // 4) % 2 == 0 else -0.05)
            base = self._base(state)
            a = self._servo(state, base[:2] + shift, None, REACH_CONF,
                            grip=GRIP_NONE, rot_first=False)
            if n % 2 == 0:
                a[10] = GRIP_OPEN
            return self._clip(a)

        # ---------------- Fallback: keep probing around the greens -------- #
        if self.phase == "fallback":
            self._read_scene(state)
            if holding:
                held2 = self._held_block_name(state)
                if held2 is not None and held2.startswith("green"):
                    self.phase = "lift_green"
                else:
                    self.dump_xy = self._pick_dump_spot(state)
                    self.phase = "retract_blocker"
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_NONE)
            # Alternate between targeting the blocker and the penned block,
            # each time with a slightly rotated approach direction.
            k = getattr(self, "_fallback_k", 0)
            self._fallback_k = k + 1
            ang = ((k % 7) - 3) * 0.18
            c, s = np.cos(ang), np.sin(ang)
            d = np.array([c * self.open_dir[0] - s * self.open_dir[1],
                          s * self.open_dir[0] + c * self.open_dir[1]])
            tgt = self._blocker_xy(state) if (k % 2 == 0) \
                else self._green0_xy(state)
            if tgt is None:
                tgt = self.pen_centre
            self._begin_grasp_search(tgt, approach_dir=d)
            self.phase = "approach_blocker" if (k % 2 == 0) else "approach_green"
            self.phase_steps = 0
            return self._zero_with_grip(GRIP_NONE)

        # Default: do nothing harmful.
        return self._zero_with_grip(GRIP_NONE)

    # ------------------------------------------------------------------ #
    # Sub-behaviours
    # ------------------------------------------------------------------ #

    def _zero_with_grip(self, grip):
        a = self._zero()
        a[10] = grip
        return self._clip(a)

    def _retract_step(self, state):
        base = self._base(state)
        back = base[:2] + self.search_dir * 0.12
        return self._servo(state, back, None, REACH_CONF,
                           grip=GRIP_NONE, rot_first=False)

    def _nearest_green_xy(self, state):
        base = self._base(state)[:2]
        best, bestd = None, 1e9
        for name in self._green_names(state):
            o = self._obj(state, name)
            if o is None or float(state.get(o, "grasp_active")) > 0.5:
                continue
            xy = np.array([float(state.get(o, "pose_x")),
                           float(state.get(o, "pose_y"))])
            # Strongly prefer the penned block (near table).
            d = float(np.linalg.norm(xy - base))
            if name == "green0":
                d -= 5.0
            if d < bestd:
                bestd, best = d, xy
        if best is None:
            best = self.pen_centre
        return best

    def _retry_dir(self, state):
        g = self._nearest_green_xy(state)
        base = self._base(state)[:2]
        d = base - g
        if np.linalg.norm(d) < 1e-6:
            return self.open_dir
        return _unit(d)

    def _green_on_plate(self, state):
        for name in self._green_names(state):
            o = self._obj(state, name)
            if o is None:
                continue
            x = float(state.get(o, "pose_x"))
            y = float(state.get(o, "pose_y"))
            z = float(state.get(o, "pose_z"))
            if (abs(x - self.plate_xy[0]) <= self.plate_half and
                    abs(y - self.plate_xy[1]) <= self.plate_half and
                    z > 0.72):
                return True
        return False

    def _grasp_search_step(self, state, next_phase_on_fail="fallback"):
        """Drive to the current standoff candidate and try closing.

        The sweep walks a list of standoff distances along the approach
        direction.  At each we settle the base and arm, then issue a close.  If
        ``grasp_active`` does not come on, we move to the next candidate.
        """
        if not self.search_list:
            self._begin_grasp_search(self.search_block, self.search_dir)

        txy = self._search_target()
        trot = self.search_rot

        # Detect a stalled base (collision rejection) and skip ahead.
        base = self._base(state)
        if self.prev_base is not None:
            if np.linalg.norm(base - self.prev_base) < 1e-4:
                self.stall_count += 1
            else:
                self.stall_count = 0
        self.prev_base = base.copy()

        if self.stall_count > 6:
            self.stall_count = 0
            self.search_idx += 1
            self.search_settled = 0
            if self.search_idx >= len(self.search_list):
                self.phase = next_phase_on_fail
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_NONE)
            # Back off a touch before the next probe.
            return self._servo(state, base[:2] + self.search_dir * 0.12, None,
                               REACH_CONF, grip=GRIP_NONE, rot_first=False)

        if not self._at_target(state, txy, trot, REACH_CONF,
                               pos_tol=0.02, rot_tol=0.045, arm_tol=0.06):
            if self.phase_steps > PHASE_BUDGET:
                self.phase = next_phase_on_fail
                self.phase_steps = 0
                return self._zero_with_grip(GRIP_NONE)
            return self._servo(state, txy, trot, REACH_CONF, grip=GRIP_NONE)

        # Settled at this candidate: attempt a close.
        self.search_settled += 1
        if self.search_settled == 1:
            self.last_grasp_standoff = self.search_list[
                min(self.search_idx, len(self.search_list) - 1)]
            return self._zero_with_grip(GRIP_CLOSE)

        # Close did not take (get_action is only reached when not holding).
        self.search_idx += 1
        self.search_settled = 0
        if self.search_idx >= len(self.search_list):
            self.phase = next_phase_on_fail
            self.phase_steps = 0
            return self._zero_with_grip(GRIP_NONE)
        # Re-open so the next close is a fresh grasp attempt.
        return self._zero_with_grip(GRIP_OPEN)