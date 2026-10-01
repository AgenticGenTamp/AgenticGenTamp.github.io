"""A hand-written policy for the variable-count Transport3D environment.

Strategy (count-invariant):

  Phase A (gather): for each cube not already inside the box, drive to the cube,
      grasp it, drive to the box, and release it into the box interior.  The box
      is a legal support surface, so the release succeeds.
  Phase B (transport): grasp the box.  Because the environment rigidly attaches
      every object whose center lies inside a grasped large object, all loaded
      cubes ride along.  Drive to the table and release the box on the table top.
  Phase C (finish): close the gripper and retreat until the end effector is at
      least the goal clearance away from every movable object.

  Fallback: if loading a cube into the box keeps failing, carry that cube
      directly to the table instead.

Everything is done with a simple hand-coded controller:
  * the base is driven in world x/y (the base rotation is left at zero, so base
    frame == world frame up to the initial rotation, which we track anyway);
  * the arm is driven by numerical inverse kinematics on the analytic forward
    kinematics we cannot see -- instead we use a *learned-free* Jacobian
    estimated by finite differences on the observed end-effector position.

Since we cannot query forward kinematics directly (no simulator access), the
end-effector position is estimated from the observed state: when an object is
grasped we know its pose and the grasp transform, which pins the end effector
exactly.  When nothing is grasped we maintain an online estimate of the
end-effector position by tracking how it moved in response to joint commands,
seeded by a fixed nominal offset from the base.  To keep this robust we use a
"probe" phase at the start of each episode that wiggles each joint and measures
the resulting motion of a grasped/近 object... which is not available either.

Therefore the controller below avoids needing end-effector kinematics for
*reaching* by exploiting a much more reliable fact: the environment's grasp
check is a collision test between the movable object and a small box fixed to
the end effector.  We drive the arm to a fixed, precomputed "carry" posture and
move the *base* so that the object comes to the gripper.  The gripper's world
position for that fixed posture is calibrated online, once per episode, by
closing on nothing / observing grasp events, and more directly by the fact that
after a successful grasp the object's pose tells us exactly where the gripper
is.  Before the first grasp we use a nominal offset and refine it by trial:
we sweep the base over a small set of candidate offsets until a grasp succeeds,
then remember the offset that worked and reuse it for every subsequent object.

This "calibrate once by trial, then reuse" scheme is what makes the policy work
without any kinematics model, and it is fully count-invariant: the calibration
cost is paid once, then each additional cube costs one drive + grasp + drive +
release.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

# ---------------------------------------------------------------------------
# Constants taken from the environment configuration (Transport3DEnvConfig).
# These are geometry constants of the family, not per-instance quantities, so
# relying on them does not break count generalization.
# ---------------------------------------------------------------------------

MAX_MAG = 0.2  # config.max_action_mag for Transport3D
GOAL_HEIGHT = 0.3  # config.goal_height_threshold
GOAL_DIST = 0.2  # config.goal_distance_threshold
GRIPPER_OPEN_THRESH = 0.01  # config.gripper_open_threshold
BOX_WALL = 0.01  # config.box_wall_thickness
MIN_PLACE_DIST = 0.01  # config.min_placement_dist

# Home / retract joint configuration (config.initial_joints).
HOME_JOINTS = np.array([0.0, -0.35, -math.pi, -2.5, 0.0, -0.87, math.pi / 2])

# A posture in which the gripper points down and sits out in front of the base
# at a moderate height.  Found by construction: joint_2 and joint_4 dominate the
# reach of the kinova-style arm; joint_6 sets the wrist pitch.
# These are targets, reached by proportional control on the joint deltas.
REACH_LOW = np.array([0.0, 0.55, -math.pi, -2.05, 0.0, -1.10, math.pi / 2])
REACH_MID = np.array([0.0, 0.20, -math.pi, -2.20, 0.0, -1.00, math.pi / 2])
REACH_HIGH = np.array([0.0, -0.10, -math.pi, -2.40, 0.0, -0.90, math.pi / 2])
CARRY = np.array([0.0, -0.20, -math.pi, -2.50, 0.0, -0.85, math.pi / 2])

JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw of a quaternion (rotation about world z)."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class _Obj:
    """A light snapshot of one cuboid object in the state."""

    __slots__ = ("name", "pos", "half", "grasped")

    def __init__(self, name, pos, half, grasped):
        self.name = name
        self.pos = pos
        self.half = half
        self.grasped = grasped


class GeneratedApproach:
    """Hand-coded policy for Transport3D with a variable number of cubes."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        self._lo = low
        self._hi = high
        self._dim = int(low.shape[0])
        # Per-step limit on base/joint deltas.
        self._mag = float(min(MAX_MAG, np.max(np.abs(high[:10]))))

        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # episode reset
    # ------------------------------------------------------------------
    def reset(self, state, info):
        self._t = 0
        self._phase = "init"
        self._target_name = None
        self._sub = 0
        self._sub_t = 0
        self._fail_count = {}
        self._direct_to_table = set()
        # Calibration: the world-frame offset, expressed in the base frame, of
        # the gripper when the arm is in the REACH_* postures.  Seeded with a
        # nominal guess and refined every time we observe a successful grasp.
        self._grip_off = np.array([0.45, 0.0])
        self._grip_off_valid = False
        # Candidate offsets swept while calibrating (dx along base heading).
        self._cal_iter = itertools.cycle(
            [
                np.array([0.45, 0.0]),
                np.array([0.38, 0.0]),
                np.array([0.52, 0.0]),
                np.array([0.45, 0.07]),
                np.array([0.45, -0.07]),
                np.array([0.33, 0.0]),
                np.array([0.58, 0.0]),
            ]
        )
        self._cal_cur = next(self._cal_iter)
        self._grasp_attempts = 0
        self._last_grasped = None
        self._stuck = 0
        self._prev_base = None
        self._nudge = np.zeros(2)
        self._nudge_t = 0

    # ------------------------------------------------------------------
    # state parsing
    # ------------------------------------------------------------------
    def _parse(self, state):
        """Pull out robot + cuboid information in a count-agnostic way."""
        robot = None
        cubes = []
        boxes = []
        table = None

        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            tname = obj.type.name
            if tname == "Kinematic3DRobot":
                robot = obj
                continue
            if tname != "Kinematic3DCuboid":
                continue
            pos = np.array(
                [
                    float(state.get(obj, "pose_x")),
                    float(state.get(obj, "pose_y")),
                    float(state.get(obj, "pose_z")),
                ]
            )
            half = np.array(
                [
                    float(state.get(obj, "half_extent_x")),
                    float(state.get(obj, "half_extent_y")),
                    float(state.get(obj, "half_extent_z")),
                ]
            )
            grasped = float(state.get(obj, "grasp_active")) > 0.5
            rec = _Obj(name, pos, half, grasped)
            if name == "table":
                table = rec
            elif name.startswith("cube"):
                cubes.append(rec)
            elif name.startswith("box"):
                boxes.append(rec)
            else:
                # Unknown movable cuboid: treat like a cube so it still gets
                # transported (keeps us safe against family variations).
                cubes.append(rec)

        base = np.array(
            [
                float(state.get(robot, "pos_base_x")),
                float(state.get(robot, "pos_base_y")),
            ]
        )
        rot = float(state.get(robot, "pos_base_rot"))
        joints = np.array([float(state.get(robot, n)) for n in JOINT_NAMES])
        finger = float(state.get(robot, "finger_state"))
        holding = None
        for rec in cubes + boxes:
            if rec.grasped:
                holding = rec.name
        return {
            "robot": robot,
            "base": base,
            "rot": rot,
            "joints": joints,
            "finger": finger,
            "cubes": cubes,
            "boxes": boxes,
            "table": table,
            "holding": holding,
        }

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _gripper_xy(self, s):
        """World xy of the gripper for the current base pose and calibration."""
        off = self._grip_off if self._grip_off_valid else self._cal_cur
        c, sn = math.cos(s["rot"]), math.sin(s["rot"])
        return s["base"] + np.array(
            [c * off[0] - sn * off[1], sn * off[0] + c * off[1]]
        )

    def _base_for_gripper(self, s, target_xy):
        """Base xy that would put the gripper at target_xy, given calibration."""
        off = self._grip_off if self._grip_off_valid else self._cal_cur
        c, sn = math.cos(s["rot"]), math.sin(s["rot"])
        rot_off = np.array([c * off[0] - sn * off[1], sn * off[0] + c * off[1]])
        return np.asarray(target_xy, dtype=np.float64) - rot_off

    @staticmethod
    def _inside_box(cube, box):
        """Replicates the env's containment test (center within box extents)."""
        d = np.abs(cube.pos - box.pos)
        return bool(np.all(d < box.half))

    @staticmethod
    def _on_table(rec):
        return rec.pos[2] > GOAL_HEIGHT

    def _box(self, s):
        """The primary container box, or None."""
        if not s["boxes"]:
            return None
        # Prefer a box big enough to act as a container (env requires all
        # half-extents > 0.04 for containment to trigger).
        containers = [b for b in s["boxes"] if np.all(b.half > 0.04)]
        if containers:
            return max(containers, key=lambda b: float(np.prod(b.half)))
        return s["boxes"][0]

    # ------------------------------------------------------------------
    # low-level action construction
    # ------------------------------------------------------------------
    def _act(self, dbase=(0.0, 0.0), drot=0.0, djoints=None, grip=0.0):
        a = np.zeros(self._dim, dtype=np.float64)
        a[0] = dbase[0]
        a[1] = dbase[1]
        a[2] = drot
        if djoints is not None:
            n = min(7, self._dim - 4)
            a[3 : 3 + n] = djoints[:n]
        a[self._dim - 1] = grip
        return np.clip(a, self._lo, self._hi).astype(np.float32)

    def _drive(self, s, target_xy, gain=1.0):
        """Delta base command (in base frame) toward a world-frame target."""
        err = np.asarray(target_xy, dtype=np.float64) - s["base"]
        c, sn = math.cos(s["rot"]), math.sin(s["rot"])
        # world -> base frame
        local = np.array([c * err[0] + sn * err[1], -sn * err[0] + c * err[1]])
        local = local + self._nudge
        n = float(np.linalg.norm(local))
        if n < 1e-9:
            return np.zeros(2)
        step = min(self._mag, n * gain)
        return local / n * step

    def _joint_step(self, s, target_joints, gain=1.0):
        err = np.asarray(target_joints, dtype=np.float64) - s["joints"]
        step = np.clip(err * gain, -self._mag, self._mag)
        return step

    @staticmethod
    def _joints_close(s, target, tol=0.12):
        return bool(np.max(np.abs(s["joints"] - np.asarray(target))) < tol)

    # ------------------------------------------------------------------
    # stuck detection: collision-rejected steps leave the base unmoved
    # ------------------------------------------------------------------
    def _update_stuck(self, s, wanted_motion):
        if self._prev_base is None:
            self._prev_base = s["base"].copy()
            return
        moved = float(np.linalg.norm(s["base"] - self._prev_base))
        if wanted_motion and moved < 1e-4:
            self._stuck += 1
        else:
            self._stuck = max(0, self._stuck - 1)
        self._prev_base = s["base"].copy()
        if self._stuck > 6 and self._nudge_t <= 0:
            # Sidestep: push perpendicular for a while.
            ang = self._rng.uniform(-math.pi, math.pi)
            self._nudge = np.array([math.cos(ang), math.sin(ang)]) * 0.8
            self._nudge_t = 12
            self._stuck = 0
        if self._nudge_t > 0:
            self._nudge_t -= 1
            if self._nudge_t == 0:
                self._nudge = np.zeros(2)

    # ------------------------------------------------------------------
    # calibration bookkeeping
    # ------------------------------------------------------------------
    def _note_grasp(self, s):
        """A grasp just succeeded: recover the true gripper offset from it."""
        name = s["holding"]
        if name is None:
            return
        rec = None
        for r in s["cubes"] + s["boxes"]:
            if r.name == name:
                rec = r
        if rec is None:
            return
        err = rec.pos[:2] - s["base"]
        c, sn = math.cos(s["rot"]), math.sin(s["rot"])
        local = np.array([c * err[0] + sn * err[1], -sn * err[0] + c * err[1]])
        if 0.15 < float(np.linalg.norm(local)) < 1.2:
            self._grip_off = local
            self._grip_off_valid = True

    def _note_grasp_failure(self):
        self._grasp_attempts += 1
        if self._grasp_attempts % 3 == 0 and not self._grip_off_valid:
            self._cal_cur = next(self._cal_iter)

    # ------------------------------------------------------------------
    # target selection
    # ------------------------------------------------------------------
    def _pick_next_cube(self, s, box):
        """Next cube needing work, or None if all cubes are handled."""
        best = None
        best_d = None
        for cube in s["cubes"]:
            if self._on_table(cube):
                continue
            if box is not None and self._inside_box(cube, box):
                continue
            d = float(np.linalg.norm(cube.pos[:2] - s["base"]))
            if best_d is None or d < best_d:
                best, best_d = cube, d
        return best

    def _all_high(self, s):
        for rec in s["cubes"] + s["boxes"]:
            if rec.pos[2] <= GOAL_HEIGHT:
                return False
        return True

    def _clearance_ok(self, s):
        """Approximate condition 3 of the goal using the calibrated gripper."""
        g = self._gripper_xy(s)
        for rec in s["cubes"] + s["boxes"]:
            if float(np.linalg.norm(rec.pos[:2] - g)) < GOAL_DIST + 0.15:
                return False
        return True

    # ------------------------------------------------------------------
    # main policy
    # ------------------------------------------------------------------
    def get_action(self, state):
        self._t += 1
        s = self._parse(state)
        box = self._box(s)
        holding = s["holding"]

        if holding is not None and holding != self._last_grasped:
            self._note_grasp(s)
        self._last_grasped = holding

        # ---------------- Phase C: everything is up; finish -------------
        if self._all_high(s) and holding is None:
            return self._finish(s)

        # ---------------- carrying something ---------------------------
        if holding is not None:
            if box is not None and holding == box.name:
                return self._deliver_box(s, box)
            return self._deliver_cube(s, box, holding)

        # ---------------- nothing held: choose a job -------------------
        cube = self._pick_next_cube(s, box)
        if cube is not None:
            return self._go_grasp(s, cube)

        # All cubes are stowed or already up.  Now the box itself.
        if box is not None and not self._on_table(box):
            return self._go_grasp(s, box)

        # Nothing left to move but not all high (e.g. a stray object): retreat.
        return self._finish(s)

    # ------------------------------------------------------------------
    # behaviours
    # ------------------------------------------------------------------
    def _go_grasp(self, s, target):
        """Drive so the gripper is over `target`, lower, and close."""
        if self._target_name != target.name:
            self._target_name = target.name
            self._sub = 0
            self._sub_t = 0
        self._sub_t += 1

        goal_base = self._base_for_gripper(s, target.pos[:2])
        err = float(np.linalg.norm(goal_base - s["base"]))

        # Approach with the arm parked high so we do not sweep objects over.
        if self._sub == 0:
            self._update_stuck(s, True)
            db = self._drive(s, goal_base)
            dj = self._joint_step(s, REACH_HIGH, gain=0.9)
            if err < 0.05 and self._joints_close(s, REACH_HIGH, tol=0.25):
                self._sub = 1
                self._sub_t = 0
            if self._sub_t > 220:
                # Give up approaching; try a different calibration offset.
                self._note_grasp_failure()
                self._sub_t = 0
            return self._act(dbase=db, djoints=dj, grip=1.0)

        # Fine-align and lower onto the object.
        if self._sub == 1:
            self._update_stuck(s, err > 0.02)
            db = self._drive(s, goal_base, gain=0.6)
            # Choose a lowering posture based on how tall the object is.
            tgt = REACH_LOW if target.half[2] < 0.06 else REACH_MID
            dj = self._joint_step(s, tgt, gain=0.7)
            if self._joints_close(s, tgt, tol=0.18) and err < 0.06:
                self._sub = 2
                self._sub_t = 0
            if self._sub_t > 160:
                self._sub = 2
                self._sub_t = 0
            return self._act(dbase=db, djoints=dj, grip=1.0)

        # Close.
        if self._sub == 2:
            self._sub = 3
            self._sub_t = 0
            return self._act(grip=-1.0)

        # Verify: if we are not holding anything, the grasp missed.
        self._note_grasp_failure()
        cnt = self._fail_count.get(target.name, 0) + 1
        self._fail_count[target.name] = cnt
        if cnt >= 6:
            # This object is awkward (e.g. neighbours in the grasp zone);
            # remember to take it straight to the table instead of the box.
            self._direct_to_table.add(target.name)
        self._sub = 0
        self._sub_t = 0
        # Back off slightly and retry from the approach posture.
        return self._act(djoints=self._joint_step(s, REACH_HIGH, 0.8), grip=1.0)

    def _deliver_cube(self, s, box, name):
        """Carrying a cube: put it in the box, or on the table as a fallback."""
        use_box = (
            box is not None
            and not self._on_table(box)
            and name not in self._direct_to_table
        )
        if use_box:
            # Aim at a point inside the box interior.
            inner = np.maximum(box.half[:2] - BOX_WALL - 0.03, 0.0)
            jitter = self._rng.uniform(-1.0, 1.0, size=2) * inner * 0.5
            target_xy = box.pos[:2] + jitter
            drop_target = box
        else:
            table = s["table"]
            if table is None:
                return self._act(grip=1.0)
            inner = np.maximum(table.half[:2] - 0.06, 0.0)
            jitter = self._rng.uniform(-1.0, 1.0, size=2) * inner * 0.6
            target_xy = table.pos[:2] + jitter
            drop_target = table

        goal_base = self._base_for_gripper(s, target_xy)
        err = float(np.linalg.norm(goal_base - s["base"]))
        self._update_stuck(s, err > 0.02)

        if err > 0.06:
            db = self._drive(s, goal_base)
            dj = self._joint_step(s, CARRY, gain=0.8)
            return self._act(dbase=db, djoints=dj, grip=0.0)

        # Over the drop point: lower until the release is accepted.  The env
        # only releases when the held object is within min_placement_dist of a
        # support surface, so an early open command is simply ignored -- we can
        # spam it while descending.
        top_z = drop_target.pos[2] + drop_target.half[2]
        if not use_box:
            tgt = REACH_MID if top_z > 0.35 else REACH_LOW
        else:
            tgt = REACH_MID
        dj = self._joint_step(s, tgt, gain=0.55)
        db = self._drive(s, goal_base, gain=0.5)
        return self._act(dbase=db, djoints=dj, grip=1.0)

    def _deliver_box(self, s, box):
        """Carrying the loaded box: place it on the table."""
        table = s["table"]
        if table is None:
            return self._act(grip=1.0)
        target_xy = table.pos[:2]
        goal_base = self._base_for_gripper(s, target_xy)
        err = float(np.linalg.norm(goal_base - s["base"]))
        self._update_stuck(s, err > 0.02)

        if err > 0.07:
            db = self._drive(s, goal_base)
            dj = self._joint_step(s, CARRY, gain=0.8)
            return self._act(dbase=db, djoints=dj, grip=0.0)

        # Descend toward the table top while requesting release.
        dj = self._joint_step(s, REACH_MID, gain=0.5)
        db = self._drive(s, goal_base, gain=0.5)
        return self._act(dbase=db, djoints=dj, grip=1.0)

    def _finish(self, s):
        """Close the gripper and retreat until the clearance condition holds."""
        # Retract the arm to the home/retract posture first: this pulls the end
        # effector back toward the base and away from the table.
        dj = self._joint_step(s, HOME_JOINTS, gain=0.9)

        # Retreat direction: away from the centroid of all objects.
        pts = [r.pos[:2] for r in s["cubes"] + s["boxes"]]
        if pts:
            centroid = np.mean(np.stack(pts, axis=0), axis=0)
        else:
            centroid = s["base"]
        away = s["base"] - centroid
        n = float(np.linalg.norm(away))
        if n < 1e-6:
            away = np.array([-1.0, 0.0])
            n = 1.0
        away = away / n
        # Target a point comfortably outside the clearance radius.
        goal = centroid + away * (GOAL_DIST + 1.0)
        db = self._drive(s, goal)

        # Keep the gripper closed (goal requires finger_state <= threshold).
        grip = -1.0 if s["finger"] > GRIPPER_OPEN_THRESH * 0.5 else 0.0
        return self._act(dbase=db, djoints=dj, grip=grip)