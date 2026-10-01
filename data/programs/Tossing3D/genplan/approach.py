"""
GeneratedApproach for Tossing3DEnv (variable object count).

IMPORTANT CAVEAT (please read before trusting this):

I was asked to "implement the strategy" for solving all instances of this
environment. In my previous turn I said honestly that I do not know of a
simple strategy that solves it, and that I could not derive one from the
material provided. That is still true. The two artifacts that would decide
the question are NOT in the supplied source:

  1. tasks/Tossing3D/Tossing3D-o{1,2}.json  -- the goal region geometry
     (which region, which target, what `ranges`). `_check_goals` treats
     "in" as "on" and only tests whether the object's CENTER point lies in
     an axis-aligned region box, with a placement tolerance added on every
     side. If that region were large or generously padded, the task could
     be near-degenerate. I cannot see it, so I cannot claim that.

  2. kinder/envs/dynamic3d/robots/ -- TidyBot3DRobotActionSpace. I do not
     know the channel order, the per-channel bounds, or (critically)
     whether `act_delta=True` makes the base-pose channels relative or
     absolute. Tossing3DEnv forces use_arm_velocities=True, so the action
     is the 18-D variant: base(3), arm joints(7), gripper(1), arm joint
     velocity targets(7). The ORDER of those blocks is an assumption.

So rather than fabricate a confident "trick" and dress it up as the
intended solution, this module implements the most defensible thing I can
actually justify from the information I do have, and it does so in a way
that is explicitly count-generic and never crashes:

  STRATEGY: drive the holonomic base toward the bin while holding a neutral
  arm posture, and command the base to a standoff pose adjacent to the bin
  on the near side of the barrier. Cycle the target among the remaining
  unplaced cubes so that, if base motion is absolute (teleport-like) or
  strongly tracking, the robot ends up co-located with cubes and with the
  bin. Keep the gripper closed-ish and issue zero arm-velocity targets so
  the arm does not thrash.

This is a *base-navigation* policy, not a throwing policy. I expect it to
score poorly on the real task. I am not going to pretend otherwise. If you
can supply the task JSON and the action-space definition, the correct
policy is straightforward to write and I will rewrite this properly.

Everything below is defensive: unknown feature -> default, missing object
-> skipped, any exception in the policy -> a zero action. It will run on
any object count, including zero cubes.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Small safe accessors. The schema is known from the env card, but I do not
# want a KeyError in a rollout to be the thing that "fails" the episode, so
# every read is guarded.
# ----------------------------------------------------------------------------

def _feat(state, obj, name, default=0.0):
    """state.get(obj, name) with a default if the feature is absent."""
    try:
        return float(state.get(obj, name))
    except Exception:
        return float(default)


def _has_feat(state, obj, name):
    try:
        feats = state.type_features[obj.type]
    except Exception:
        return False
    return name in feats


def _xyz(state, obj):
    return np.array(
        [_feat(state, obj, "x"), _feat(state, obj, "y"), _feat(state, obj, "z")],
        dtype=np.float64,
    )


def _type_name(obj):
    try:
        return obj.type.name
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# Scene parsing. Everything is by NAME PREFIX / feature signature, never by
# index and never by assuming a count. Note in particular that bin_0 is typed
# `mujoco_movable_object`, exactly like the cubes, so type alone cannot
# distinguish them -- name prefix is the only reliable discriminator here.
# ----------------------------------------------------------------------------

class _Scene:
    def __init__(self, state):
        self.robot = None
        self.cubes = []     # list of objects named cube_*
        self.bins = []      # list of objects named bin_*
        self.fixtures = []  # static geometry, e.g. cuboid_barrier

        try:
            names = list(state.get_object_names())
        except Exception:
            names = []

        for nm in names:
            try:
                obj = state.get_object_from_name(nm)
            except Exception:
                continue

            tn = _type_name(obj)

            # The robot is the only object carrying pos_base_x. Identify it by
            # feature rather than by the literal name "robot", since the robot
            # name comes from the task config and could differ.
            if _has_feat(state, obj, "pos_base_x"):
                self.robot = obj
                continue

            if nm.startswith("cube_"):
                self.cubes.append((nm, obj))
            elif nm.startswith("bin_"):
                self.bins.append((nm, obj))
            elif tn == "mujoco_fixture":
                self.fixtures.append((nm, obj))

        # Deterministic ordering by numeric suffix where possible, so behaviour
        # does not depend on set iteration order.
        self.cubes.sort(key=lambda p: _suffix_key(p[0]))
        self.bins.sort(key=lambda p: _suffix_key(p[0]))
        self.fixtures.sort(key=lambda p: p[0])

    @property
    def cube_objs(self):
        return [o for _, o in self.cubes]

    @property
    def bin_objs(self):
        return [o for _, o in self.bins]


def _suffix_key(name):
    tail = name.rsplit("_", 1)[-1]
    try:
        return (0, int(tail))
    except Exception:
        return (1, name)


# ----------------------------------------------------------------------------
# Goal test. The env's own check is a region-box containment test that I cannot
# see. I approximate "cube is in the bin" geometrically: inside the bin's
# horizontal footprint (from bb_x/bb_y) and not far below the bin rim. This is
# only used to decide which cube to head toward next; it does not affect the
# env's actual termination.
# ----------------------------------------------------------------------------

def _cube_in_bin(state, cube, bin_obj):
    if bin_obj is None:
        return False
    c = _xyz(state, cube)
    b = _xyz(state, bin_obj)
    hx = 0.5 * _feat(state, bin_obj, "bb_x", 0.30)
    hy = 0.5 * _feat(state, bin_obj, "bb_y", 0.30)
    hz = _feat(state, bin_obj, "bb_z", 0.20)
    if abs(c[0] - b[0]) > hx:
        return False
    if abs(c[1] - b[1]) > hy:
        return False
    # Allow a generous vertical band: resting on the bin floor up to above rim.
    if c[2] < b[2] - 0.5 * hz:
        return False
    if c[2] > b[2] + 2.0 * hz:
        return False
    return True


def _nearest_bin(state, scene, point):
    best, best_d = None, None
    for b in scene.bin_objs:
        d = float(np.linalg.norm(_xyz(state, b)[:2] - point[:2]))
        if best_d is None or d < best_d:
            best, best_d = b, d
    return best


# ----------------------------------------------------------------------------
# Action assembly.
#
# The action is 18-D for this family (use_arm_velocities=True):
#   base(3) + arm joints(7) + gripper(1) + arm joint velocities(7).
# The block ORDER above is taken from the env card's wording
# ("base pos and yaw (3), arm joints (7), gripper pos (1), and arm joint
# velocity targets (7)"). I have not seen TidyBot3DRobotActionSpace, so if the
# real layout differs this policy will command the wrong channels. Everything
# is clipped into action_space bounds regardless, so it stays legal.
#
# If the action space turns out NOT to be 18-D (e.g. the 11-D variant without
# velocities), _build falls back to filling whatever length is required.
# ----------------------------------------------------------------------------

_NEUTRAL_ARM = np.array(
    # Roughly the posture seen in the provided initial states: arm folded,
    # nothing near a joint limit. Used as a hold target so the arm is quiet.
    [0.0, -0.349065906, 3.14159266, -2.54818060, 0.0, -0.872664631, 1.57079633],
    dtype=np.float64,
)


class _ActionBuilder:
    def __init__(self, action_space):
        self.space = action_space
        self.n = 18
        self.low = None
        self.high = None
        try:
            shape = action_space.shape
            if shape is not None and len(shape) >= 1 and shape[0] > 0:
                self.n = int(shape[0])
        except Exception:
            pass
        try:
            self.low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
            self.high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
            if self.low.shape[0] != self.n:
                self.low = None
                self.high = None
        except Exception:
            self.low = None
            self.high = None

        # Block offsets, only meaningful if n >= 11.
        self.i_base = 0
        self.i_arm = 3
        self.i_grip = 10
        self.i_vel = 11

    def zeros(self):
        return np.zeros(self.n, dtype=np.float64)

    def finish(self, a):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.shape[0] < self.n:
            a = np.concatenate([a, np.zeros(self.n - a.shape[0])])
        elif a.shape[0] > self.n:
            a = a[: self.n]
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if self.low is not None and self.high is not None:
            a = np.clip(a, self.low, self.high)
        # Match the dtype the space advertises, defaulting to float32.
        try:
            dt = self.space.dtype
        except Exception:
            dt = np.float32
        try:
            return a.astype(dt)
        except Exception:
            return a.astype(np.float32)

    def set_base(self, a, bx, by, brot):
        if self.n >= 3:
            a[self.i_base + 0] = bx
            a[self.i_base + 1] = by
            a[self.i_base + 2] = brot

    def set_arm(self, a, q):
        if self.n >= self.i_arm + 7:
            a[self.i_arm : self.i_arm + 7] = q[:7]

    def set_grip(self, a, g):
        if self.n >= self.i_grip + 1:
            a[self.i_grip] = g

    def set_arm_vel(self, a, v):
        if self.n >= self.i_vel + 7:
            a[self.i_vel : self.i_vel + 7] = v[:7]


def _wrap_angle(t):
    return (t + np.pi) % (2.0 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# The approach.
# ----------------------------------------------------------------------------

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives if primitives is not None else {}

        self.ab = _ActionBuilder(action_space)

        # Episode state.
        self._t = 0
        self._target_idx = 0
        self._switch_t = 0
        self._last_good_action = None

        # Tunables. Deliberately conservative: small per-step base commands so
        # that if the base channels are DELTAS we move smoothly, and if they
        # are ABSOLUTE targets we still command a sane nearby pose (we build
        # absolute targets and, when the space bounds look small, fall back to
        # a delta interpretation).
        self._standoff = 0.55       # metres to hold off from a target
        self._max_base_step = 0.35  # cap on any single base command magnitude
        self._switch_period = 90    # steps before re-picking a target

    # -- helpers -------------------------------------------------------------

    def _base_pose(self, state, scene):
        if scene.robot is None:
            return np.zeros(3)
        return np.array(
            [
                _feat(state, scene.robot, "pos_base_x"),
                _feat(state, scene.robot, "pos_base_y"),
                _feat(state, scene.robot, "pos_base_rot"),
            ],
            dtype=np.float64,
        )

    def _remaining(self, state, scene):
        """Cubes not yet (approximately) in a bin."""
        out = []
        for c in scene.cube_objs:
            done = False
            for b in scene.bin_objs:
                if _cube_in_bin(state, c, b):
                    done = True
                    break
            if not done:
                out.append(c)
        return out

    def _choose_goal_point(self, state, scene):
        """
        Pick an (x, y) the base should head toward.

        Alternate between (a) the next unplaced cube and (b) the bin, so the
        base sweeps the corridor between them. With base motion of unknown
        semantics this is the most useful generic behaviour I can commit to:
        it at least keeps the robot in the neighbourhood of the objects that
        matter instead of sitting at the origin accruing -0.01 per step.
        """
        base = self._base_pose(state, scene)
        rem = self._remaining(state, scene)

        if not rem:
            # Nothing left to do by our own estimate; hover near a bin.
            b = _nearest_bin(state, scene, base)
            if b is None:
                return base[:2].copy()
            return _xyz(state, b)[:2]

        # Rotate target every _switch_period steps, and index modulo the
        # CURRENT number of remaining cubes -- never assume a fixed count.
        if self._t - self._switch_t >= self._switch_period:
            self._switch_t = self._t
            self._target_idx += 1

        phase = (self._target_idx // 1) % 2
        cube = rem[self._target_idx % len(rem)]

        if phase == 0:
            return _xyz(state, cube)[:2]
        b = _nearest_bin(state, scene, _xyz(state, cube))
        if b is None:
            return _xyz(state, cube)[:2]
        return _xyz(state, b)[:2]

    def _standoff_point(self, base_xy, goal_xy):
        """Stop `self._standoff` short of the goal, so we do not drive into it."""
        d = goal_xy - base_xy
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return goal_xy.copy()
        if n <= self._standoff:
            return base_xy.copy()
        return goal_xy - (d / n) * self._standoff

    # -- API -----------------------------------------------------------------

    def reset(self, state, info):
        self._t = 0
        self._target_idx = 0
        self._switch_t = 0
        self._last_good_action = None
        # Nothing is precomputed: the cube set is read fresh every step so the
        # policy is indifferent to how many cubes this instance has.
        return None

    def get_action(self, state):
        self._t += 1
        try:
            a = self._policy(state)
            self._last_good_action = a
            return a
        except Exception:
            # Never let a policy bug turn into a crashed rollout.
            if self._last_good_action is not None:
                return self._last_good_action
            return self.ab.finish(self.ab.zeros())

    def _policy(self, state):
        ab = self.ab
        a = ab.zeros()

        scene = _Scene(state)
        if scene.robot is None:
            return ab.finish(a)

        base = self._base_pose(state, scene)
        goal_xy = self._choose_goal_point(state, scene)
        want_xy = self._standoff_point(base[:2], goal_xy)

        # Face the goal.
        d = goal_xy - base[:2]
        if float(np.linalg.norm(d)) > 1e-6:
            want_rot = float(np.arctan2(d[1], d[0]))
        else:
            want_rot = base[2]

        # Build BOTH interpretations and pick by the action-space bounds.
        # If |high| for the base channels is small (order ~1), the channels are
        # almost certainly deltas; if large (order of metres of world extent),
        # absolute targets.
        delta_xy = want_xy - base[:2]
        nrm = float(np.linalg.norm(delta_xy))
        if nrm > self._max_base_step:
            delta_xy = delta_xy / nrm * self._max_base_step
        delta_rot = _wrap_angle(want_rot - base[2])
        delta_rot = float(np.clip(delta_rot, -0.3, 0.3))

        use_delta = True
        if ab.low is not None and ab.high is not None and ab.n >= 3:
            span = float(np.max(np.abs(ab.high[:2])))
            if span > 2.5:
                use_delta = False

        if use_delta:
            ab.set_base(a, delta_xy[0], delta_xy[1], delta_rot)
        else:
            ab.set_base(a, want_xy[0], want_xy[1], want_rot)

        # Arm: hold the neutral posture. If the arm channels are deltas, zeros
        # are the correct "hold"; if absolute, the neutral posture is correct.
        # Distinguish the same way as for the base.
        arm_absolute = False
        if ab.low is not None and ab.high is not None and ab.n >= ab.i_arm + 7:
            arm_span = float(np.max(np.abs(ab.high[ab.i_arm : ab.i_arm + 7])))
            if arm_span > 2.0:
                arm_absolute = True
        if arm_absolute:
            ab.set_arm(a, _NEUTRAL_ARM)
        else:
            ab.set_arm(a, np.zeros(7))

        # Gripper: hold closed. Closed is 1.0 in the [0,1] convention used by
        # pos_gripper (ctrl/255). If the channel is a delta this is a mild
        # close command, which is harmless.
        ab.set_grip(a, 1.0)

        # Arm joint velocity targets: zero. This is the block that a genuine
        # tossing policy would drive, and the reason I flagged the missing
        # action-space definition -- without knowing sign conventions and
        # limits, commanding it is more likely to destabilise than to help.
        ab.set_arm_vel(a, np.zeros(7))

        return ab.finish(a)