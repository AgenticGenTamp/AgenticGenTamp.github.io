"""
GeneratedApproach for kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0

IMPORTANT HONESTY NOTE (please read before trusting this policy):

I was asked to implement "the simple strategy that solves all instances of this
environment." I do not know such a strategy, and I said so. I have not been told
what it is, and I have not been able to verify one by inspecting the task JSON
(the goal predicates / region ranges attached to bowl_0), which is what would be
needed to know whether a shortcut exists. So what follows is NOT a verified
solution. It is my best-effort policy given what the observation and action
spaces actually permit, written so that it degrades gracefully rather than
pretending to a capability it does not have.

What I believe about the task, from the provided source:

  * Action space is 11-dim: base (x, y, yaw) + 7 arm joint commands + 1 gripper.
    There is NO Cartesian end-effector interface exposed here, so a grasp needs
    either an IK solve against Kinova Gen3 kinematics (whose DH/URDF parameters
    are not given to me in the prompt) or a hand-tuned joint trajectory. I have
    neither, so I cannot write a reliable pick-and-place.
  * The env config has act_delta=True by default for TidyBot3DConfig, meaning
    actions are most likely interpreted as DELTAS. I am not fully certain of the
    per-dimension semantics or scaling of TidyBot3DRobotActionSpace, because
    that class body was not included in the prompt. This uncertainty is the main
    reason the policy below is conservative.
  * Success requires BOTH boxed_drink_0 and can_0 to be placed next to bowl_0,
    within ~5cm, judged by a region attached to the bowl.

The one genuine asymmetry I can see, and which this policy leans on:

  bowl_0 is itself a MOVABLE object. The goal region is attached to the bowl.
  Therefore moving the BOWL toward the drink and can may satisfy both predicates
  at once, instead of performing two separate grasp-transport-place cycles. If
  the region is generous, a single short nudge of the bowl to the centroid of
  the other two objects is dramatically easier than two grasps. This is the
  closest thing to a "simple strategy" that I can justify from the source, and
  it is what the policy attempts. I want to be explicit that I have NOT verified
  the region bounds make this sufficient.

Strategy implemented, in order of preference:

  1. NO-OP CHECK. First, hold still for a few steps and see whether the episode
     terminates on its own (i.e. the initial state already satisfies the goal).
     Costs almost nothing (-0.01/step) and rules the possibility in or out.
  2. BASE-PUSH OF THE BOWL. Drive the mobile base so as to push bowl_0 toward
     the centroid of boxed_drink_0 and can_0, using closed-loop feedback on the
     observed bowl position. The arm is parked in a low, forward posture so the
     robot body/arm can contact the bowl. This is crude but it is the only
     manipulation primitive I can implement without arm IK.
  3. SAFE IDLE. If the bowl stops responding (no measurable progress), stop
     issuing pushes rather than thrash and risk knocking objects off the
     counter. Emit zero/hold actions.

Explicitly NOT done: I do not call env.set_state or otherwise write object poses
directly. That would "solve" the task by editing the simulator rather than by
acting in it, which is not a solution to the stated problem.

Expected outcome: this will likely NOT reliably solve the task. I would rather
submit something honest and legible than something that looks confident and
isn't. If you have the actual intended strategy, tell me and I will implement it
properly.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Observation layout (from the documented table in the env description).
# ----------------------------------------------------------------------------

IDX_BOWL_X, IDX_BOWL_Y, IDX_BOWL_Z = 0, 1, 2
IDX_BOWL_BB = 13  # bb_x, bb_y, bb_z at 13,14,15

IDX_DRINK_X, IDX_DRINK_Y, IDX_DRINK_Z = 16, 17, 18
IDX_DRINK_BB = 29

IDX_CAN_X, IDX_CAN_Y, IDX_CAN_Z = 32, 33, 34
IDX_CAN_BB = 45

IDX_ROBOT_BASE_X = 93
IDX_ROBOT_BASE_Y = 94
IDX_ROBOT_BASE_ROT = 95
IDX_ROBOT_ARM_J1 = 96   # through 102
IDX_ROBOT_GRIPPER = 103
IDX_ROBOT_VEL_BASE_X = 104

# Arm "park low and forward" posture. This is the documented home pose from the
# example initial states, which is a known-safe configuration (the arm starts
# there every episode), tilted slightly so the body is lower/forward. I keep it
# close to home deliberately: I cannot verify joint limits or self-collision for
# poses I invent, and driving into a bad configuration risks destabilising the
# sim far more than it gains.
ARM_HOME = np.array(
    [0.0, -0.349066, 3.14159265, -2.54818058, 0.0, -0.87266463, 1.57079637],
    dtype=np.float64,
)


def _wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class GeneratedApproach:
    # --- tuning -------------------------------------------------------------
    # Number of initial steps spent doing nothing, to test whether the goal is
    # already satisfied (strategy step 1).
    N_NOOP_PROBE = 6

    # How close (metres) the bowl must get to the target centroid before we stop
    # pushing. The reward doc says 5cm tolerance; aim tighter than that.
    BOWL_GOAL_TOL = 0.030

    # Standoff behind the bowl, along the push direction, where the base aims.
    PUSH_STANDOFF = 0.34

    # Base gains (these act on whatever the action units turn out to be; kept
    # small because the semantics are uncertain and overshoot is destructive).
    K_BASE_POS = 0.8
    K_BASE_YAW = 0.8

    # Progress watchdog: if the bowl has not moved this far over this many
    # steps, give up pushing and idle (strategy step 3).
    PROGRESS_WINDOW = 120
    PROGRESS_MIN_DELTA = 0.015

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives if primitives is not None else {}

        self.adim = int(np.prod(action_space.shape))
        self.low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
        self.high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)

        # Whether the action space looks like deltas centred on zero. If low<0<high
        # for the base dims, treat as delta control; otherwise treat as absolute.
        self.delta_mode = bool(self.low[0] < -1e-6 and self.high[0] > 1e-6)

        self._reset_internal()

    # ------------------------------------------------------------------ setup
    def _reset_internal(self):
        self.t = 0
        self.phase = "probe"
        self.target_xy = None          # where the bowl should end up
        self.last_bowl_xy = None
        self.progress_ref_xy = None
        self.progress_ref_t = 0

    def reset(self, state, info):
        self._reset_internal()
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        self.target_xy = self._compute_bowl_target(state)
        self.progress_ref_xy = state[[IDX_BOWL_X, IDX_BOWL_Y]].copy()
        self.progress_ref_t = 0
        return None

    # ------------------------------------------------------------- goal logic
    def _compute_bowl_target(self, state):
        """Where to push the bowl: the centroid of drink and can.

        Rationale: the goal is 'drink next to bowl' AND 'can next to bowl'. If
        the bowl sits at the midpoint of the two, both are as close to it as a
        single bowl placement can make them. This is the whole basis of the
        move-the-bowl strategy.
        """
        drink = state[[IDX_DRINK_X, IDX_DRINK_Y]]
        can = state[[IDX_CAN_X, IDX_CAN_Y]]
        centroid = 0.5 * (drink + can)

        # Do not aim exactly at the centroid if the drink and can are far apart;
        # in that case the bowl cannot be near both and the whole premise fails.
        # Aim at the centroid anyway (best available), but note it.
        self._pair_separation = float(np.linalg.norm(drink - can))
        return centroid

    # ------------------------------------------------------------ act helpers
    def _zero_action(self, state):
        """An action that requests 'no change'.

        In delta mode that is all zeros. In absolute mode we must command the
        CURRENT measured configuration, otherwise zeros would be a hard command
        to drive everything to the origin, which would be catastrophic.
        """
        a = np.zeros(self.adim, dtype=np.float64)
        if not self.delta_mode:
            a[0] = state[IDX_ROBOT_BASE_X]
            a[1] = state[IDX_ROBOT_BASE_Y]
            a[2] = state[IDX_ROBOT_BASE_ROT]
            n_arm = min(7, max(0, self.adim - 4))
            a[3:3 + n_arm] = state[IDX_ROBOT_ARM_J1:IDX_ROBOT_ARM_J1 + n_arm]
            if self.adim >= 11:
                a[10] = state[IDX_ROBOT_GRIPPER]
        else:
            # Hold the arm where it is: zero delta. Gripper: hold open (zero
            # delta) so we never accidentally clamp on something.
            pass
        return self._finish(a)

    def _arm_hold_terms(self, state, a):
        """Fill arm + gripper entries so the arm stays at/near home."""
        n_arm = min(7, max(0, self.adim - 4))
        if n_arm <= 0:
            return
        cur = state[IDX_ROBOT_ARM_J1:IDX_ROBOT_ARM_J1 + n_arm]
        tgt = ARM_HOME[:n_arm]
        if self.delta_mode:
            # Gentle proportional pull back toward home, so drift is corrected
            # but we never command a large joint jump.
            a[3:3 + n_arm] = np.clip(0.5 * (tgt - cur), -0.05, 0.05)
        else:
            a[3:3 + n_arm] = tgt
        if self.adim >= 11:
            # Keep gripper open / unchanged.
            a[10] = 0.0 if self.delta_mode else state[IDX_ROBOT_GRIPPER]

    def _finish(self, a):
        a = np.clip(a, self.low, self.high)
        return a.astype(np.float32, copy=False)

    # --------------------------------------------------------------- planning
    def _base_command(self, state, desired_xy, desired_yaw):
        """Produce base action entries driving toward (desired_xy, desired_yaw)."""
        a = np.zeros(self.adim, dtype=np.float64)
        bx = state[IDX_ROBOT_BASE_X]
        by = state[IDX_ROBOT_BASE_Y]
        brot = state[IDX_ROBOT_BASE_ROT]

        if self.delta_mode:
            ex = desired_xy[0] - bx
            ey = desired_xy[1] - by
            eyaw = _wrap_angle(desired_yaw - brot)
            a[0] = self.K_BASE_POS * ex
            a[1] = self.K_BASE_POS * ey
            a[2] = self.K_BASE_YAW * eyaw
        else:
            a[0] = desired_xy[0]
            a[1] = desired_xy[1]
            a[2] = desired_yaw

        self._arm_hold_terms(state, a)
        return a

    def _push_plan(self, state):
        """Aim the base at a standoff point behind the bowl, along the push line."""
        bowl = state[[IDX_BOWL_X, IDX_BOWL_Y]]
        goal = self.target_xy
        d = goal - bowl
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return None, None, dist
        u = d / dist
        # Stand behind the bowl, opposite the goal, so moving forward pushes it.
        stand = bowl - u * self.PUSH_STANDOFF
        yaw = float(np.arctan2(u[1], u[0]))
        return stand, yaw, dist

    # ----------------------------------------------------------------- policy
    def get_action(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        self.t += 1

        if self.target_xy is None:
            self.target_xy = self._compute_bowl_target(state)
            self.progress_ref_xy = state[[IDX_BOWL_X, IDX_BOWL_Y]].copy()
            self.progress_ref_t = self.t

        # --- Phase 1: no-op probe. Does the episode already terminate? -------
        if self.phase == "probe":
            if self.t <= self.N_NOOP_PROBE:
                return self._zero_action(state)
            self.phase = "push"

        bowl = state[[IDX_BOWL_X, IDX_BOWL_Y]]

        # --- Success check on our own proxy goal ----------------------------
        if self.phase == "push":
            if float(np.linalg.norm(self.target_xy - bowl)) <= self.BOWL_GOAL_TOL:
                self.phase = "idle"

        # --- Progress watchdog ----------------------------------------------
        if self.phase == "push":
            if self.t - self.progress_ref_t >= self.PROGRESS_WINDOW:
                moved = float(np.linalg.norm(bowl - self.progress_ref_xy))
                if moved < self.PROGRESS_MIN_DELTA:
                    # Pushing is not working. Stop rather than thrash: further
                    # blind pushing mostly risks knocking the tall can over or
                    # sweeping objects off the counter, which can only make
                    # things worse.
                    self.phase = "idle"
                else:
                    self.progress_ref_xy = bowl.copy()
                    self.progress_ref_t = self.t

        # --- Phase 3: idle ---------------------------------------------------
        if self.phase == "idle":
            return self._zero_action(state)

        # --- Phase 2: base push ---------------------------------------------
        stand, yaw, dist = self._push_plan(state)
        if stand is None:
            return self._zero_action(state)

        a = self._base_command(state, stand, yaw)
        return self._finish(a)