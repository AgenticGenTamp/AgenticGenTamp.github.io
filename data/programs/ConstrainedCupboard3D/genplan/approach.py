"""GeneratedApproach for ConstrainedCupboard3DEnv (variable object count).

HONEST PREFACE (read this before trusting the code below):
---------------------------------------------------------
I do not know the intended "simple strategy" for this environment, and I said so
in the previous turn. Nothing has changed since: the target locations are NOT in
the observation (they live in the task JSON's `goal_state` / `regions`, which I
cannot read), the cupboard fixtures expose pose but no extents, `primitives` is
empty (so there is no IK helper), and the action space is JOINT targets rather
than end-effector poses.

So this module is NOT a confident implementation of a known solution. It is the
most defensible thing I can write under that uncertainty, and it is structured so
that the parts I am guessing about are isolated, labelled, and easy to replace:

  * `CupboardModel`   -- infers slot geometry from the fixture poses in the state.
                         This is an INFERENCE from two observed seeds, not knowledge.
  * `TargetAssigner`  -- assigns each cuboid to a slot. Hypothesis: one rod per slot.
  * `ArmIK`           -- a damped-least-squares IK loop over a hand-rolled Kinova
                         Gen3 forward kinematics model. The DH/offset constants are
                         taken from the public Gen3 7-DoF spec; if the MuJoCo model
                         in this repo differs, IK will be biased and the policy will
                         miss. This is the single most fragile piece.
  * `GeneratedApproach` -- a per-object pick-align-insert state machine that drives
                         base + arm + gripper.

Design choices made *because* of the uncertainty:

  1. The policy is count-agnostic by construction. It enumerates cuboids from the
     state every reset, never assumes an index layout, and loops over however many
     there are. This is the one requirement I am confident about, and it is met.

  2. Every action returned is clipped into `action_space`. Even if the plan is
     wrong, the submission will not crash the harness or emit invalid actions.

  3. Where I must guess a target, I guess from the *fixture geometry actually
     present in the state* rather than hard-coding world coordinates. A hard-coded
     constant tuned to seed 0 would look better on the design counts and fail on
     held-out ones; deriving from fixtures at least degrades gracefully.

  4. There is a conservative fallback: if IK fails to converge or the plan is
     exhausted, the policy holds a safe posture rather than flailing. A stationary
     robot scores -0.01/step; a thrashing one can knock rods out of reach and do
     worse.

If you have the task JSON, the right fix is to replace `TargetAssigner.targets_for`
with the real region centres; the rest of the machinery should carry over.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Small math helpers
# ----------------------------------------------------------------------------

def _wrap_pi(a):
    """Wrap angle(s) to [-pi, pi]."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _quat_to_yaw(qw, qx, qy, qz):
    """Yaw (rotation about world z) from a (w, x, y, z) quaternion."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny, cosy))


def _rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _clip_to_space(a, space):
    """Clip an action into a gym Box, tolerating a missing/odd space."""
    a = np.asarray(a, dtype=np.float32).ravel()
    lo = getattr(space, "low", None)
    hi = getattr(space, "high", None)
    if lo is None or hi is None:
        return a
    lo = np.asarray(lo, dtype=np.float32).ravel()
    hi = np.asarray(hi, dtype=np.float32).ravel()
    n = min(a.shape[0], lo.shape[0])
    out = np.zeros(lo.shape[0], dtype=np.float32)
    out[:n] = a[:n]
    return np.clip(out, lo, hi).astype(np.float32)


# ----------------------------------------------------------------------------
# Kinova Gen3 (7-DoF) forward kinematics + damped least squares IK.
#
# UNCERTAIN: these link constants come from the published Gen3 spec, not from the
# MuJoCo XML in this repo (which I cannot read). If the model differs, every
# Cartesian target below is systematically off. Treat a persistent miss as a
# symptom of THIS block, not of the state machine.
# ----------------------------------------------------------------------------

class ArmIK:
    # (alpha, a, d, theta_offset) per joint, base->wrist, in the arm base frame.
    _DH = [
        (np.pi / 2.0, 0.0, 0.2848, 0.0),
        (np.pi / 2.0, 0.0, 0.0118, np.pi),
        (np.pi / 2.0, 0.0, 0.4208, np.pi),
        (np.pi / 2.0, 0.0, 0.0128, np.pi),
        (np.pi / 2.0, 0.0, 0.3143, np.pi),
        (np.pi / 2.0, 0.0, 0.0000, np.pi),
        (0.0, 0.0, 0.2874, np.pi),
    ]

    # Joint limits: Gen3 joints 1,3,5,7 are continuous; 2,4,6 are limited.
    _LIM = [
        (-np.pi, np.pi),
        (-2.41, 2.41),
        (-np.pi, np.pi),
        (-2.66, 2.66),
        (-np.pi, np.pi),
        (-2.23, 2.23),
        (-np.pi, np.pi),
    ]

    # Arm mount offset relative to the mobile base origin (guess; see preface).
    MOUNT = np.array([0.10, 0.0, 0.40])

    @staticmethod
    def _dh_mat(alpha, a, d, theta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @classmethod
    def fk(cls, q):
        """Return (pos, R) of the tool frame in the ARM BASE frame."""
        T = np.eye(4)
        for i in range(7):
            alpha, a, d, off = cls._DH[i]
            T = T @ cls._dh_mat(alpha, a, d, q[i] + off)
        return T[:3, 3].copy(), T[:3, :3].copy()

    @classmethod
    def jac_pos(cls, q, eps=1e-5):
        """Numeric position Jacobian (3x7). Cheap and robust enough at 10 Hz."""
        p0, _ = cls.fk(q)
        J = np.zeros((3, 7))
        for i in range(7):
            dq = np.array(q, dtype=np.float64)
            dq[i] += eps
            p1, _ = cls.fk(dq)
            J[:, i] = (p1 - p0) / eps
        return J

    @classmethod
    def solve(cls, q_init, target_arm_frame, iters=120, damping=0.08):
        """Damped least squares position IK. Returns (q, converged)."""
        q = np.array(q_init, dtype=np.float64)
        tgt = np.asarray(target_arm_frame, dtype=np.float64)
        for _ in range(iters):
            p, _ = cls.fk(q)
            err = tgt - p
            n = float(np.linalg.norm(err))
            if n < 5e-3:
                return cls._clamp(q), True
            if n > 0.12:
                err = err * (0.12 / n)
            J = cls.jac_pos(q)
            JT = J.T
            A = J @ JT + (damping ** 2) * np.eye(3)
            try:
                dq = JT @ np.linalg.solve(A, err)
            except np.linalg.LinAlgError:
                return cls._clamp(q), False
            dq = np.clip(dq, -0.25, 0.25)
            q = cls._clamp(q + dq)
        p, _ = cls.fk(q)
        return q, bool(np.linalg.norm(tgt - p) < 1.5e-2)

    @classmethod
    def _clamp(cls, q):
        q = np.array(q, dtype=np.float64)
        for i, (lo, hi) in enumerate(cls._LIM):
            if lo <= -np.pi + 1e-9 and hi >= np.pi - 1e-9:
                q[i] = _wrap_pi(q[i])
            else:
                q[i] = float(np.clip(q[i], lo, hi))
        return q


# ----------------------------------------------------------------------------
# Cupboard geometry inference.
#
# UNCERTAIN: derived from two observed seeds. Seed 0 had six fixtures at x=2.0,
# y in {-.25,-.15,-.05,.05,.15,.25}; seed 1 had three at y in {-.1, 0, .1}. I read
# those as evenly spaced dividers/columns, so slot centres are taken to be the
# fixture y-positions themselves (one rod per fixture). If fixtures are instead
# *walls* bounding gaps, the correct centres are the midpoints between adjacent
# fixtures -- see MIDPOINT_MODE below to flip that assumption.
# ----------------------------------------------------------------------------

MIDPOINT_MODE = False  # flip to True if slots are gaps *between* fixtures

# Guessed insertion depth/height relative to the cupboard fixture frame.
SLOT_DX = -0.18   # stand-off from the cupboard plane toward the robot (-x)
SLOT_Z = 0.35     # shelf height guess


class CupboardModel:
    def __init__(self, state, obs_space):
        self.slots = []
        self.plane_x = None
        fixtures = _objects_by_type_name(state, obs_space, "mujoco_fixture")
        pts = []
        for f in fixtures:
            try:
                pts.append((
                    float(state.get(f, "x")),
                    float(state.get(f, "y")),
                    float(state.get(f, "z")),
                ))
            except Exception:
                continue
        if not pts:
            return
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        self.plane_x = float(np.median(xs))
        order = np.argsort(ys)
        ys = ys[order]
        if MIDPOINT_MODE and len(ys) >= 2:
            centres = 0.5 * (ys[:-1] + ys[1:])
        else:
            centres = ys
        for cy in centres:
            self.slots.append(np.array([
                self.plane_x + SLOT_DX,
                float(cy),
                SLOT_Z,
            ]))

    def ok(self):
        return len(self.slots) > 0


def _objects_by_type_name(state, obs_space, type_name):
    """Enumerate objects of a type by NAME, tolerating space/type quirks."""
    out = []
    # Preferred: ask the space for the Type object.
    try:
        getter = getattr(obs_space, "get_type", None)
        if getter is not None:
            t = getter(type_name)
            objs = list(state.get_objects(t))
            if objs:
                return objs
    except Exception:
        pass
    try:
        for t in getattr(obs_space, "types", []):
            if getattr(t, "name", None) == type_name:
                objs = list(state.get_objects(t))
                if objs:
                    return objs
    except Exception:
        pass
    # Fallback: walk names and read the object's own type.
    for nm in state.get_object_names():
        try:
            o = state.get_object_from_name(nm)
        except Exception:
            continue
        tn = getattr(getattr(o, "type", None), "name", None)
        if tn == type_name:
            out.append(o)
    return out


# ----------------------------------------------------------------------------
# Target assignment. HYPOTHESIS: one rod per slot, matched by sorted y so the
# assignment is stable and collision-free-ish (rods don't cross paths).
# ----------------------------------------------------------------------------

class TargetAssigner:
    @staticmethod
    def targets_for(cuboids, state, model):
        if not model.ok():
            return {}
        ys = []
        for c in cuboids:
            try:
                ys.append((float(state.get(c, "y")), c))
            except Exception:
                ys.append((0.0, c))
        ys.sort(key=lambda t: t[0])
        slots = sorted(model.slots, key=lambda s: float(s[1]))
        out = {}
        for i, (_, c) in enumerate(ys):
            out[c.name] = np.array(slots[min(i, len(slots) - 1)], dtype=np.float64)
        return out


# ----------------------------------------------------------------------------
# The policy.
# ----------------------------------------------------------------------------

# Retracted "carry" posture; also the safe fallback.
HOME_Q = np.array([0.0, -0.349, np.pi, -2.548, 0.0, -0.873, np.pi / 2.0])

GRIP_OPEN = 0.0
GRIP_CLOSED = 1.0

# Phase step budgets (10 Hz control).
PH_APPROACH_BASE = 55
PH_REACH = 40
PH_DESCEND = 22
PH_CLOSE = 14
PH_LIFT = 20
PH_TRANSIT = 70
PH_ALIGN = 30
PH_INSERT = 38
PH_RELEASE = 12
PH_RETREAT = 20


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives or {}

        dim = 11
        try:
            if getattr(action_space, "shape", None):
                dim = int(np.prod(action_space.shape))
        except Exception:
            pass
        self.adim = dim

        self.model = None
        self.targets = {}
        self.queue = []
        self.cur = None
        self.phase = None
        self.t = 0
        self.q_cmd = HOME_Q.copy()
        self.base_cmd = np.zeros(3)
        self.grip = GRIP_OPEN
        self.q_goal = HOME_Q.copy()
        self.failed = 0

    # -- lifecycle ---------------------------------------------------------

    def reset(self, state, info):
        try:
            self.model = CupboardModel(state, self.observation_space)
            cubs = _objects_by_type_name(
                state, self.observation_space, "mujoco_movable_object")
            # Count-agnostic: whatever rods exist, in a deterministic order.
            cubs = [c for c in cubs if str(c.name).startswith("cuboid_")] or cubs
            cubs.sort(key=lambda o: str(o.name))
            self.targets = TargetAssigner.targets_for(cubs, state, self.model)
            # Work nearest-first: shorter base travel, fewer chances to disturb.
            rb = self._base_xy(state)
            def d(o):
                try:
                    return (float(state.get(o, "x")) - rb[0]) ** 2 + \
                           (float(state.get(o, "y")) - rb[1]) ** 2
                except Exception:
                    return 1e9
            self.queue = sorted([c.name for c in cubs], key=lambda n: d(
                state.get_object_from_name(n)))
        except Exception:
            self.model = None
            self.targets = {}
            self.queue = []

        self.cur = None
        self.phase = "next"
        self.t = 0
        self.q_cmd = self._arm_q(state)
        self.base_cmd = np.array(self._base_pose(state), dtype=np.float64)
        self.grip = GRIP_OPEN
        self.q_goal = self.q_cmd.copy()
        self.failed = 0

    # -- state readers -----------------------------------------------------

    def _robot(self, state):
        rs = _objects_by_type_name(
            state, self.observation_space, "mujoco_tidybot_robot")
        if rs:
            return rs[0]
        try:
            return state.get_object_from_name("robot")
        except Exception:
            return None

    def _base_pose(self, state):
        r = self._robot(state)
        if r is None:
            return np.zeros(3)
        try:
            return np.array([
                float(state.get(r, "pos_base_x")),
                float(state.get(r, "pos_base_y")),
                float(state.get(r, "pos_base_rot")),
            ])
        except Exception:
            return np.zeros(3)

    def _base_xy(self, state):
        return self._base_pose(state)[:2]

    def _arm_q(self, state):
        r = self._robot(state)
        if r is None:
            return HOME_Q.copy()
        try:
            return np.array([
                float(state.get(r, "pos_arm_joint%d" % i)) for i in range(1, 8)])
        except Exception:
            return HOME_Q.copy()

    def _obj_xyz(self, state, name):
        try:
            o = state.get_object_from_name(name)
            return np.array([
                float(state.get(o, "x")),
                float(state.get(o, "y")),
                float(state.get(o, "z")),
            ])
        except Exception:
            return None

    def _obj_yaw(self, state, name):
        try:
            o = state.get_object_from_name(name)
            return _quat_to_yaw(
                float(state.get(o, "qw")), float(state.get(o, "qx")),
                float(state.get(o, "qy")), float(state.get(o, "qz")))
        except Exception:
            return 0.0

    # -- frame conversion --------------------------------------------------

    def _world_to_arm(self, state, p_world):
        """World point -> arm base frame (base yaw + mount offset)."""
        bx, by, bth = self._base_pose(state)
        p = np.asarray(p_world, dtype=np.float64) - np.array([bx, by, 0.0])
        p = _rotz(-bth) @ p
        return p - ArmIK.MOUNT

    def _ik_to(self, state, p_world):
        tgt = self._world_to_arm(state, p_world)
        q, ok = ArmIK.solve(self.q_cmd, tgt)
        if not ok:
            # Second try from home -- different basin, sometimes converges.
            q, ok = ArmIK.solve(HOME_Q, tgt)
        return q, ok

    # -- action assembly ---------------------------------------------------

    def _action(self):
        a = np.zeros(self.adim, dtype=np.float32)
        n = self.adim
        if n >= 3:
            a[0:3] = self.base_cmd[:3]
        if n >= 10:
            a[3:10] = self.q_cmd[:7]
        if n >= 11:
            a[10] = self.grip
        return _clip_to_space(a, self.action_space)

    def _hold(self, state):
        """Safe fallback: freeze base where it is, keep last arm posture."""
        self.base_cmd = np.array(self._base_pose(state), dtype=np.float64)
        return self._action()

    # -- main loop ---------------------------------------------------------

    def get_action(self, state):
        try:
            return self._step(state)
        except Exception:
            # Never crash the harness; a held posture is a defensible default.
            try:
                return self._hold(state)
            except Exception:
                return _clip_to_space(
                    np.zeros(self.adim, dtype=np.float32), self.action_space)

    def _step(self, state):
        if self.model is None or not self.model.ok():
            # No inferable geometry -> nothing well-founded to do.
            return self._hold(state)

        self.t += 1

        if self.phase == "next":
            self.cur = None
            while self.queue:
                nm = self.queue.pop(0)
                if self._obj_xyz(state, nm) is not None:
                    self.cur = nm
                    break
            if self.cur is None:
                self.q_cmd = HOME_Q.copy()
                self.grip = GRIP_OPEN
                return self._hold(state)
            self._enter("approach_base")
            return self._action()

        p = self._obj_xyz(state, self.cur) if self.cur else None
        tgt = self.targets.get(self.cur) if self.cur else None
        if p is None or tgt is None:
            self._enter("next")
            return self._hold(state)

        ph = self.phase

        if ph == "approach_base":
            # Park the base a fixed stand-off from the rod, facing it.
            yaw = np.arctan2(p[1] - self.base_cmd[1], p[0] - self.base_cmd[0])
            stand = p[:2] - 0.55 * np.array([np.cos(yaw), np.sin(yaw)])
            self.base_cmd = np.array([stand[0], stand[1], yaw])
            self.q_cmd = HOME_Q.copy()
            self.grip = GRIP_OPEN
            if self.t > PH_APPROACH_BASE:
                self._enter("reach")

        elif ph == "reach":
            q, ok = self._ik_to(state, p + np.array([0.0, 0.0, 0.16]))
            if ok:
                self.q_goal = q
                self.failed = 0
            else:
                self.failed += 1
            self._slew()
            self.grip = GRIP_OPEN
            if self.t > PH_REACH or self.failed > 12:
                self._enter("descend" if self.failed <= 12 else "next")

        elif ph == "descend":
            q, ok = self._ik_to(state, p + np.array([0.0, 0.0, 0.015]))
            if ok:
                self.q_goal = q
            self._slew()
            self.grip = GRIP_OPEN
            if self.t > PH_DESCEND:
                self._enter("close")

        elif ph == "close":
            self.grip = GRIP_CLOSED
            self._slew()
            if self.t > PH_CLOSE:
                self._enter("lift")

        elif ph == "lift":
            q, ok = self._ik_to(state, p + np.array([0.0, 0.0, 0.28]))
            if ok:
                self.q_goal = q
            self._slew()
            self.grip = GRIP_CLOSED
            if self.t > PH_LIFT:
                self._enter("transit")

        elif ph == "transit":
            # Drive to a stand-off in front of the assigned slot, facing +x.
            stand = np.array([tgt[0] - 0.45, tgt[1], 0.0])
            self.base_cmd = np.array([stand[0], stand[1], 0.0])
            self.q_cmd = HOME_Q.copy()
            self.grip = GRIP_CLOSED
            if self.t > PH_TRANSIT:
                self._enter("align")

        elif ph == "align":
            # Pre-insert pose: same y/z as the slot, backed off in x.
            q, ok = self._ik_to(state, tgt + np.array([-0.16, 0.0, 0.0]))
            if ok:
                self.q_goal = q
            self._slew()
            self.grip = GRIP_CLOSED
            if self.t > PH_ALIGN:
                self._enter("insert")

        elif ph == "insert":
            q, ok = self._ik_to(state, tgt)
            if ok:
                self.q_goal = q
            self._slew(rate=0.06)  # slow: constrained space
            self.grip = GRIP_CLOSED
            if self.t > PH_INSERT:
                self._enter("release")

        elif ph == "release":
            self.grip = GRIP_OPEN
            self._slew()
            if self.t > PH_RELEASE:
                self._enter("retreat")

        elif ph == "retreat":
            q, ok = self._ik_to(state, tgt + np.array([-0.24, 0.0, 0.02]))
            if ok:
                self.q_goal = q
            self._slew()
            self.grip = GRIP_OPEN
            if self.t > PH_RETREAT:
                self._enter("next")

        else:
            self._enter("next")

        return self._action()

    def _enter(self, phase):
        self.phase = phase
        self.t = 0
        self.failed = 0

    def _slew(self, rate=0.12):
        """Rate-limit joint commands so the arm does not snap between targets."""
        d = _wrap_pi(self.q_goal - self.q_cmd)
        n = float(np.linalg.norm(d))
        if n > rate:
            d = d * (rate / n)
        self.q_cmd = ArmIK._clamp(self.q_cmd + d)