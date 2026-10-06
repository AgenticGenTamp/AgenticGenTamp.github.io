"""Model-based pick approach for Table3DEnv (variable cube count).

Builds an internal copy of the kinematic env, simulates candidate pick plans
(cube x gripper yaw x base offset) and executes the first plan that reaches the
goal in simulation. Falls back to closed-loop replanning if reality diverges.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from kinder.envs.kinematic3d.table3d import (
    ObjectCentricTable3DEnv,
    Table3DObjectCentricState,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import pybullet_inverse_kinematics

MAX_D = 0.4


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self._sims = {}
        self._plan = []
        self._expected = None
        self._t0 = None

    # ------------------------------------------------------------------ utils
    def _sim(self, n):
        if n not in self._sims:
            self._sims[n] = ObjectCentricTable3DEnv(num_cubes=n, realistic_bg=False)
        return self._sims[n]

    @staticmethod
    def _cubes(state):
        return sorted(
            [o for o in state.get_object_names() if o.startswith("cube")],
            key=lambda s: int(s[4:]) if s[4:].isdigit() else 0,
        )

    def _to_sim_state(self, state):
        return Table3DObjectCentricState(dict(state.data), state.type_features)

    def _set(self, sim, state):
        sim.reset(options={"init_state": self._to_sim_state(state)})

    @staticmethod
    def _joints(state):
        r = state.get_object_from_name("robot")
        return np.array([state.get(r, f"joint_{i + 1}") for i in range(7)])

    @staticmethod
    def _base(state):
        r = state.get_object_from_name("robot")
        return np.array(
            [state.get(r, "pos_base_x"), state.get(r, "pos_base_y"),
             state.get(r, "pos_base_rot")]
        )

    def _ik(self, sim, pos, quat):
        arm = sim.robot.arm
        saved = arm.get_joint_positions()
        try:
            q = pybullet_inverse_kinematics(arm, Pose(tuple(pos), tuple(quat)),
                                            validate=True)
            return np.array(q[:7])
        except Exception:
            return None
        finally:
            arm.set_joints(saved)

    # ------------------------------------------------------------ simulation
    def _goto(self, sim, s, q, base_t, grip_last, acts, maxsteps=40):
        """Step base+joints toward (base_t, q). Gripper command on final step.

        Returns (state, status) with status in {"ok", "term", "fail"}.
        """
        for _ in range(maxsteps):
            d = _wrap(q - self._joints(s))
            b = self._base(s)
            db = base_t - b
            db[2] = _wrap(db[2])
            if np.max(np.abs(d)) < 1e-5 and np.max(np.abs(db)) < 1e-6:
                return s, "ok"
            a = np.zeros(11, dtype=np.float32)
            a[0:3] = np.clip(db, -MAX_D, MAX_D)
            a[3:10] = np.clip(d, -MAX_D, MAX_D)
            last = np.max(np.abs(d)) <= MAX_D and np.max(np.abs(db)) <= MAX_D
            if last:
                a[10] = grip_last
            prev = self._joints(s)
            s, _, term, _, _ = sim.step(a)
            acts.append(a)
            if term:
                return s, "term"
            if np.allclose(prev, self._joints(s), atol=1e-9) and np.allclose(
                    b, self._base(s), atol=1e-9):
                return s, "fail"  # collision revert
            if last:
                return s, "ok"
        return s, "fail"

    def _try_plan(self, sim, state, cube, yaw, base_off, dz, direct, merge):
        base_t = self._base(state)
        if base_off is not None:
            base_t = base_t + np.array(base_off)
        # set sim to a state with the robot at the target base for IK
        st = state.copy()
        robot = st.get_object_from_name("robot")
        st.set(robot, "pos_base_x", base_t[0])
        st.set(robot, "pos_base_y", base_t[1])
        st.set(robot, "pos_base_rot", base_t[2])
        self._set(sim, st)
        c = state.get_object_from_name(cube)
        cx, cy, cz = (state.get(c, "pose_x"), state.get(c, "pose_y"),
                      state.get(c, "pose_z"))
        cyaw = R.from_quat([state.get(c, "pose_qx"), state.get(c, "pose_qy"),
                            state.get(c, "pose_qz"),
                            state.get(c, "pose_qw")]).as_euler("xyz")[2]
        quat = (R.from_euler("z", yaw + cyaw) * R.from_quat([1, 0, 0, 0])).as_quat()
        q_app = None
        if not direct:
            q_app = self._ik(sim, (cx, cy, cz + 0.15), quat)
            if q_app is None:
                return "ik_fail"
        q_grasp = self._ik(sim, (cx, cy, cz + dz), quat)
        if q_grasp is None:
            return "ik_fail" if direct else None
        lifts = []
        for h in (dz + 0.09, 0.2, 0.3):
            ql = self._ik(sim, (cx, cy, cz + h), quat)
            if ql is not None:
                lifts.append(ql)
        if not lifts:
            return None
        # now simulate from the real state
        self._set(sim, state)
        s = state
        acts = []
        r = s.get_object_from_name("robot")
        if s.get(r, "grasp_active") > 0.5:
            return None
        if s.get(r, "finger_state") > 0.05:
            a = np.zeros(11, dtype=np.float32)
            a[10] = 1.0
            s, _, _, _, _ = sim.step(a)
            acts.append(a)
        if q_app is not None:
            s, st_ = self._goto(sim, s, q_app, base_t, 0.0, acts)
            if st_ == "term":
                return acts
            if st_ == "fail":
                return None
        s, st_ = self._goto(sim, s, q_grasp, base_t, -1.0 if merge else 0.0, acts)
        if st_ == "term":
            return acts
        if st_ == "fail":
            return None
        if not merge:
            a = np.zeros(11, dtype=np.float32)
            a[10] = -1.0
            s, _, term, _, _ = sim.step(a)
            acts.append(a)
            if term:
                return acts
        r = s.get_object_from_name("robot")
        if s.get(r, "grasp_active") < 0.5:
            return None
        for ql in lifts:
            s, st_ = self._goto(sim, s, ql, base_t, 0.0, acts)
            if st_ == "term":
                return acts
        return None

    def _plan_from(self, state):
        t0 = time.time()
        n = len(self._cubes(state))
        sim = self._sim(n)
        rb = self._base(state)
        cubes = self._cubes(state)

        def dist(cn):
            c = state.get_object_from_name(cn)
            return np.hypot(state.get(c, "pose_x") - rb[0] - 0.55,
                            state.get(c, "pose_y") - rb[1])

        cubes = sorted(cubes, key=dist)
        yaws = [0.0, np.pi / 2, np.pi / 4, -np.pi / 4, np.pi / 8, -np.pi / 8,
                3 * np.pi / 8, -3 * np.pi / 8]
        dzs = [0.03, 0.04, 0.02]

        def base_offs(cn):
            c = state.get_object_from_name(cn)
            cx, cy = state.get(c, "pose_x"), state.get(c, "pose_y")
            th = rb[2]
            offs = []
            for tx in (0.6, 0.55, 0.65, 0.5):
                # place base so cube is tx ahead along the base heading
                offs.append((cx - tx * np.cos(th) - rb[0],
                             cy - tx * np.sin(th) - rb[1], 0.0))
            return offs

        def comfortable(cn):
            c = state.get_object_from_name(cn)
            dx = state.get(c, "pose_x") - rb[0]
            dy = state.get(c, "pose_y") - rb[1]
            return 0.4 <= dx <= 0.7 and abs(dy) <= 0.25

        cands = []  # (priority, cube, off)
        for i, cn in enumerate(cubes):
            if comfortable(cn):
                cands.append((0, i, cn, None))
                for off in base_offs(cn):
                    cands.append((2, i, cn, off))
            else:
                for j, off in enumerate(base_offs(cn)):
                    cands.append((1 if j < 2 else 2, i, cn, off))
                cands.append((1, i, cn, None))
        cands.sort(key=lambda x: (x[0], x[1]))
        variants = [(True, True), (False, True), (False, False)]
        best = None
        first_t = None
        for _, _, cn, off in cands:
            base_bad = False
            for yaw in yaws:
                for direct, merge in variants:
                    for dz in dzs:
                        now = time.time()
                        if now - t0 > 40 or (self._t0 is not None
                                             and now - self._t0 > 45):
                            return best or []
                        if first_t is not None and now - first_t > 1.5:
                            return best
                        p = self._try_plan(sim, state, cn, yaw, off, dz,
                                           direct, merge)
                        if isinstance(p, str):
                            break
                        if p is not None:
                            if best is None or len(p) < len(best):
                                best = p
                            if first_t is None:
                                first_t = now
                            break
            if best is not None and len(best) <= 3:
                return best
        return best or []

    # ------------------------------------------------------------- interface
    def reset(self, state, info):
        self._plan = []
        self._expected = None
        self._t0 = time.time()
        try:
            self._plan = list(self._plan_from(state))
        except Exception:
            self._plan = []

    def get_action(self, state):
        if not self._plan:
            try:
                self._plan = list(self._plan_from(state))
            except Exception:
                self._plan = []
        if not self._plan:
            a = np.zeros(11, dtype=np.float32)
            return a
        return np.asarray(self._plan.pop(0), dtype=np.float32)
