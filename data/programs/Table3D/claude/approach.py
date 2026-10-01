"""Approach for Table3DEnv: grasp a cube and lift it above the table.

The robot is a 7-DoF Kinova Gen3 mounted BASE_Z above the world origin.
Forward kinematics uses the Gen3 modified-DH parameters; the transform from
the FK "interface" frame to the point that must coincide with the cube centre
when the gripper closes was calibrated empirically (GRASP_V, expressed in the
tool frame).  Strategy per episode:

  1. pick a cube, drive the mobile base to a comfortable stand-off pose,
  2. IK a top-down approach (tool z pointing down), move above the cube,
     descend to the grasp pose,
  3. close the gripper, lift.

Blocked motions (the environment rejects colliding actions, leaving the state
unchanged) are detected and trigger the next candidate (different yaw, grasp
offset, or cube).
"""
import time

import numpy as np
from scipy.optimize import least_squares

PI = np.pi
BASE_Z = 0.4
GRASP_A = 0.12          # tool-frame x offset of the grasp point
GRASP_C = 0.22          # tool-frame z offset (depth) of the grasp point
JOINT_NAMES = ["joint_%d" % i for i in range(1, 8)]
HOME = np.array([0.0, -0.35, -PI, -2.5, 0.0, -0.87, PI / 2])
LIM_LO = np.array([-1e9, -2.41, -1e9, -2.66, -1e9, -2.23, -1e9])
LIM_HI = np.array([1e9, 2.41, 1e9, 2.66, 1e9, 2.23, 1e9])
DOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
STAND_OFF = 0.55        # desired base <-> cube horizontal distance
TIME_BUDGET = 45.0      # seconds of wall clock we allow ourselves


def _dh(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1.0]])


_DH = [(PI, 0, -0.2848, 0.0), (PI / 2, 0, -0.0118, PI), (PI / 2, 0, -0.4208, PI),
       (PI / 2, 0, -0.0128, PI), (PI / 2, 0, -0.3143, PI), (PI / 2, 0, 0.0, PI),
       (PI / 2, 0, -0.1674, PI)]
_TOOL = _dh(PI, 0, -0.0615, 0.0)


def fk(q):
    T = np.eye(4)
    for i in range(7):
        alpha, a, d, th = _DH[i]
        T = T @ _dh(alpha, a, d, th + q[i])
    return T @ _TOOL


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _residual(q, pos, R):
    T = fk(q)
    e = np.empty(9)
    e[:3] = T[:3, 3] - pos
    e[3:6] = (T[:3, 0] - R[:, 0]) * 0.4
    e[6:9] = (T[:3, 2] - R[:, 2]) * 0.4
    return e


def ik(pos, R, seeds, tol=1e-4):
    best_q, best_e = None, 1e9
    for s0 in seeds:
        try:
            sol = least_squares(_residual, s0, args=(pos, R), xtol=1e-11,
                                ftol=1e-11, max_nfev=120)
        except Exception:
            continue
        q = sol.x
        if np.any(q < LIM_LO - 1e-6) or np.any(q > LIM_HI + 1e-6):
            continue
        e = float(np.linalg.norm(sol.fun))
        if e < best_e:
            best_q, best_e = q, e
        if e < tol:
            break
    return best_q, best_e


class GeneratedApproach:
    def __init__(self, action_space=None, observation_space=None, primitives=None):
        self.action_space = action_space
        self.dim = 11
        if action_space is not None and getattr(action_space, "shape", None):
            self.dim = int(action_space.shape[0])

    # ------------------------------------------------------------------ state
    def _robot(self, state):
        try:
            for name in state.get_object_names():
                obj = state.get_object_from_name(name)
                if self._type_name(obj) == "Kinematic3DRobot":
                    return obj
        except Exception:
            pass
        return state.get_object_from_name("robot")

    @staticmethod
    def _type_name(obj):
        try:
            return obj.type.name
        except Exception:
            return ""

    def _joints(self, state):
        r = self._robot(state)
        return np.array([float(state.get(r, f)) for f in JOINT_NAMES])

    def _base(self, state):
        r = self._robot(state)
        return np.array([float(state.get(r, "pos_base_x")),
                         float(state.get(r, "pos_base_y")),
                         float(state.get(r, "pos_base_rot"))])

    def _grasped(self, state):
        r = self._robot(state)
        return float(state.get(r, "grasp_active")) > 0.5

    def _cubes(self, state):
        out = []
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            if self._type_name(obj) != "Kinematic3DCuboid":
                if not str(name).startswith("cube"):
                    continue
            he = np.array([float(state.get(obj, "half_extent_%s" % a)) for a in "xyz"])
            if float(np.max(he)) > 0.12:
                continue
            p = np.array([float(state.get(obj, "pose_%s" % a)) for a in "xyz"])
            out.append((name, p, he))
        return out

    def _cube_pos(self, state, name):
        obj = state.get_object_from_name(name)
        return np.array([float(state.get(obj, "pose_%s" % a)) for a in "xyz"])

    def _to_base(self, p, base):
        v = np.array([p[0] - base[0], p[1] - base[1], p[2] - BASE_Z])
        return rotz(-base[2]) @ v

    # ------------------------------------------------------------------ setup
    def reset(self, state, info=None):
        self.t0 = time.time()
        self.t = 0
        self.prev_q = None
        self.prev_base = None
        self.stall = 0
        self.plan = None
        self.stage = 0
        self.grasp_try = 0
        self.lift_target = None

        base = self._base(state)
        cubes = self._cubes(state)
        scored = []
        for name, p, he in cubes:
            pb = self._to_base(p, base)
            dist = float(np.hypot(pb[0], pb[1]))
            near = 1e9
            for n2, p2, _ in cubes:
                if n2 != name:
                    near = min(near, float(np.linalg.norm(p2[:2] - p[:2])))
            score = abs(dist - STAND_OFF) + (0.0 if near > 0.13 else 0.6)
            scored.append((score, name))
        scored.sort()
        self.cube_order = [n for _, n in scored]
        self.cube_i = 0
        self.wave = 0
        self.attempts = None
        self.phase = "base"
        self._start_cube(state)

    def _base_target(self, cube_world):
        bx = float(np.clip(cube_world[0] - STAND_OFF, -0.45, 0.15))
        by = float(np.clip(cube_world[1], -0.6, 0.6))
        return np.array([bx, by])

    def _start_cube(self, state):
        """Begin working on cube_order[cube_i]: move base, then plan."""
        if self.cube_i >= len(self.cube_order):
            # start another wave over all cubes with different grasp params
            if self.wave >= 4 or time.time() - self.t0 > TIME_BUDGET:
                self.phase = "done"
                return
            self.wave += 1
            self.cube_i = 0
        self.cube_name = self.cube_order[self.cube_i]
        cw = self._cube_pos(state, self.cube_name)
        self.base_goal = self._base_target(cw)
        self.attempts = self._attempt_list(self.wave)
        self.attempt_i = 0
        self.phase = "base"
        self.base_steps = 0

    def _attempt_list(self, wave=0):
        """Grasp attempts for one cube: (yaw, delta_a, delta_c)."""
        if wave == 0:
            yaws = (0.0, 0.3, -0.3, 0.6)
            offs = ((0.0, 0.0), (0.01, 0.0), (-0.01, 0.0))
        elif wave == 1:
            yaws = (-0.6, 0.15, -0.15, 0.9, -0.9, 0.45, -0.45)
            offs = ((0.0, 0.015), (0.0, -0.01), (0.01, 0.015), (-0.01, 0.015),
                    (0.02, 0.0), (-0.02, 0.0))
        else:
            rng = np.random.default_rng(1000 + wave)
            return [(float(rng.uniform(-1.2, 1.2)),
                     float(rng.uniform(-0.02, 0.02)),
                     float(rng.uniform(-0.015, 0.02))) for _ in range(40)]
        return [(y, da, dc) for y in yaws for da, dc in offs]

    def _plan(self, state):
        """Build joint waypoints for the current attempt; None if impossible."""
        base = self._base(state)
        q_now = self._joints(state)
        pb = self._to_base(self._cube_pos(state, self.cube_name), base)
        rng = np.random.default_rng(7)
        while self.attempt_i < len(self.attempts):
            if time.time() - self.t0 > TIME_BUDGET:
                return None
            yaw, da, dc = self.attempts[self.attempt_i]
            self.attempt_i += 1
            R = rotz(yaw) @ DOWN
            v = np.array([GRASP_A + da, 0.0, GRASP_C + dc])
            gpos = pb - R @ v
            seeds = [q_now, HOME] + [HOME + rng.normal(0, 1.0, 7) for _ in range(4)]
            qg, eg = ik(gpos, R, seeds)
            if qg is None or eg > 3e-3:
                continue
            way = []
            if self.attempt_i % 3 == 0 and float(np.max(np.abs(q_now - HOME))) > 0.2:
                way.append(HOME.copy())   # retract before re-approaching
            for dz in (0.28, 0.12):
                qw, ew = ik(gpos + np.array([0.0, 0.0, dz]), R, [qg, q_now, HOME])
                if qw is not None and ew < 5e-3:
                    way.append(qw)
            way.append(qg)
            self.grasp_R = R
            self.grasp_pos = gpos
            self.n_pre = len(way)
            # lift waypoints
            for dz in (0.10, 0.25):
                qw, ew = ik(gpos + np.array([0.0, 0.0, dz]), R, [qg, way[0]])
                if qw is not None and ew < 1e-2:
                    way.append(qw)
            self.stage = 0
            self.grasp_try = 0
            return way
        return None

    # ---------------------------------------------------------------- actions
    def _zero(self):
        return np.zeros(self.dim, dtype=np.float32)

    def get_action(self, state):
        if not hasattr(self, "phase"):
            self.reset(state, None)
        try:
            return self._get_action(state)
        except Exception:
            a = self._zero()
            a[10] = -1.0 if getattr(self, "_closing", False) else 0.0
            return a

    def _get_action(self, state):
        self.t += 1
        q = self._joints(state)
        base = self._base(state)
        grasped = self._grasped(state)

        if grasped:
            self._closing = True
            return self._lift(state, q)

        if self.phase == "done":
            return self._zero()

        if self.phase == "base":
            a = self._zero()
            d = self.base_goal - base[:2]
            self.base_steps += 1
            moved = self.prev_base is None or not np.allclose(base[:2], self.prev_base, atol=1e-9)
            self.prev_base = base[:2].copy()
            if np.max(np.abs(d)) > 0.01 and self.base_steps < 12 and (moved or self.base_steps < 3):
                a[0] = float(np.clip(d[0], -0.35, 0.35))
                a[1] = float(np.clip(d[1], -0.35, 0.35))
                if not moved:
                    # blocked: try moving only in y, then only in x
                    if abs(d[1]) > 0.01:
                        a[0] = 0.0
                    else:
                        a[1] = 0.0
                return a
            self.phase = "arm"
            self.plan = self._plan(state)
            if self.plan is None:
                self.cube_i += 1
                self._start_cube(state)
            return self._zero()

        # phase == "arm"
        if self.plan is None:
            self.cube_i += 1
            self._start_cube(state)
            return self._zero()

        if self.stage >= self.n_pre:
            # at the grasp pose: close
            a = self._zero()
            a[10] = -1.0
            self.grasp_try += 1
            if self.grasp_try > 2:
                self._retry(state)
            return a

        qt = self.plan[self.stage]
        d = qt - q
        if np.max(np.abs(d)) < 4e-3:
            self.stage += 1
            self.stall = 0
            self.prev_q = None
            return self._zero()
        a = self._zero()
        a[3:10] = np.clip(d, -0.3, 0.3)
        if self.prev_q is not None and np.allclose(q, self.prev_q, atol=1e-9):
            self.stall += 1
            if self.stall > 3:
                self.stall = 0
                self._retry(state)
                return self._zero()
        else:
            self.stall = 0
        self.prev_q = q.copy()
        return a

    def _retry(self, state):
        """Current attempt failed: move on to the next one (or next cube)."""
        self.prev_q = None
        self.stall = 0
        if time.time() - self.t0 > TIME_BUDGET:
            self.plan = None
            self.phase = "done"
            return
        self.plan = self._plan(state)
        if self.plan is None:
            self.cube_i += 1
            self._start_cube(state)

    def _lift(self, state, q):
        """Holding a cube: raise it well above the table."""
        a = self._zero()
        a[10] = -1.0
        if self.plan is not None and self.stage < len(self.plan) - 1:
            self.stage = max(self.stage, self.n_pre)
        if self.plan is not None and self.stage < len(self.plan):
            qt = self.plan[self.stage]
            d = qt - q
            if np.max(np.abs(d)) < 5e-3:
                self.stage += 1
                return a
            a[3:10] = np.clip(d, -0.25, 0.25)
            if self.prev_q is not None and np.allclose(q, self.prev_q, atol=1e-9):
                self.stall += 1
                if self.stall > 3:
                    self.stage += 1
                    self.stall = 0
            else:
                self.stall = 0
            self.prev_q = q.copy()
            return a
        # fallback: lift straight up in Cartesian space
        base = self._base(state)
        T = fk(q)
        pos = T[:3, 3] + np.array([0.0, 0.0, 0.06])
        qt, e = ik(pos, T[:3, :3], [q])
        if qt is not None and e < 1e-2:
            a[3:10] = np.clip(qt - q, -0.2, 0.2)
        return a
