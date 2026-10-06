"""Obstruction3D approach: sim-verified pick-and-place with IK.

Strategy
--------
1. Remove every obstruction that sits on / near the target region by picking
   it and placing it on a free table spot.
2. Pick the target block and place it on the target region center.

Each pick/place "segment" is planned with Cartesian IK waypoints (top-down
gripper), converted into joint-delta actions, and verified by simulating it in
an internal copy of the environment (exact same kinematic dynamics). Several
alternative parameters (gripper yaw, placement spot, grasp height) are tried
until a segment verifies. The verified actions are then replayed in the real
env; on divergence we replan.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

TABLE_TOP = 0.075
TABLE_X = (0.1, 0.5)
TABLE_Y = (-0.4, 0.4)
MAX_D = 0.2
COARSE_STEP = 0.02
BASE_X = -0.2
CONT_JOINTS = np.array([True, False, True, False, True, False, True])
R0 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

_SIM_CACHE: dict[int, Any] = {}
import os
DEBUG = bool(os.environ.get("OBS_DEBUG"))


def _log(*a):
    if DEBUG:
        print("[dbg]", *a, flush=True)


def _get_sim(n: int):
    if n not in _SIM_CACHE:
        from kinder.envs.kinematic3d.obstruction3d import (
            ObjectCentricObstruction3DEnv,
        )

        sim = ObjectCentricObstruction3DEnv(
            num_obstructions=n, allow_state_access=True
        )
        sim.reset(seed=0)
        _SIM_CACHE[n] = sim
    return _SIM_CACHE[n]


def _coerce(state, sim):
    """Rebuild the state as the sim's concrete state class."""
    from relational_structs import Object

    state_cls = getattr(sim, "state_cls")
    feats = sim.type_features
    if isinstance(state, state_cls) and state.type_features is feats:
        return state
    types = {t.name: t for t in feats}
    data = {}
    for obj in state:
        t = types[obj.type.name]
        data[Object(obj.name, t)] = np.asarray(
            [state.get(obj, f) for f in feats[t]], dtype=np.float32
        )
    return state_cls(data, feats)


def _yaw_quat(yaw: float) -> tuple:
    rot = R.from_euler("z", yaw).as_matrix() @ R0
    return tuple(R.from_matrix(rot).as_quat())


class _Info:
    """Lightweight snapshot of the object-centric state."""

    def __init__(self, state):
        self.objs: dict[str, dict[str, float]] = {}
        for obj in state:
            self.objs[obj.name] = {
                f: float(state.get(obj, f)) for f in state.type_features[obj.type]
            }

    def pos(self, name):
        o = self.objs[name]
        return np.array([o["pose_x"], o["pose_y"], o["pose_z"]])

    def he(self, name):
        o = self.objs[name]
        return np.array([o["half_extent_x"], o["half_extent_y"], o["half_extent_z"]])

    def joints(self):
        r = self.objs["robot"]
        return np.array([r[f"joint_{i}"] for i in range(1, 8)])

    def held(self):
        r = self.objs["robot"]
        if r["grasp_active"] < 0.5:
            return None
        for n, o in self.objs.items():
            if n != "robot" and o.get("grasp_active", 0.0) > 0.5:
                return n
        return "?"

    def obstructions(self):
        return sorted(n for n in self.objs if n.startswith("obstruction"))


def _rect_gap(c1, h1, c2, h2):
    """Max-axis separation between two axis-aligned rectangles (neg=overlap)."""
    dx = abs(c1[0] - c2[0]) - h1[0] - h2[0]
    dy = abs(c1[1] - c2[1]) - h1[1] - h2[1]
    return max(dx, dy)


class Planner:
    def __init__(self, n: int, deadline: float):
        self.sim = _get_sim(n)
        self.arm = self.sim.robot.arm
        self.deadline = deadline

    # ------------------------------------------------------------------ sim
    def set(self, state):
        self.sim.set_state(state)

    def step(self, action):
        obs, _, term, _, _ = self.sim.step(np.asarray(action, dtype=np.float32))
        return obs, term

    def ee_pose(self):
        p = self.arm.get_end_effector_pose()
        return np.array(p.position), np.array(p.orientation)

    def ik(self, pos, quat, seed, tol=1e-3):
        from pybullet_helpers.geometry import Pose
        from pybullet_helpers.inverse_kinematics import (
            InverseKinematicsError,
            InverseKinematicsHyperparameters,
            pybullet_inverse_kinematics,
        )

        cur = list(self.arm.get_joint_positions())
        self.arm.set_joints(list(seed) + cur[7:])
        try:
            sol = pybullet_inverse_kinematics(
                self.arm, Pose(tuple(pos), tuple(quat)), validate=True,
                hyperparameters=InverseKinematicsHyperparameters(
                    convergence_atol=tol, max_iters=100 if tol >= 1e-3 else 300),
            )
        except InverseKinematicsError:
            return None
        return np.array(sol[:7])

    # ------------------------------------------------------------ motions
    def move_actions(self, state, target_pos, quat, step_len=0.02, merge=3):
        """Joint-delta actions moving the EE along a straight line to target."""
        self.set(state)
        start_pos, start_quat = self.ee_pose()
        q = _Info(state).joints()
        dist = np.linalg.norm(np.asarray(target_pos) - start_pos)
        n = max(1, int(math.ceil(dist / step_len)))
        rots = R.from_quat([start_quat, quat])
        from scipy.spatial.transform import Slerp

        slerp = Slerp([0, 1], rots)
        waypoints = []
        seed = q.copy()
        for i in range(1, n + 1):
            t = i / n
            pos = start_pos + t * (np.asarray(target_pos) - start_pos)
            qt = slerp([t]).as_quat()[0]
            sol = self.ik(pos, qt, seed, 1e-3 if step_len >= 0.01 else 2e-5)
            if sol is None:
                return None
            # Unwrap continuous joints to stay close to the seed.
            d = sol - seed
            d[CONT_JOINTS] = (d[CONT_JOINTS] + np.pi) % (2 * np.pi) - np.pi
            sol = seed + d
            if np.max(np.abs(d)) > 1.0:
                return None  # IK jumped to a different branch
            waypoints.append(sol)
            seed = sol
        # Convert waypoints into actions with max delta MAX_D.
        actions = []
        cur = q.copy()
        idx = 0
        while idx < len(waypoints):
            # furthest waypoint reachable within MAX_D from cur
            j = idx
            while j + 1 < len(waypoints) and np.max(
                np.abs(waypoints[j + 1] - cur)
            ) <= MAX_D - 1e-6 and j - idx < merge:
                j += 1
            tgt = waypoints[j]
            d = tgt - cur
            m = np.max(np.abs(d))
            if m > MAX_D - 1e-6:
                d = d * (MAX_D - 1e-6) / m
            else:
                idx = j + 1
            a = np.zeros(11)
            a[3:10] = d
            actions.append(a)
            cur = cur + d
        return actions

    def run(self, state, actions):
        """Simulate actions from state; return final state or None on revert."""
        self.set(state)
        s = state
        q = _Info(state).joints()
        for a in actions:
            s, _ = self.step(a)
            q = q + a[3:10]
            qs = _Info(s).joints()
            if np.max(np.abs(qs - q)) > 1e-3 and np.any(a[3:10] != 0):
                return None
            q = qs
        return s

    def gripper(self, state, close: bool):
        self.set(state)
        a = np.zeros(11)
        a[10] = -1.0 if close else 1.0
        s, term = self.step(a)
        return a, s, term

    def go(self, state, pos, quat, plan, step_len=0.02, merge=3, coarse=False):
        acts = self.move_actions(state, pos, quat, step_len, merge)
        if acts is None:
            return None
        cfg = self.sim.config
        fine = cfg.max_collision_check_step
        if coarse:
            object.__setattr__(cfg, "max_collision_check_step", COARSE_STEP)
        try:
            s = self.run(state, acts)
        finally:
            object.__setattr__(cfg, "max_collision_check_step", fine)
        if s is None:
            return None
        plan.extend(acts)
        return s

    # ------------------------------------------------------------ skills
    def with_joints(self, state, q):
        s = state.copy()
        robot = s.get_object_from_name("robot")
        for i in range(7):
            s.set(robot, f"joint_{i + 1}", float(q[i]))
        return s

    def lift_dist(self, info, name):
        c = info.pos(name)
        he = info.he(name)
        bottom = c[2] - he[2]
        d = 0.03
        for n in info.objs:
            if n in ("robot", "target_region", name):
                continue
            if _rect_gap(c, he, info.pos(n), info.he(n)) < 0.02:
                d = max(d, info.pos(n)[2] + info.he(n)[2] - bottom + 0.008)
        return d

    def precheck_pick(self, state, name, yaw, z_off, off=(0.0, 0.0)):
        info = _Info(state)
        c = info.pos(name).copy()
        c[:2] += np.asarray(off)
        he = info.he(name)
        top = c[2] + he[2]
        quat = _yaw_quat(yaw)
        hi = max(0.25, top + 0.08)
        self.set(state)
        q = self.ik([c[0], c[1], hi], quat, info.joints())
        if q is None:
            return False
        fake = self.with_joints(state, q)
        plan: list = []
        s = self.go(fake, [c[0], c[1], top + z_off], quat, plan)
        if s is None:
            return False
        _, s, _ = self.gripper(s, True)
        if _Info(s).held() != name:
            return False
        self.set(s)
        ee_now, _ = self.ee_pose()
        s = self.go(s, [ee_now[0], ee_now[1], ee_now[2] + self.lift_dist(info, name)], quat, plan, 0.003, 0)
        return s is not None

    def pick(self, state, name, yaw, z_off=0.0, off=(0.0, 0.0)):
        """Returns (plan, state) after grasp and lift, or None."""
        if not self.precheck_pick(state, name, yaw, z_off, off):
            _log("precheck fail")
            return None
        info = _Info(state)
        c = info.pos(name).copy()
        c[:2] += np.asarray(off)
        he = info.he(name)
        top = c[2] + he[2]
        quat = _yaw_quat(yaw)
        ee_z = top + z_off
        hi = max(0.25, top + 0.08)
        plan: list = []
        s = state
        cur_ee, _ = (self.set(s), self.ee_pose())[1]
        if info.held() is None and cur_ee[2] < hi - 0.03:
            # lift first
            s = self.go(s, [cur_ee[0], cur_ee[1], hi], quat, plan)
            if s is None:
                _log('pickfail@260'); return None
        s = self.go(s, [c[0], c[1], hi], quat, plan, coarse=True)
        if s is None:
            _log('pickfail@263'); return None
        s = self.go(s, [c[0], c[1], ee_z], quat, plan)
        if s is None:
            _log('pickfail@266'); return None
        a, s, _ = self.gripper(s, True)
        plan.append(a)
        if _Info(s).held() != name:
            _log('pickfail@270'); return None
        self.set(s)
        ee_now, _ = self.ee_pose()
        s = self.go(s, [ee_now[0], ee_now[1], ee_now[2] + self.lift_dist(info, name)], quat, plan, 0.003, 0)
        if s is None:
            _log('pickfail@lift1'); return None
        s = self.go(s, [c[0], c[1], hi], quat, plan)
        if s is None:
            _log('pickfail@273'); return None
        return plan, s

    def place(self, state, name, xy, support_top, yaw):
        """Place the held object with its center at xy on support. Returns
        (plan, state, terminated)."""
        info = _Info(state)
        self.set(state)
        ee, _ = self.ee_pose()
        c = info.pos(name)
        he = info.he(name)
        off = ee - c  # EE relative to object center (constant while held)
        quat = _yaw_quat(yaw)
        plan: list = []
        hi_z = ee[2]
        tgt = np.array([xy[0], xy[1], support_top + he[2] + 0.002]) + off
        s = self.go(state, [tgt[0], tgt[1], hi_z], quat, plan, coarse=True)
        if s is None:
            _log("placefail@transfer")
            return None
        s = self.go(s, tgt, quat, plan)
        if s is None:
            _log("placefail@descend")
            return None
        a, s, term = self.gripper(s, False)
        plan.append(a)
        if _Info(s).held() is not None:
            return None
        if not term:
            # retreat upward
            self.set(s)
            ee2, _ = self.ee_pose()
            s2 = self.go(s, [ee2[0], ee2[1], hi_z], quat, plan)
            if s2 is not None:
                s = s2
        return plan, s, term

    # ------------------------------------------------------------ helpers
    def yaws_for(self, info, name):
        he = info.he(name)
        # Fingers close along world-y for yaw=0, along world-x for yaw=pi/2.
        if he[1] <= he[0]:
            return [0.0, np.pi / 2]
        return [np.pi / 2, 0.0]

    def grasp_test(self, state, name, yaw, z_off, off):
        """Cheap test: teleport to the grasp pose and close the gripper."""
        info = _Info(state)
        c = info.pos(name)
        he = info.he(name)
        top = c[2] + he[2]
        quat = _yaw_quat(yaw)
        gx, gy = c[0] + off[0], c[1] + off[1]
        self.set(state)
        q = self.ik([gx, gy, max(0.25, top + 0.08)], quat, info.joints())
        if q is None:
            return False
        q = self.ik([gx, gy, top + z_off], quat, q)
        if q is None:
            return False
        fake = self.with_joints(state, q)
        self.set(fake)
        if self.sim._robot_or_held_object_collision_exists():
            return False
        _, s, _ = self.gripper(fake, True)
        return _Info(s).held() == name

    def extended_grasp_options(self, state, name):
        info = _Info(state)
        he = info.he(name)
        yaws = [k * np.pi / 8 for k in range(-3, 5)]
        offs = []
        for fx in (-1, 0, 1):
            for fy in (-1, 0, 1):
                if fx or fy:
                    offs.append((fx * (he[0] + 0.006), fy * (he[1] + 0.006)))
        for fx in (-1, 0, 1):
            for fy in (-1, 0, 1):
                if fx or fy:
                    offs.append((fx * he[0] * 0.5, fy * he[1] * 0.5))
        offs.insert(0, (0.0, 0.0))
        out = []
        for z in (0.032, 0.026):
            for off in offs:
                for yaw in yaws:
                    if time.time() > self.deadline:
                        return out
                    if self.grasp_test(state, name, yaw, z, off):
                        out.append((yaw, z, off))
                        if len(out) >= 4:
                            return out
        return out

    def all_grasp_options(self, state, name):
        info = _Info(state)
        for yaw, z in self.grasp_options(info, name):
            yield yaw, z, (0.0, 0.0)
        _log("extended grasp search", name)
        for o in self.extended_grasp_options(state, name):
            yield o

    def grasp_options(self, info, name):
        """(yaw, z offset of EE above object top) candidates, best first."""
        he = info.he(name)
        support = info.pos(name)[2] - he[2]
        top = info.pos(name)[2] + he[2]
        low = max(support + 0.05 - top, -0.01)
        yaws = self.yaws_for(info, name)
        opts = []
        for z in (0.032, 0.02, low):
            for y in yaws:
                opts.append((y, z))
        for z in (0.032, low):
            for y in (np.pi / 4, -np.pi / 4):
                opts.append((y, z))
        return opts

    def free_spots(self, info, name, avoid_names):
        he = info.he(name)
        rc = info.pos("target_region")
        rh = info.he("target_region")
        cur = info.pos(name)
        spots = []
        for x in np.arange(0.13, 0.48, 0.02):
            for y in np.arange(-0.37, 0.371, 0.02):
                if x - he[0] < TABLE_X[0] + 0.01 or x + he[0] > TABLE_X[1] - 0.01:
                    continue
                if y - he[1] < TABLE_Y[0] + 0.01 or y + he[1] > TABLE_Y[1] - 0.01:
                    continue
                p = np.array([x, y])
                if _rect_gap(p, he, rc, rh) < 0.07:
                    continue
                ok = True
                for o in avoid_names:
                    if o == name:
                        continue
                    if _rect_gap(p, he, info.pos(o), info.he(o)) < 0.06:
                        ok = False
                        break
                if not ok:
                    continue
                spots.append((np.linalg.norm(p - cur[:2]) + 0.3 * abs(y), p))
        spots.sort(key=lambda t: t[0] + (0.3 if t[1][0] < 0.18 else 0.0))
        chosen: list = []
        for _, p in spots:
            if all(np.linalg.norm(p - q) >= 0.05 for q in chosen):
                chosen.append(p)
        return chosen

    def move_obstruction(self, state, name):
        info = _Info(state)
        avoid = [n for n in info.objs if n not in ("robot", "target_region")]
        spots = self.free_spots(info, name, avoid)
        for yaw, z_off, goff in self.all_grasp_options(state, name):
            if True:
                if time.time() > self.deadline:
                    return None
                r = self.pick(state, name, yaw, z_off, goff)
                _log("pick", name, yaw, z_off, r is not None)
                if r is None:
                    continue
                plan, s = r
                for spot in spots[:6]:
                    if time.time() > self.deadline:
                        return None
                    r2 = self.place(s, name, spot, TABLE_TOP, yaw)
                    _log("place", name, spot, r2 is not None)
                    if r2 is None:
                        continue
                    plan2, s2, _ = r2
                    return plan + plan2, s2
        return None

    def place_target(self, state):
        info = _Info(state)
        rc = info.pos("target_region")
        rh = info.he("target_region")
        offsets = [(0.0, 0.0)]
        sl = 0.2 * rh[:2] * 0.9
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    offsets.append((dx * sl[0], dy * sl[1]))
        for yaw, z_off, goff in self.all_grasp_options(state, "target_block"):
            if True:
                if time.time() > self.deadline:
                    return None
                r = self.pick(state, "target_block", yaw, z_off, goff)
                _log("pick target", yaw, z_off, r is not None)
                if r is None:
                    continue
                plan, s = r
                for off in offsets:
                    if time.time() > self.deadline:
                        return None
                    xy = rc[:2] + np.array(off)
                    r2 = self.place(s, "target_block", xy, rc[2] + rh[2], yaw)
                    _log("place target", off, None if r2 is None else r2[2])
                    if r2 is None:
                        continue
                    plan2, s2, term = r2
                    if term:
                        return plan + plan2, s2
        return None


def _obstructs(info, name, margin=0.0):
    rc = info.pos("target_region")
    rh = info.he("target_region")
    return _rect_gap(info.pos(name), info.he(name), rc, rh) < margin


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.plan: list = []
        self.expected = None

    def reset(self, state, info):
        self.plan = []
        self.t0 = time.time()
        self.n = len([n for n in _Info(state).objs if n.startswith("obstruction")])
        self.planner = None
        try:
            self.planner = Planner(self.n, self.t0 + 50.0)
        except Exception:  # pragma: no cover
            self.planner = None
        self.failed: set = set()
        self.fail_cache: dict = {}
        self.base_moved = False

    def _sim_state(self, state):
        return _coerce(state, self.planner.sim)

    def _sig(self, info, name):
        c = info.pos(name)
        items = []
        for n in sorted(info.objs):
            if n in ("robot", "target_region"):
                continue
            p = info.pos(n)
            if np.linalg.norm(p[:2] - c[:2]) < 0.15:
                items.append((n, round(p[0], 3), round(p[1], 3), round(p[2], 3)))
        return tuple(items)

    def _try_move(self, s, info, name):
        sig = self._sig(info, name)
        if self.fail_cache.get(name) == sig:
            _log("skip cached failure", name)
            return None
        r = self.planner.move_obstruction(s, name)
        if r is None and time.time() < self.planner.deadline:
            self.fail_cache[name] = sig
        return r

    def _make_plan(self, state):
        pl = self.planner
        pl.deadline = self.t0 + 52.0
        s = self._sim_state(state)
        info = _Info(s)
        held = info.held()
        if held is not None:
            # Something is held: place it somewhere sensible.
            if held == "target_block":
                r = pl.place(s, held, info.pos("target_region")[:2],
                             info.pos("target_region")[2] + info.he("target_region")[2], 0.0)
                if r is not None:
                    return r[0]
            avoid = [n for n in info.objs if n not in ("robot", "target_region")]
            for spot in pl.free_spots(info, held, avoid)[:6]:
                for yaw in (0.0, np.pi / 2):
                    r = pl.place(s, held, spot, TABLE_TOP, yaw)
                    if r is not None:
                        return r[0]
            return None
        # Obstructions to clear, closest to region first.
        obs = [o for o in info.obstructions() if _obstructs(info, o, 0.005)]
        def crowd(o):
            cnt = 0.0
            for n in info.objs:
                if n in ("robot", "target_region", o):
                    continue
                g = _rect_gap(info.pos(o), info.he(o), info.pos(n), info.he(n))
                if g < 0.03 and info.pos(n)[2] + info.he(n)[2] > info.pos(o)[2] + info.he(o)[2] - 0.012:
                    cnt += 1.0
            return cnt - 0.01 * (info.pos(o)[2] + info.he(o)[2])
        obs.sort(key=crowd)
        for o in obs:
            r = self._try_move(s, info, o)
            if r is not None:
                return r[0]
        if obs:
            # None of the on-region obstructions could be moved: relocate a
            # neighbouring object (off-region obstruction or the target block).
            others = [n for n in info.obstructions() if n not in obs]
            others.append("target_block")
            def gap(n):
                return min(_rect_gap(info.pos(n), info.he(n), info.pos(o), info.he(o))
                           for o in obs)
            others.sort(key=gap)
            for n in others[:3]:
                if gap(n) > 0.06:
                    break
                r = self._try_move(s, info, n)
                if r is not None:
                    return r[0]
        if not obs:
            r = pl.place_target(s)
            if r is not None:
                return r[0]
            # Could not place target: clear nearby obstructions too.
            near = [o for o in info.obstructions() if _obstructs(info, o, 0.08)]
            rc = info.pos("target_region")
            near.sort(key=lambda o: np.linalg.norm(info.pos(o)[:2] - rc[:2]))
            for o in near:
                r = self._try_move(s, info, o)
                if r is not None:
                    return r[0]
            # Target block maybe hard to grasp due to neighbors.
            tb = info.pos("target_block")
            near = sorted(info.obstructions(),
                          key=lambda o: np.linalg.norm(info.pos(o)[:2] - tb[:2]))
            for o in near[:2]:
                r = self._try_move(s, info, o)
                if r is not None:
                    return r[0]
        return None

    def get_action(self, state):
        if self.plan and self.expected is not None:
            info = _Info(state)
            q_exp, held_exp = self.expected
            d = info.joints() - q_exp
            d[CONT_JOINTS] = (d[CONT_JOINTS] + np.pi) % (2 * np.pi) - np.pi
            if np.max(np.abs(d)) > 2e-3 or (
                held_exp is not None and info.held() != held_exp
            ):
                _log("divergence -> replan")
                self.plan = []
        self.expected = None
        if not self.base_moved:
            self.base_moved = True
            bx = _Info(state).objs["robot"]["pos_base_x"]
            if abs(bx - BASE_X) > 1e-3:
                a = np.zeros(11, dtype=np.float32)
                a[0] = np.clip(BASE_X - bx, -MAX_D, MAX_D)
                return a
        if not self.plan and self.planner is not None:
            try:
                p = self._make_plan(state)
            except Exception:
                p = None
            if p:
                self.plan = list(p)
        if self.plan:
            a = np.asarray(self.plan.pop(0), dtype=np.float32)
            info = _Info(state)
            held_exp = None
            if a[10] < -0.5:
                held_exp = "__any__"
            self.expected = (info.joints() + a[3:10], None)
            return a
        return np.zeros(11, dtype=np.float32)
