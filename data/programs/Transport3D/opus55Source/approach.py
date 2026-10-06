"""Scripted planner for Transport3D (variable number of cubes + one box).

At reset we build an internal copy of the environment (same dynamics), sync it
to the observed state, and plan the full action sequence by simulating it:
  1. pick the box (by the wall nearest to the robot) and put it on a table corner
  2. for each cube: pick it from the floor and place it on a free table slot
     (stacking on top of already placed cubes if the table runs out).
Each primitive is verified in simulation; on failure we roll back and try
alternative approach directions / slots.
"""

import dataclasses
import logging
import math
import time

import numpy as np

logging.disable(logging.WARNING)

TABLE_LO = np.array([0.4, -0.4])
TABLE_HI = np.array([0.8, 0.4])
TABLE_TOP = 0.4
NAV_MARGIN = 0.3
UNBOUNDED = (0, 2, 4, 6)
MAXD = 0.2
FAST = True
PRE_H = 0.08
PRE_FAST = 0.02
BOX_LIFT = False
MERGE_GRIPPER = True
EVAL_M = 3
NOGATE = bool(int(__import__('os').environ.get('NOGATE', '0')))
LOOKAHEAD = bool(int(__import__('os').environ.get('LOOK', '0')))
_AL = [float(v) for v in __import__('os').environ.get('ALPHAS', '0,0.785,-0.785').split(',')]
ALPHAS = tuple(_AL)
ALPHA_PEN = float(__import__('os').environ.get('ALPHA_PEN', '0.2'))
ARM_COST = bool(int(__import__('os').environ.get('ARM_COST', '1')))
CARRY_Z = float(__import__('os').environ.get('CARRY_Z', '0.8'))
LOOK_N = 30
RET_D = 0.5
RETREAT = 0.0
SLOT_PITCH = 0.1
PITCHES = [(0.1, 0.1), (0.1, 0.085), (0.1, 0.075), (0.085, 0.075), (0.075, 0.075)]
NDIR = 16
IK_SEEDS = (
    (0.0, -0.35, -math.pi, -2.5, 0.0, -0.87, math.pi / 2),
    (0.0, 0.5, -math.pi, -1.5, 0.0, -1.0, math.pi / 2),
    (0.0, 0.0, 0.0, -1.5, 0.0, -1.0, math.pi / 2),
)
PLACE_DS = (0.45, 0.5, 0.55, 0.6, 0.65, 0.68)
PLACE_ALL_D = False
PICK_DS = (0.5, 0.42, 0.58, 0.65, 0.7)
PICK_DPEN = 2.0


def _quat_down(yaw):
    from scipy.spatial.transform import Rotation as R

    return tuple((R.from_euler("z", yaw) * R.from_euler("x", math.pi)).as_quat())


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _seg_hits_rect(p, q, lo, hi):
    """Does segment p->q pass through the open rectangle (lo, hi)?"""
    t0, t1 = 0.0, 1.0
    d = q - p
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if p[i] <= lo[i] or p[i] >= hi[i]:
                return False
        else:
            a = (lo[i] - p[i]) / d[i]
            b = (hi[i] - p[i]) / d[i]
            if a > b:
                a, b = b, a
            t0 = max(t0, a)
            t1 = min(t1, b)
            if t0 >= t1 - 1e-9:
                return False
    return True


def he_z(sim, name):
    """Height of the bottom of object `name` below the grasp point offset."""
    return sim.obj_pos(name)[2] - sim.half_extents(name)[2]


class PlanFail(Exception):
    pass


class Sim:
    def __init__(self, k):
        from kinder.envs.kinematic3d.transport3d import (
            ObjectCentricTransport3DEnv,
            Transport3DEnvConfig,
        )

        try:
            cfg = dataclasses.replace(Transport3DEnvConfig(), realistic_bg=False)
            self.env = ObjectCentricTransport3DEnv(
                num_cubes=k, num_boxes=1, config=cfg, allow_state_access=True
            )
        except Exception:  # pragma: no cover
            self.env = ObjectCentricTransport3DEnv(
                num_cubes=k, num_boxes=1, allow_state_access=True
            )
        self.env.reset(seed=0)
        self.arm = self.env.robot.arm
        self.s = None
        self._bj_cache = (None, None, None)
        self.eval_m = 1
        self.t_deadline = 1e18
        self.alphas = ALPHAS
        self.acts = []
        self.done = False
        self.log = []
        self._cache = {}

    # ------------------------------------------------------------ basics
    def load(self, state):
        self.env.set_state(state)
        self.s = state
        self.acts = []
        self.done = False
        self.log = []
        self._cache = {}

    def checkpoint(self):
        return (self.env.get_state(), len(self.acts), self.done)

    def snapshot(self, extra=None):
        return (self.env.get_state(), list(self.acts), self.done, extra)

    def restore_snapshot(self, snap):
        st, acts, done, extra = snap
        self.env.set_state(st)
        self.s = st
        self.acts[:] = acts
        self.done = done
        return extra

    def restore(self, cp):
        st, n, done = cp
        self.env.set_state(st)
        self.s = st
        del self.acts[n:]
        self.done = done

    def step(self, a):
        a = np.asarray(a, dtype=np.float32)
        s2, _, term, _, _ = self.env.step(a)
        self.acts.append(a)
        self.s = s2
        if term:
            self.done = True
        return s2

    def base(self):
        c = self._bj_cache
        if c[0] is not self.s:
            self._fill_bj()
        return self._bj_cache[1].copy()

    def joints(self):
        c = self._bj_cache
        if c[0] is not self.s:
            self._fill_bj()
        return self._bj_cache[2].copy()

    def _fill_bj(self):
        b = self.s.base_pose
        self._bj_cache = (self.s, np.array([b.x, b.y, b.rot]),
                          np.array(self.s.joint_positions[:7], dtype=float))

    def ee(self):
        return np.array(self.arm.get_end_effector_pose().position)

    def obj_pos(self, name):
        return np.array(self.s.get_object_pose(name).position)

    def all_up(self):
        for n in self.s.get_object_names():
            if n.startswith("cube") or n.startswith("box"):
                if self.obj_pos(n)[2] < 0.3:
                    return False
        return True

    def can_escape(self, yaw):
        """Check (without committing) that the arm can lift away after a place."""
        cp = self.checkpoint()
        e = self.ee()
        try:
            for dz in (0.07, 0.04):
                if self.move_ee(e + [0, 0, dz], yaw):
                    return True
                self.restore(cp)
            b = self.base()
            a = np.zeros(11, dtype=np.float32)
            a[0] = -0.05 * math.cos(b[2])
            a[1] = -0.05 * math.sin(b[2])
            self.step(a)
            return np.abs(self.base() - b).max() > 1e-9
        except PlanFail:
            return False
        finally:
            self.restore(cp)

    def backoff(self, yaw):
        """Final retreat: drive the base backwards while lifting the EE."""
        cp = self.checkpoint()
        e = self.ee()
        jl = self.ik(e + [0, 0, 0.15], yaw)
        for _ in range(3):
            b, cj = self.base(), self.joints()
            a = np.zeros(11, dtype=np.float32)
            a[0] = -MAXD * math.cos(b[2])
            a[1] = -MAXD * math.sin(b[2])
            if jl is not None:
                jd = self._jdiff(jl, cj)
                m = np.abs(jd).max()
                if m > MAXD:
                    jd = jd * (MAXD / m)
                a[3:10] = jd
            self.step(a)
            if self.done:
                return True
            if np.abs(self.base() - b).max() < 1e-9:
                break
        self.restore(cp)
        return False

    def gripper(self, close):
        a = np.zeros(11, dtype=np.float32)
        a[10] = -1.0 if close else 1.0
        self.step(a)

    # ------------------------------------------------------------ motion
    def move_joints(self, target, maxit=80):
        target = np.asarray(target, dtype=float)
        for _ in range(maxit):
            cur = self.joints()
            d = target - cur
            for i in UNBOUNDED:
                d[i] = _wrap(d[i])
            m = np.abs(d).max()
            if m < 1e-6:
                return True
            if m > MAXD:
                d = d * (MAXD / m)
            a = np.zeros(11, dtype=np.float32)
            a[3:10] = d
            self.step(a)
            if np.abs(self.joints() - cur).max() < 1e-9:
                return False
        return False

    def move_base(self, target, maxit=200):
        target = np.asarray(target, dtype=float)
        for _ in range(maxit):
            cur = self.base()
            d = target - cur
            m = np.abs(d).max()
            if m < 1e-6:
                return True
            if m > MAXD:
                d = d * (MAXD / m)
            a = np.zeros(11, dtype=np.float32)
            a[0:3] = d
            self.step(a)
            if np.abs(self.base() - cur).max() < 1e-9:
                return False
        return False

    def _ik_once(self, pos, yaw):
        from pybullet_helpers.geometry import Pose
        from pybullet_helpers.inverse_kinematics import pybullet_inverse_kinematics

        try:
            sol = pybullet_inverse_kinematics(
                self.arm, Pose(tuple(float(v) for v in pos), _quat_down(yaw)), validate=True
            )
        except Exception:
            return None
        return np.array(sol[:7], dtype=float)

    def ik(self, pos, yaw):
        sol = self._ik_once(pos, yaw)
        if sol is not None:
            return sol
        # retry from other seeds
        full = list(self.arm.get_joint_positions())
        cur = np.array(full[:7], dtype=float)
        best = None
        try:
            for seed in IK_SEEDS:
                self.arm.set_joints(list(seed) + full[7:])
                s2 = self._ik_once(pos, yaw)
                if s2 is not None:
                    if best is None or np.abs(self._jdiff(s2, cur)).max() < np.abs(self._jdiff(best, cur)).max():
                        best = s2
                    break
        finally:
            self.arm.set_joints(full)
        return best

    def move_ee(self, pos, yaw):
        sol = self.ik(pos, yaw)
        if sol is None:
            return False
        return self.move_joints(sol)

    def to_carry(self, z=0.8):
        b = self.base()
        th = b[2]
        p = np.array([b[0] + 0.3 * math.cos(th), b[1] + 0.3 * math.sin(th), z])
        if self.move_ee(p, th):
            return True
        p[2] = 0.7
        return self.move_ee(p, th)

    # ------------------------------------------------------------ navigation
    @staticmethod
    def base_ok(xy):
        lo = TABLE_LO - NAV_MARGIN + 1e-6
        hi = TABLE_HI + NAV_MARGIN - 1e-6
        if np.all(xy > lo) and np.all(xy < hi):
            return False
        return bool(np.all(np.abs(xy) < 9.0))

    @staticmethod
    def route(p, q):
        lo = TABLE_LO - NAV_MARGIN + 0.02
        hi = TABLE_HI + NAV_MARGIN - 0.02
        pre = []
        if np.all(p > lo) and np.all(p < hi):
            # escape straight out through the nearest side
            dists = [p[0] - lo[0], hi[0] - p[0], p[1] - lo[1], hi[1] - p[1]]
            i = int(np.argmin(dists))
            p = p.copy()
            if i == 0:
                p[0] = lo[0] - 0.03
            elif i == 1:
                p[0] = hi[0] + 0.03
            elif i == 2:
                p[1] = lo[1] - 0.03
            else:
                p[1] = hi[1] + 0.03
            pre = [p]
        if not _seg_hits_rect(p, q, lo, hi):
            return pre + [q]
        e = 0.05
        corners = [
            np.array([lo[0] - e, lo[1] - e]),
            np.array([hi[0] + e, lo[1] - e]),
            np.array([hi[0] + e, hi[1] + e]),
            np.array([lo[0] - e, hi[1] + e]),
        ]
        nodes = [p, q] + corners
        n = len(nodes)
        INF = 1e18
        dist = [INF] * n
        prev = [-1] * n
        dist[0] = 0.0
        seen = [False] * n
        for _ in range(n):
            u = min((i for i in range(n) if not seen[i]), key=lambda i: dist[i])
            if dist[u] >= INF:
                break
            seen[u] = True
            for v in range(n):
                if seen[v] or _seg_hits_rect(nodes[u], nodes[v], lo, hi):
                    continue
                nd = dist[u] + float(np.linalg.norm(nodes[u] - nodes[v]))
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
        path = []
        v = 1
        while v != 0 and v != -1:
            path.append(nodes[v])
            v = prev[v]
        return pre + path[::-1]

    def _ckey(self, *extra):
        return (tuple(np.round(self.base(), 5)), tuple(np.round(self.joints(), 5)),
                tuple(np.round(np.concatenate([np.ravel(np.asarray(e, dtype=float)) for e in extra]), 5))
                if extra else ())

    def rel_ik(self, bgoal, pos, yaw):
        key = ("rel",) + self._ckey(bgoal, pos, [yaw])
        if key in self._cache:
            return self._cache[key]
        r = self._rel_ik(bgoal, pos, yaw)
        self._cache[key] = r
        return r

    def _rel_ik(self, bgoal, pos, yaw):
        """IK for world target `pos` assuming the base were at `bgoal`."""
        b = self.base()
        dx, dy = pos[0] - bgoal[0], pos[1] - bgoal[1]
        c, s = math.cos(-bgoal[2]), math.sin(-bgoal[2])
        rx, ry = c * dx - s * dy, s * dx + c * dy
        c, s = math.cos(b[2]), math.sin(b[2])
        w = [b[0] + c * rx - s * ry, b[1] + s * rx + c * ry, pos[2]]
        return self.ik(w, yaw - bgoal[2] + b[2])

    @staticmethod
    def _jdiff(a, b):
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        for i in UNBOUNDED:
            d[i] = _wrap(d[i])
        return d

    def move_combined(self, btarget, jfinal, jwait=None, rest=0.0, maxit=200, via=True):
        """Move base and joints together.

        Joints first go to `jwait` (if given); they then stay there until the
        remaining base motion (`rest` = steps after this waypoint) is short
        enough that heading to `jfinal` makes both arrive together.
        Returns True when the base reached btarget (joints may not have
        reached jfinal yet if this is not the last waypoint)."""
        btarget = np.asarray(btarget, dtype=float)
        reached_wait = jwait is None
        for _ in range(maxit):
            cb, cj = self.base(), self.joints()
            nb = np.abs(btarget - cb).max() / MAXD + rest
            if jwait is not None and not reached_wait:
                if not via or np.abs(self._jdiff(jwait, cj)).max() < 1e-6:
                    reached_wait = True
            if jfinal is None:
                jt = jwait
            elif not reached_wait:
                jt = jwait
            else:
                nj = np.abs(self._jdiff(jfinal, cj)).max() / MAXD
                jt = jfinal if math.ceil(nb - 1e-9) <= math.ceil(nj - 1e-9) else jwait
            bd = btarget - cb
            m = np.abs(bd).max()
            if m > MAXD:
                bd = bd * (MAXD / m)
            if via and not reached_wait:
                bd = np.zeros(3)  # raise the arm first, then drive
            jd = np.zeros(7) if jt is None else self._jdiff(jt, cj)
            m2 = np.abs(jd).max()
            if m2 > MAXD:
                jd = jd * (MAXD / m2)
            if np.abs(bd).max() < 1e-6:
                if rest > 0 or np.abs(jd).max() < 1e-6:
                    return True
            a = np.zeros(11, dtype=np.float32)
            a[0:3] = bd
            a[3:10] = jd
            self.step(a)
            if (np.abs(self.joints() - cj).max() < 1e-9
                    and np.abs(self.base() - cb).max() < 1e-9):
                return False
        return False

    def _cur_yaw(self):
        from scipy.spatial.transform import Rotation as R

        q = self.arm.get_end_effector_pose().orientation
        m = R.from_quat(q).as_matrix()
        return math.atan2(m[1, 0], m[0, 0])

    def carry_joints(self, z=0.8):
        key = ("carry", z) + self._ckey()
        if key in self._cache:
            return self._cache[key]
        r = self._carry_joints(z)
        self._cache[key] = r
        return r

    def _carry_joints(self, z=0.8):
        b = self.base()
        p = [b[0] + 0.3 * math.cos(b[2]), b[1] + 0.3 * math.sin(b[2]), z]
        j = self.ik(p, b[2])
        if j is None:
            p[2] = 0.7
            j = self.ik(p, b[2])
        return j

    def navigate_slow(self, x, y, th):
        if not self.to_carry():
            raise PlanFail("carry")
        b = self.base()
        th = float(th)
        if not self.move_base([b[0], b[1], th]):
            raise PlanFail("rot")
        if not self.to_carry():
            raise PlanFail("carry2")
        for wp in self.route(self.base()[:2], np.array([x, y])):
            if not self.move_base([wp[0], wp[1], th]):
                raise PlanFail("nav")

    def navigate(self, x, y, th, final=None, phase=None):
        """Drive to base pose (x, y, th); end with EE at `final` (pos, yaw) if given.

        phase: None = all modes; "fast" = only direct/lift modes; "slow" = via + slow."""
        th = float(th)
        bgoal = np.array([x, y, th])
        modes = {None: (0, 2, 3, 4, 1), "fast": (0, 2, 3, 4), "slow": (1,)}.get(phase)
        if modes is None:
            modes = (phase,)
        for attempt in (modes if FAST else ()):
            cp = self.checkpoint()
            if attempt >= 2:
                e = self.ee()
                lz = e[2] + {2: 0.07, 3: 0.15, 4: 0.15}[attempt]
                if attempt == 4:
                    if lz >= 0.6:
                        continue
                    lz = 0.6
                if e[2] > 0.75 or not self.move_ee(np.array([e[0], e[1], lz]), self._cur_yaw()):
                    self.restore(cp)
                    continue
            jc = self.carry_joints(CARRY_Z)
            jf = None
            if final is not None:
                jf = self.rel_ik(bgoal, final[0], final[1])
            if jc is None or (final is not None and jf is None):
                break
            wps = self.route(self.base()[:2], np.array([x, y]))
            seg = []
            prev = self.base()[:2]
            for wp in wps:
                seg.append(float(np.abs(wp - prev).max()) / MAXD)
                prev = wp
            ok = True
            for i, wp in enumerate(wps):
                rest = sum(seg[i + 1:])
                if not self.move_combined([wp[0], wp[1], th], jf, jwait=jc, rest=rest,
                                          via=attempt == 1):
                    ok = False
                    break
            if ok:
                if jf is None or np.abs(self._jdiff(jf, self.joints())).max() < 1e-6:
                    return
                if self.move_joints(jf):
                    return
            self.log.append(f"combined nav blocked at base={np.round(self.base(),2)} ee={np.round(self.ee(),2)} goal={np.round(bgoal,2)} held={self.s.grasped_object}")
            self.restore(cp)
        if phase is not None and phase != "slow":
            raise PlanFail("fast nav")
        self.navigate_slow(x, y, th)
        if final is not None and not self.move_ee(final[0], final[1]):
            raise PlanFail("final")

    # ------------------------------------------------------------ skills
    def half_extents(self, name):
        o = self.s.get_object_from_name(name)
        return np.array([self.s.get(o, "half_extent_" + c) for c in "xyz"])

    def heading(self, phi):
        """Representation of angle phi within [-pi, pi] closest to current heading."""
        cur = self.base()[2]
        best = None
        for k in (-1, 0, 1):
            v = phi + 2 * math.pi * k
            if -math.pi - 1e-9 <= v <= math.pi + 1e-9:
                if best is None or abs(v - cur) < abs(best - cur):
                    best = v
        if best is None:
            best = _wrap(phi)
        return float(np.clip(best, -math.pi, math.pi))

    @staticmethod
    def heading_from(cur, phi):
        best = None
        for k in (-1, 0, 1):
            v = phi + 2 * math.pi * k
            if -math.pi - 1e-9 <= v <= math.pi + 1e-9:
                if best is None or abs(v - cur) < abs(best - cur):
                    best = v
        if best is None:
            best = _wrap(phi)
        return float(np.clip(best, -math.pi, math.pi))

    def return_est(self, bg, targets):
        """Estimated steps from pick base pose bg to a place pose for any target."""
        best = 1e9
        for t in targets:
            for i in range(NDIR):
                phi = -math.pi + i * 2 * math.pi / NDIR
                u = np.array([math.cos(phi), math.sin(phi)])
                pb = np.asarray(t[:2]) - RET_D * u
                if not self.base_ok(pb):
                    continue
                th = self.heading_from(bg[2], phi)
                best = min(best, self.travel_steps(np.array([pb[0], pb[1], th]), start=bg))
        return 0.0 if best >= 1e9 else best

    def cur_alpha(self):
        b, e = self.base(), self.ee()
        return _wrap(math.atan2(e[1] - b[1], e[0] - b[0]) - b[2])

    def travel_steps(self, bgoal, start=None, alpha=None):
        b = self.base() if start is None else np.asarray(start, dtype=float)
        path = self.route(b[:2], np.asarray(bgoal[:2], dtype=float))
        pts = [b[:2]] + list(path)
        L = 0.0
        for i in range(len(pts) - 1):
            L += float(np.abs(pts[i + 1] - pts[i]).max())
        dth = abs(bgoal[2] - b[2])
        if alpha is not None and ARM_COST:
            dth = max(dth, abs(_wrap(alpha - self.cur_alpha())))
        return max(L, dth) / MAXD

    def pick_geom(self, name, phi, d, gy, alpha=0.0):
        p = self.obj_pos(name)
        he = self.half_extents(name)
        card = round(phi / (math.pi / 2)) * (math.pi / 2)
        if he[0] > 0.04:  # box: grasp the wall facing the robot
            u = np.array([math.cos(card), math.sin(card)])
            along = abs(u[0]) * he[0] + abs(u[1]) * he[1]
            g = np.array([p[0] - u[0] * (along - 0.005), p[1] - u[1] * (along - 0.005),
                          p[2] + he[2]])
        else:
            g = np.array([p[0], p[1], p[2] + 0.03])
        uphi = np.array([math.cos(phi), math.sin(phi)])
        bxy = g[:2] - d * uphi
        th = self.heading(phi + alpha)
        return g, card + gy, np.array([bxy[0], bxy[1], th])

    def nav_seq(self, phs):
        if self.eval_m <= 1 and len(phs) >= 3:
            return [(0, phs[0]), (2, phs[0]), (3, phs[-1]), (4, phs[-1]),
                    ("slow", phs[0]), ("slow", phs[-1])]
        seq = [(m, ph) for m in (0, 2, 3) for ph in phs]
        seq += [(4, phs[0]), (4, phs[-1]), ("slow", phs[0]), ("slow", phs[-1])]
        return seq

    def try_pick(self, name, phi, d, gy=0.0, alpha=0.0):
        g, gyaw, bg = self.pick_geom(name, phi, d, gy, alpha)
        if not self.base_ok(bg[:2]):
            raise PlanFail("base pose")
        ok = False
        for phase, ph in self.nav_seq((0.0, PRE_FAST, PRE_H)):
            cp = self.checkpoint()
            try:
                self.navigate(bg[0], bg[1], bg[2], final=(g + [0, 0, ph], gyaw), phase=phase)
                if self.move_ee(g, gyaw):
                    ok = True
                    break
            except PlanFail:
                pass
            self.restore(cp)
        if not ok:
            raise PlanFail("down")
        self.gripper(close=True)
        if self.s.grasped_object != name:
            raise PlanFail("grasp")
        off = self.obj_pos(name) - self.ee()
        if self.half_extents(name)[0] > 0.04 and BOX_LIFT:
            if not self.move_ee(g + [0, 0, 0.45 - he_z(self, name)], gyaw):
                raise PlanFail("lift")
        self.log.append(f"picked {name} phi={phi:.2f} base={np.round(self.base(),2)} step {len(self.acts)}")
        return off, gyaw

    def pick(self, name, targets=None):
        is_box = self.half_extents(name)[0] > 0.04
        cands = []
        for i in range(NDIR):
            phi = -math.pi + i * 2 * math.pi / NDIR
            for d in PICK_DS:
                for gy in ((0.0,) if is_box else (0.0, math.pi / 2)):
                  for al in ((0.0,) if is_box else self.alphas):
                    g, gyaw, bg = self.pick_geom(name, phi, d, gy, al)
                    if not self.base_ok(bg[:2]):
                        continue
                    cost = self.travel_steps(bg, alpha=-al) + (0 if d == 0.5 else 0.3) + (0 if gy == 0 else 0.5)
                    cost += PICK_DPEN * max(0.0, d - 0.58) + ALPHA_PEN * abs(al)
                    cands.append((cost, phi, d, gy, al))
        cands.sort()
        if targets and LOOKAHEAD:
            top = []
            for cost, phi, d, gy, al in cands[:LOOK_N]:
                _, _, bg = self.pick_geom(name, phi, d, gy, al)
                top.append((cost + self.return_est(bg, targets), phi, d, gy, al))
            top.sort()
            cands = top + cands[LOOK_N:]
        best = None
        nok = 0
        # Top candidates first; if all fail, fall back to diverse (phi, gy) choices.
        seen = set((round(c[1], 3), c[3]) for c in cands[:16])
        extra = []
        for c in cands[16:]:
            key = (round(c[1], 3), c[3])
            if key in seen:
                continue
            seen.add(key)
            extra.append(c)
        order = cands[:16] + extra[:32]
        for idx, (_, phi, d, gy, al) in enumerate(order):
            if idx >= 16 and (best is not None or time.time() > self.t_deadline):
                break
            cp = self.checkpoint()
            try:
                res = self.try_pick(name, phi, d, gy, al)
            except PlanFail as e:
                self.log.append(f"pick {name} phi={phi:.2f} d={d} fail {e} base={np.round(self.base(),2)}")
                self.restore(cp)
                continue
            if self.eval_m <= 1:
                return res
            snap = self.snapshot(res)
            if best is None or len(snap[1]) < len(best[1]):
                best = snap
            nok += 1
            self.restore(cp)
            if nok >= self.eval_m:
                break
        if best is None:
            raise PlanFail("pick " + name)
        return self.restore_snapshot(best)

    @staticmethod
    def held_offset(off, th_g, gyaw):
        c, s_ = math.cos(gyaw - th_g), math.sin(gyaw - th_g)
        return np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1], off[2]])

    def place_cands(self, center, off, th_g, extra=0.0, alphas=None):
        """Candidate (cost, phi, d) for placing the held object at center."""
        out = []
        for i in range(NDIR):
            phi = -math.pi + i * 2 * math.pi / NDIR
            gyaw = round(phi / (math.pi / 2)) * (math.pi / 2)
            e = np.asarray(center) - self.held_offset(off, th_g, gyaw)
            u = np.array([math.cos(phi), math.sin(phi)])
            for d in PLACE_DS:
                bxy = e[:2] - d * u
                if not self.base_ok(bxy):
                    continue
                for al in (self.alphas if alphas is None else alphas):
                    bg = np.array([bxy[0], bxy[1], self.heading(phi + al)])
                    cost = self.travel_steps(bg, alpha=-al) + extra - 0.5 * d + ALPHA_PEN * abs(al)
                    out.append((cost, phi, d, al))
                if not PLACE_ALL_D:
                    break  # smallest feasible d for this direction
        return out

    def try_place(self, name, center, off, th_g, phi, d, pre_h=PRE_H, alpha=0.0):
        """Place held object so that its center ends up at `center`."""
        gyaw = round(phi / (math.pi / 2)) * (math.pi / 2)
        e = np.asarray(center) - self.held_offset(off, th_g, gyaw)
        u = np.array([math.cos(phi), math.sin(phi)])
        bxy = e[:2] - d * u
        ok = False
        if pre_h > PRE_FAST:
            seq = self.nav_seq((0.004, PRE_FAST, pre_h))
        else:
            seq = ((None, pre_h),)
        for phase, ph in seq:
            cp = self.checkpoint()
            try:
                self.navigate(bxy[0], bxy[1], self.heading(phi + alpha), final=(e + [0, 0, ph], gyaw),
                              phase=phase)
                for dz in (0.004, 0.007, 0.002, 0.0):
                    if self.move_ee(e + [0, 0, dz], gyaw):
                        ok = True
                        break
            except PlanFail:
                pass
            if ok:
                break
            self.restore(cp)
        if not ok:
            raise PlanFail("placedown")
        self.gripper(close=False)
        if self.s.grasped_object is not None:
            raise PlanFail("release")
        if self.done:
            return
        if self.all_up() and self.backoff(gyaw):
            return
        if RETREAT > 0 and not self.move_ee(e + [0, 0, RETREAT], gyaw):
            raise PlanFail("retreat")
        if not self.can_escape(gyaw):
            raise PlanFail("trapped")
        self.log.append(f"placed {name} at {np.round(center,2)} phi {phi:.2f} d {d} step {len(self.acts)}")


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.sims = {}
        self.plan = []
        self.deadline = time.time() + 1e9

    def _sim(self, k):
        if k not in self.sims:
            self.sims[k] = Sim(k)
        return self.sims[k]

    # ---------------------------------------------------------------- planning
    def _make_plan(self, state):
        names = list(state.get_object_names())
        cubes = [n for n in names if n.startswith("cube")]
        boxes = [n for n in names if n.startswith("box")]
        self.t_plan = time.time()
        sim = self._sim(len(cubes))
        sim.load(state)
        sim.eval_m = EVAL_M
        sim.t_deadline = self.deadline
        sim.alphas = ALPHAS
        occupied = []  # (center_xy, half_xy) on the table
        SIDES = ("W", "E", "S", "N")

        def he_of(n):
            o = state.get_object_from_name(n)
            return np.array([state.get(o, "half_extent_" + c) for c in "xyz"])

        def free(xy, hxy, margin):
            return not any(np.all(np.abs(xy - oc) < oh + hxy + margin) for oc, oh in occupied)

        # ---------------- boxes
        for bn in boxes:
            if sim.obj_pos(bn)[2] >= 0.3 + 0.05:
                continue
            cp_b = sim.checkpoint()
            try:
                off, th_g = sim.pick(bn)
            except Exception:
                sim.restore(cp_b)
                continue
            he = he_of(bn)
            cands = []
            for fx in (-1, 0, 1):
                for fy in (-1, 0, 1):
                    for rot in (0, 1):
                        ex, ey = (he[1], he[0]) if rot else (he[0], he[1])
                        center = np.array([0.6 + fx * (0.2 - ex), fy * (0.4 - ey), TABLE_TOP + he[2]])
                        if not free(center[:2], np.array([ex, ey]), 0.0):
                            continue
                        for cost, phi, d, _al in sim.place_cands(center, off, th_g, alphas=(0.0,)):
                            gyaw = round(phi / (math.pi / 2)) * (math.pi / 2)
                            if (abs(math.sin(gyaw - th_g)) > 0.5) != bool(rot):
                                continue
                            cost += 0.3 * (fx == 0) + 0.6 * (fy == 0)
                            cands.append((cost, phi, d, tuple(center), ex, ey))
            cands.sort(key=lambda c: c[0])
            placed = False
            for cost, phi, d, center, ex, ey in cands[:10]:
                cp = sim.checkpoint()
                try:
                    sim.try_place(bn, np.array(center), off, th_g, phi, d)
                    occupied.append((np.array(center[:2]), np.array([ex, ey])))
                    placed = True
                    break
                except Exception as e:
                    sim.log.append(f"place box {center} {phi:.2f} fail {e}")
                    sim.restore(cp)
            if not placed:
                # fallback: lift the box high and release (box counts as its own support)
                b = sim.base()
                e = sim.ee()
                sim.move_ee(np.array([e[0], e[1], 0.75]), b[2])
                sim.gripper(close=False)
                sim.move_ee(np.array([e[0], e[1], 0.95]), b[2])

        # ---------------- cube slots on the table grid
        def make_slots(px, py):
            out = []
            xs = np.arange(0.8 - 0.035, 0.4 + 0.03, -px)
            xs = xs - (xs.mean() - 0.6)
            ys = np.arange(-0.4 + 0.035, 0.4 - 0.034, py)
            ys = ys - ys.mean()
            for x in xs:
                for y in ys:
                    xy = np.array([x, y])
                    if free(xy, np.zeros(2), 0.065):
                        out.append(xy)
            return out

        n_need = sum(1 for c in cubes if sim.obj_pos(c)[2] < 0.3)
        slots = []
        for px, py in PITCHES:
            slots = make_slots(px, py)
            if len(slots) >= n_need + 1:
                break
        used = [False] * len(slots)

        remaining = [c for c in cubes if sim.obj_pos(c)[2] < 0.3]
        tries = {c: 0 for c in remaining}
        n_done = 0
        while remaining and not sim.done:
            if time.time() > self.deadline:
                break
            b = sim.base()[:2]
            remaining.sort(key=lambda c: (tries[c], np.linalg.norm(sim.obj_pos(c)[:2] - b)))
            cn = remaining.pop(0)
            el = time.time() - self.t_plan
            per = el / max(1, n_done)
            quality = NOGATE or (el < 12 and el + per * (len(remaining) + 1) < 20)
            sim.eval_m = EVAL_M if quality else 1
            sim.alphas = ALPHAS if quality else (0.0,)
            n_done += 1
            cp_c = sim.checkpoint()
            cpos = sim.obj_pos(cn)[:2]
            fs = sorted((float(np.linalg.norm(slots[i] - cpos)), i) for i in range(len(slots))
                        if not used[i])
            targets = [slots[i] for _, i in fs[:2]]
            try:
                off, th_g = sim.pick(cn, targets)
            except Exception:
                sim.restore(cp_c)
                tries[cn] += 1
                if tries[cn] < 3 and remaining:
                    remaining.append(cn)
                continue
            hz = he_of(cn)[2]
            cands = []
            for i, xy in enumerate(slots):
                if used[i]:
                    continue
                center = np.array([xy[0], xy[1], TABLE_TOP + hz])
                for cost, phi, d, al in sim.place_cands(center, off, th_g):
                    cands.append((cost, i, phi, d, al))
            cands.sort(key=lambda c: c[0])
            placed = False
            best = None
            nok = 0
            for cost, i, phi, d, al in cands[:12]:
                xy = slots[i]
                cp = sim.checkpoint()
                try:
                    sim.try_place(cn, np.array([xy[0], xy[1], TABLE_TOP + hz]), off, th_g, phi, d,
                                  alpha=al)
                except Exception as e:
                    sim.log.append(f"place {cn} slot={xy} {phi:.2f} fail {e}")
                    sim.restore(cp)
                    continue
                snap = sim.snapshot(i)
                if best is None or len(snap[1]) < len(best[1]):
                    best = snap
                nok += 1
                if nok >= sim.eval_m:
                    break
                sim.restore(cp)
            if best is not None:
                used[sim.restore_snapshot(best)] = True
                placed = True
            if not placed:
                sim.restore(cp_c)
                tries[cn] += 1
                if tries[cn] < 3 and remaining:
                    remaining.append(cn)
        if not sim.done:
            # make sure the gripper is open and away from objects
            if sim.s.grasped_object is not None:
                sim.gripper(close=False)
            b = sim.base()
            e = sim.ee()
            sim.move_ee(np.array([e[0], e[1], e[2] + 0.22]), self._ee_yaw(sim))
        acts = list(sim.acts)
        self.last_log = list(sim.log)
        if sim.done and MERGE_GRIPPER:
            try:
                merged = self._merge_gripper(acts)
                if len(merged) < len(acts):
                    sim.load(state)
                    for a in merged:
                        sim.step(a)
                        if sim.done:
                            break
                    if sim.done and len(sim.acts) <= len(merged):
                        return list(sim.acts)
            except Exception:
                pass
        return acts

    @staticmethod
    def _merge_gripper(acts):
        """Fold gripper-only actions into the preceding motion step (the env
        applies motion before the gripper command within one step)."""
        out = []
        for a in acts:
            a = np.asarray(a, dtype=np.float32)
            g_only = np.abs(a[:10]).max() < 1e-12 and abs(a[10]) > 0.5
            if (g_only and out and abs(out[-1][10]) <= 0.5
                    and np.abs(out[-1][:10]).max() > 1e-12):
                prev = out[-1].copy()
                prev[10] = a[10]
                out[-1] = prev
            else:
                out.append(a)
        return out

    @staticmethod
    def _ee_yaw(sim):
        from scipy.spatial.transform import Rotation as R

        q = sim.arm.get_end_effector_pose().orientation
        m = R.from_quat(q).as_matrix()
        return math.atan2(m[1, 0], m[0, 0])

    def _safe_plan(self, state):
        try:
            return self._make_plan(state)
        except Exception:
            try:
                sims = [s for s in self.sims.values() if s.acts]
                return list(sims[0].acts) if sims else []
            except Exception:
                return []

    def reset(self, state, info):
        self.t0 = time.time()
        self.deadline = self.t0 + 40.0
        self.replans = 0
        self.plan = self._safe_plan(state)
        self.idx = 0

    def get_action(self, state):
        if self.idx >= len(self.plan) and self.replans < 3 and time.time() - self.t0 < 40:
            # plan exhausted without termination: replan from the current state
            self.replans += 1
            self.plan = self._safe_plan(state)
            self.idx = 0
        if self.idx < len(self.plan):
            a = self.plan[self.idx]
            self.idx += 1
            return np.asarray(a, dtype=np.float32)
        return np.zeros(11, dtype=np.float32)
