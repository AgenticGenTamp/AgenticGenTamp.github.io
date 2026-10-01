"""Approach for Transport3DEnv: pick and place all objects onto the table."""
from __future__ import annotations

import math
import time

import numpy as np

import kin

JOINT_FEATS = ["joint_%d" % i for i in range(1, 8)]
# Calibrated geometry of the mobile manipulator (empirically identified).
ARM_FWD = 0.12          # arm column is this far ahead of the base origin (base frame +x)
MZ = 0.240              # arm mount height such that fk() yields the true grasp point
FINGER_BELOW = 0.045    # fingertips extend this far below the grasp point
BASE_CLEAR = 0.22       # keep base origin this far outside table AABB


def _gf(state, obj, feat, default=0.0):
    try:
        return float(state.get(obj, feat))
    except Exception:
        return default


def _has(state, obj, feat):
    try:
        state.get(obj, feat)
        return True
    except Exception:
        return False


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.low = np.asarray(action_space.low, dtype=np.float64)
        self.high = np.asarray(action_space.high, dtype=np.float64)

    # ------------------------------------------------------------------ setup
    def reset(self, state, info):
        self._parse(state)
        self.plan = []          # list of (kind, payload)
        self.phase = None
        self.q_target = None
        self.base_target = None
        self.path = None
        self.stuck = 0
        self.cur = None
        self.place_idx = 0
        self.place_gap = 0.002
        self.open_tries = 0
        self.grasp_yaw = 0.0
        self.dither_idx = 0
        self.big_cur = False
        self.arm_traj = None
        self.arm_key = None
        self.assigned = {}
        self.fail_count = {}
        self.attempt = {}
        self.spot_of = {}
        self._t0 = None
        self._budget_low = False
        self._prev_sig = None
        self.stall = 0
        self.t = 0
        self._assign_spots(state)

    def _set_phase(self, ph):
        self.phase = ph
        self.arm_traj = None
        self.arm_key = None
        self.stall = 0

    def _abort(self, state):
        """Current sub-goal is unreachable: give up on this object for now."""
        held = self._held(state)
        if held is not None:
            # keep going: we still have to put it down somewhere
            self.cur = held
            if self.phase in ("drive_pick", "approach", "pregrasp", "descend", "close", "lift"):
                self._set_phase("carry")
            elif self.phase == "preplace":
                self._set_phase("lower")
            elif self.phase == "lower":
                self.place_gap += 0.002
                self._set_phase("open")
            else:
                self._set_phase("carry")
            self.path = None
            self.retry = getattr(self, "retry", 0) + 1
            if self.retry > 12:
                self.phase = "open"
            return
        if self.cur is not None and self.phase in ("approach", "pregrasp", "descend", "close"):
            self.dither_idx += 1
            if self.dither_idx < (len(self.EDGE_DITHER) if self.big_cur
                                  else len(self.DITHER)):
                self._set_phase("pregrasp")
                return
        if self.cur is not None:
            self.fail_count[self.cur.name] = self.fail_count.get(self.cur.name, 0) + 1
        self.dither_idx = 0
        self.cur = None
        self.phase = "done"
        self.path = None

    def _held(self, state):
        for o in self.movable:
            if _gf(state, o, "grasp_active") > 0.5:
                return o
        return None

    def _parse(self, state):
        self.robot = None
        cuboids = []
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            if _has(state, obj, "joint_1"):
                self.robot = obj
            elif _has(state, obj, "half_extent_x"):
                cuboids.append(obj)
        # table: the cuboid with the largest horizontal footprint
        table = None
        best = -1.0
        for obj in cuboids:
            a = _gf(state, obj, "half_extent_x") * _gf(state, obj, "half_extent_y")
            if obj.name == "table":
                a += 100.0
            if a > best:
                best = a
                table = obj
        self.table = table
        self.movable = [o for o in cuboids if o is not table]
        self.tx = _gf(state, table, "pose_x")
        self.ty = _gf(state, table, "pose_y")
        self.thx = _gf(state, table, "half_extent_x")
        self.thy = _gf(state, table, "half_extent_y")
        self.ttop = _gf(state, table, "pose_z") + _gf(state, table, "half_extent_z")

    def _assign_spots(self, state):
        """Assign a non-overlapping footprint on the table to every movable object."""
        objs = sorted(self.movable,
                      key=lambda o: -float(np.prod(self._footprint(state, o))))
        placed = []
        self.spot_of = {}
        for o in objs:
            big = self._is_big(state, o)
            f = self._footprint(state, o) + (0.055 if len(placed) else 0.03)
            best = None
            xlo = max(self.tx - self.thx + f[0], self.tx - self.thx + 0.12)
            xhi = min(self.tx + self.thx - f[0], self.tx + self.thx - 0.03)
            if xhi < xlo:
                xlo = xhi = self.tx
            xs = np.arange(xlo, xhi + 1e-9, 0.02)
            ys = np.arange(self.ty - self.thy + f[1], self.ty + self.thy - f[1] + 1e-9, 0.02)
            if len(xs) == 0:
                xs = np.array([self.tx])
            if len(ys) == 0:
                ys = np.array([self.ty])
            for y in ys:
                for x in xs:
                    ok = True
                    for (px, py, pf) in placed:
                        if (abs(px - x) < f[0] + pf[0]) and (abs(py - y) < f[1] + pf[1]):
                            ok = False
                            break
                    if ok:
                        # prefer central x, and y close to the table edge already used
                        if big:
                            cost = abs(x - self.tx) * 0.5 + abs(y - (self.ty - self.thy))
                        else:
                            cost = abs(x - self.tx) * 0.5 + abs(y - (self.ty + self.thy))
                        if best is None or cost < best[0]:
                            best = (cost, x, y)
                if best is not None:
                    break
            if best is None:
                x, y = self.tx, self.ty
            else:
                _, x, y = best
            placed.append((x, y, f))
            self.spot_of[o.name] = (float(x), float(y))

    # ------------------------------------------------------- state extraction
    def _rob(self, state):
        r = self.robot
        b = np.array([_gf(state, r, "pos_base_x"), _gf(state, r, "pos_base_y"),
                      _gf(state, r, "pos_base_rot")])
        q = np.array([_gf(state, r, f) for f in JOINT_FEATS])
        g = _gf(state, r, "grasp_active")
        return b, q, g

    def _opos(self, state, obj):
        return np.array([_gf(state, obj, "pose_x"), _gf(state, obj, "pose_y"),
                         _gf(state, obj, "pose_z")])

    def _half(self, state, obj):
        return np.array([_gf(state, obj, "half_extent_x"), _gf(state, obj, "half_extent_y"),
                         _gf(state, obj, "half_extent_z")])

    def _oyaw(self, state, obj):
        qz = _gf(state, obj, "pose_qz")
        qw = _gf(state, obj, "pose_qw", 1.0)
        return 2.0 * math.atan2(qz, qw)

    def _footprint(self, state, obj):
        """World axis-aligned half extents of the object footprint."""
        h = self._half(state, obj)
        c, s_ = abs(math.cos(self._oyaw(state, obj))), abs(math.sin(self._oyaw(state, obj)))
        return np.array([h[0] * c + h[1] * s_, h[0] * s_ + h[1] * c])

    def _grasp_offset(self, state, obj):
        """Offset (world frame) from object centre to the grasp point."""
        h = self._half(state, obj)
        if 2.0 * min(h[0], h[1]) <= 0.07:
            return np.zeros(3)
        yaw = self._oyaw(state, obj)
        # midpoint of a top-face edge: pick the local axis with the smaller extent
        if h[0] <= h[1]:
            loc = np.array([h[0], 0.0])
        else:
            loc = np.array([0.0, h[1]])
        c, s_ = math.cos(yaw), math.sin(yaw)
        w = np.array([c * loc[0] - s_ * loc[1], s_ * loc[0] + c * loc[1]])
        return np.array([w[0], w[1], h[2]])

    def _on_table(self, state, obj):
        p = self._opos(state, obj)
        h = self._half(state, obj)
        if abs(p[2] - (self.ttop + h[2])) > 0.02:
            return False
        if not (self.tx - self.thx - 1e-3 <= p[0] <= self.tx + self.thx + 1e-3):
            return False
        if not (self.ty - self.thy - 1e-3 <= p[1] <= self.ty + self.thy + 1e-3):
            return False
        return True

    def _armbase(self, b):
        return b[0] + ARM_FWD * math.cos(b[2]), b[1] + ARM_FWD * math.sin(b[2])

    def _ik(self, b, q, pos, yaw=None, restarts=0):
        ax, ay = self._armbase(b)
        if yaw is None:
            yaw = b[2]
        pos = np.asarray(pos, dtype=float)
        seeds = [q]
        if restarts:
            seeds += [np.asarray(kin.Q_HOME, dtype=float),
                      np.asarray(kin.Q_RETRACT, dtype=float)]
        if self._budget_low:
            seeds = seeds[:1]
        best = None
        besterr = 1e9
        for i, seed in enumerate(seeds):
            qd = kin.ik_top_down(pos, yaw=yaw, q_init=seed,
                                 base_x=ax, base_y=ay, base_rot=b[2], mount=(0.0, 0.0, MZ),
                                 restarts=(1 if i else 0), max_iters=40, time_budget=0.015)
            if qd is None:
                continue
            T = kin.fk(qd, base_x=ax, base_y=ay, base_rot=b[2], mount=(0.0, 0.0, MZ))
            err = float(np.linalg.norm(T[:3, 3] - pos)) + 0.2 * (1.0 + float(T[2, 2]))
            if err < besterr:
                besterr, best = err, qd
            if err < 2e-3:
                return qd
        if best is not None and besterr < 0.02:
            return best
        return None

    def _ee(self, b, q):
        ax, ay = self._armbase(b)
        return kin.fk(q, base_x=ax, base_y=ay, base_rot=b[2], mount=(0.0, 0.0, MZ))[:3, 3]

    # ------------------------------------------------------------ base motion
    def _base_blocked_rect(self):
        m = BASE_CLEAR
        return (self.tx - self.thx - m, self.tx + self.thx + m,
                self.ty - self.thy - m, self.ty + self.thy + m)

    def _seg_free(self, p0, p1):
        x0, x1, y0, y1 = self._base_blocked_rect()
        # sample the segment
        d = np.linalg.norm(np.asarray(p1) - np.asarray(p0))
        n = max(2, int(d / 0.03) + 1)
        for t in np.linspace(0.0, 1.0, n):
            p = (1 - t) * np.asarray(p0) + t * np.asarray(p1)
            if x0 < p[0] < x1 and y0 < p[1] < y1:
                return False
        return True

    def _base_path(self, start, goal):
        if self._seg_free(start, goal):
            return [np.asarray(goal, dtype=float)]
        x0, x1, y0, y1 = self._base_blocked_rect()
        e = 0.02
        corners = [np.array([x0 - e, y0 - e]), np.array([x0 - e, y1 + e]),
                   np.array([x1 + e, y0 - e]), np.array([x1 + e, y1 + e])]
        best = None
        for c in corners:
            if self._seg_free(start, c) and self._seg_free(c, goal):
                cost = np.linalg.norm(c - start) + np.linalg.norm(np.asarray(goal) - c)
                if best is None or cost < best[0]:
                    best = (cost, [c, np.asarray(goal, dtype=float)])
        if best is not None:
            return best[1]
        for c1 in corners:
            for c2 in corners:
                if c1 is c2:
                    continue
                if (self._seg_free(start, c1) and self._seg_free(c1, c2)
                        and self._seg_free(c2, goal)):
                    return [c1, c2, np.asarray(goal, dtype=float)]
        return [np.asarray(goal, dtype=float)]

    def _outside_rect(self, p):
        x0, x1, y0, y1 = self._base_blocked_rect()
        p = np.array(p, dtype=float)
        if not (x0 < p[0] < x1 and y0 < p[1] < y1):
            return p
        # push out to nearest side
        cand = [np.array([x0 - 0.01, p[1]]), np.array([x1 + 0.01, p[1]]),
                np.array([p[0], y0 - 0.01]), np.array([p[0], y1 + 0.01])]
        return min(cand, key=lambda c: np.linalg.norm(c - p))

    # ------------------------------------------------------------ main policy
    def get_action(self, state):
        if self._t0 is None:
            self._t0 = time.time()
        self._budget_low = (time.time() - self._t0) > 48.0
        a = np.zeros(11, dtype=np.float32)
        try:
            a = self._act(state)
        except Exception:
            a = np.zeros(11, dtype=np.float32)
        return np.clip(np.asarray(a, dtype=np.float32), self.low, self.high).astype(np.float32)

    def _act(self, state):
        b, q, g = self._rob(state)
        sig = np.concatenate([b, q, [g]])
        if self._prev_sig is not None and np.allclose(sig, self._prev_sig):
            self.stall += 1
        else:
            self.stall = 0
        self._prev_sig = sig
        if self.stall > 6:
            self.stall = 0
            self._abort(state)
        if self.cur is None or self.phase == "done":
            self._pick_next(state, b)
        if self.cur is None:
            return self._park(b, q)
        return self._run_phase(state, b, q, g)

    def _park(self, b, q):
        """Nothing left to do: retract the arm so the goal check is clean."""
        qd = self._wrap_near(np.asarray(kin.Q_RETRACT, dtype=float), q)
        e = qd - q
        a = np.zeros(11)
        if np.abs(e).max() > 1e-4:
            a[3:10] = np.clip(e, -0.2, 0.2)
        elif b[0] > self.tx - self.thx - 0.6:
            a[0] = -0.2
        return a

    def _pick_next(self, state, b):
        held = self._held(state)
        if held is not None:
            self.cur = held
            self.big_cur = self._is_big(state, held)
            self._set_phase("carry")
            self.retry = 0
            return
        remaining = [o for o in self.movable if not self._on_table(state, o)]
        pool = [o for o in remaining if self.fail_count.get(o.name, 0) < 3]
        if remaining and not pool:
            self.fail_count = {}
            pool = remaining
        remaining = pool
        if not remaining:
            self.cur = None
            self.phase = None
            return
        # nearest first
        remaining.sort(key=lambda o: (-float(np.prod(self._footprint(state, o))),
                                      np.linalg.norm(self._opos(state, o)[:2] - b[:2])))
        self.attempt[remaining[0].name] = self.attempt.get(remaining[0].name, 0) + 1
        self.cur = remaining[0]
        self.big_cur = self._is_big(state, self.cur)
        self.retry = 0
        self._set_phase("drive_pick")
        self._plan_drive_pick(state, b)

    # -- phase helpers
    def _plan_drive_pick(self, state, b):
        p = self._opos(state, self.cur)
        h = self._half(state, self.cur)
        d = 0.45 + (max(h[0], h[1]) if self._is_big(state, self.cur) else 0.0)
        x0, x1, y0, y1 = self._base_blocked_rect()
        best = None
        base_ang = math.atan2(p[1] - b[1], p[0] - b[0])
        base_ang += 0.8 * self.attempt.get(self.cur.name, 0)
        for k in range(36):
            th = base_ang + math.pi + (k // 2 + 1) * (1 if k % 2 else -1) * math.pi / 18.0
            if k == 0:
                th = base_ang + math.pi
            cand = p[:2] + d * np.array([math.cos(th), math.sin(th)])
            if x0 < cand[0] < x1 and y0 < cand[1] < y1:
                continue
            cost = np.linalg.norm(cand - b[:2])
            if not self._seg_free(b[:2], cand):
                cost += 3.0
            if best is None or cost < best[0]:
                best = (cost, cand)
            if best is not None and cost < 1e9 and k > 6 and best[0] < 1.0:
                break
        goal = best[1] if best is not None else self._outside_rect(
            p[:2] - d * np.array([math.cos(base_ang), math.sin(base_ang)]))
        self.base_goal_rot = math.atan2(p[1] - goal[1], p[0] - goal[0])
        self.path = self._base_path(b[:2], goal)

    def _drive(self, b, wp, rot):
        e = np.zeros(3)
        e[:2] = np.asarray(wp) - b[:2]
        e[2] = (rot - b[2] + math.pi) % (2 * math.pi) - math.pi
        a = np.zeros(11)
        a[:3] = np.clip(e, -0.2, 0.2)
        return a, np.abs(e).max()

    @staticmethod
    def _wrap_near(qd, q):
        qd = np.asarray(qd, dtype=float).copy()
        for i in (0, 2, 4, 6):
            qd[i] += 2 * math.pi * round((q[i] - qd[i]) / (2 * math.pi))
        return qd

    def _plan_cart(self, b, q, target, yaw=None, res=0.06):
        if self._budget_low:
            res = max(res, 0.12)
        """Plan a joint trajectory along a straight Cartesian line."""
        cur = self._ee(b, q)
        target = np.asarray(target, dtype=float)
        d = target - cur
        n = float(np.linalg.norm(d))
        nsteps = max(1, int(math.ceil(n / res)))
        traj = []
        qc = np.asarray(q, dtype=float)
        for i in range(1, nsteps + 1):
            p = cur + d * (float(i) / nsteps)
            qd = self._ik(b, qc, p, yaw)
            if qd is None:
                qd = self._ik(b, qc, p, yaw, restarts=2)
            if qd is None:
                break
            qd = self._wrap_near(qd, qc)
            traj.append(qd)
            qc = qd
        if not traj:
            return None
        return traj

    def _arm_goto(self, b, q, target, yaw=None, res=0.06):
        """Follow a cached Cartesian trajectory toward *target*.

        Returns (action, done); action is None when planning failed.
        """
        target = np.asarray(target, dtype=float)
        key = (round(float(target[0]), 4), round(float(target[1]), 4),
               round(float(target[2]), 4), None if yaw is None else round(float(yaw), 4))
        if self.arm_key != key or self.arm_traj is None:
            traj = self._plan_cart(b, q, target, yaw, res)
            if traj is None:
                return None, False
            self.arm_key = key
            self.arm_traj = traj
        traj = self.arm_traj
        # drop every waypoint up to the furthest one already reached
        last = -1
        for j in range(len(traj)):
            if np.abs(traj[j] - q).max() < 1e-5:
                last = j
        if last >= 0:
            del traj[:last + 1]
        if not traj:
            return np.zeros(11), True
        # furthest waypoint reachable in one step
        idx = 0
        for j in range(len(traj)):
            if np.abs(traj[j] - q).max() <= 0.2:
                idx = j
            else:
                break
        e = traj[idx] - q
        a = np.zeros(11)
        a[3:10] = np.clip(e, -0.2, 0.2)
        return a, False

    # (tool-frame dx, tool-frame dy, dz, extra tool yaw)
    DITHER = [(0.0, 0.0, 0.0, 0.0),
              (0.0, 0.0, 0.012, 0.0),
              (0.0, 0.0, -0.01, 0.0),
              (0.012, 0.0, 0.0, 0.0),
              (-0.012, 0.0, 0.0, 0.0),
              (0.0, 0.02, 0.0, 0.0),
              (0.0, -0.02, 0.0, 0.0),
              (0.0, 0.0, 0.0, math.pi / 2),
              (0.0, 0.0, 0.02, math.pi / 2),
              (0.012, 0.012, 0.0, 0.0),
              (-0.012, -0.012, 0.0, 0.0),
              (0.0, 0.03, 0.0, math.pi / 2),
              (0.0, -0.03, 0.0, math.pi / 2),
              (0.0, 0.0, -0.02, math.pi)]

    # for large objects: (outward offset beyond the face, dz below top, extra yaw)
    EDGE_DITHER = [(0.012, 0.029, 0.0),
                   (0.012, 0.029, math.pi / 2),
                   (0.0, 0.029, 0.0),
                   (0.03, 0.029, 0.0),
                   (0.012, 0.05, 0.0),
                   (0.03, 0.05, math.pi / 2),
                   (0.0, 0.05, 0.0),
                   (0.03, 0.015, 0.0),
                   (0.045, 0.029, 0.0),
                   (0.012, 0.07, math.pi / 2),
                   (-0.01, 0.04, 0.0),
                   (0.06, 0.029, math.pi / 2)]

    def _is_big(self, state, obj):
        h = self._half(state, obj)
        return 2.0 * min(h[0], h[1]) > 0.07

    def _grasp_point(self, state, b, obj):
        p = self._opos(state, obj)
        h = self._half(state, obj)
        if not self._is_big(state, obj):
            d = self.DITHER[self.dither_idx % len(self.DITHER)]
            yaw = self._grasp_yaw_cur()
            c, s_ = math.cos(yaw), math.sin(yaw)
            return p + np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]])
        # large object: grasp on the midpoint of a top edge facing the robot
        yawo = self._oyaw(state, obj)
        c, s_ = math.cos(yawo), math.sin(yawo)
        axes = [(np.array([c, s_]), h[0]), (np.array([-c, -s_]), h[0]),
                (np.array([-s_, c]), h[1]), (np.array([s_, -c]), h[1])]
        v = b[:2] - p[:2]
        nv = np.linalg.norm(v)
        v = v / nv if nv > 1e-6 else np.array([1.0, 0.0])
        n, ext = max(axes, key=lambda t: float(np.dot(t[0], v)))
        d = self.EDGE_DITHER[self.dither_idx % len(self.EDGE_DITHER)]
        w = n * (ext + d[0])
        return p + np.array([w[0], w[1], h[2] - d[1]])

    def _grasp_yaw_cur(self):
        if self.cur is not None and self.big_cur:
            d = self.EDGE_DITHER[self.dither_idx % len(self.EDGE_DITHER)]
            return self.grasp_yaw + d[2]
        d = self.DITHER[self.dither_idx % len(self.DITHER)]
        return self.grasp_yaw + d[3]

    def _run_phase(self, state, b, q, g):
        obj = self.cur
        p = self._opos(state, obj)
        h = self._half(state, obj)
        ph = self.phase
        safe_z = self.ttop + 0.25 + 2.0 * h[2]

        if ph == "drive_pick":
            if not self.path:
                self._plan_drive_pick(state, b)
            wp = self.path[0]
            rot = self.base_goal_rot if len(self.path) == 1 else math.atan2(
                wp[1] - b[1], wp[0] - b[0])
            a, err = self._drive(b, wp, rot)
            if err < 1e-4:
                self.path.pop(0)
                if not self.path:
                    self._set_phase("approach")
                    self.grasp_yaw = b[2]
                    self.dither_idx = 0
                return self._run_phase(state, b, q, g)
            return a

        if ph == "approach":
            gp = self._grasp_point(state, b, obj)
            near_table = (abs(gp[0] - self.tx) < self.thx + 0.35
                          and abs(gp[1] - self.ty) < self.thy + 0.35)
            hz = max(self.ttop + 0.12, gp[2] + 0.22) if near_table else max(0.32, gp[2] + 0.22)
            tgt = np.array([gp[0], gp[1], hz])
            a, done = self._arm_goto(b, q, tgt, yaw=self._grasp_yaw_cur(), res=0.1)
            if a is None:
                self._abort(state)
                return np.zeros(11)
            if done:
                self._set_phase("pregrasp")
                return self._run_phase(state, b, q, g)
            return a

        if ph == "pregrasp":
            gp = self._grasp_point(state, b, obj)
            tgt = np.array([gp[0], gp[1], gp[2] + 0.11])
            a, done = self._arm_goto(b, q, tgt, yaw=self._grasp_yaw_cur())
            if a is None:
                self._abort(state)
                return np.zeros(11)
            if done:
                self._set_phase("descend")
                return self._run_phase(state, b, q, g)
            return a

        if ph == "descend":
            gp = self._grasp_point(state, b, obj)
            a, done = self._arm_goto(b, q, gp, yaw=self._grasp_yaw_cur(), res=0.04)
            if a is None:
                self._abort(state)
                return np.zeros(11)
            if done:
                self._set_phase("close")
                return self._run_phase(state, b, q, g)
            return a

        if ph == "close":
            if g > 0.5:
                self._set_phase("lift")
                return self._run_phase(state, b, q, g)
            self.close_tries = getattr(self, "close_tries", 0) + 1
            if self.close_tries > 1:
                self.close_tries = 0
                self.dither_idx += 1
                if self.dither_idx >= (len(self.EDGE_DITHER) if self.big_cur
                                       else len(self.DITHER)):
                    self.dither_idx = 0
                    self._abort(state)
                    return np.zeros(11)
                self._set_phase("pregrasp")
                return self._run_phase(state, b, q, g)
            a = np.zeros(11)
            a[10] = -1.0
            return a

        if ph == "lift":
            self.grasp_yaw = self._grasp_yaw_cur()
            self.dither_idx = 0
            ee = self._ee(b, q)
            ax, ay = self._armbase(b)
            tgt = np.array([0.5 * (ee[0] + ax + 0.32 * math.cos(b[2])),
                            0.5 * (ee[1] + ay + 0.32 * math.sin(b[2])), safe_z])
            a, done = self._arm_goto(b, q, tgt, yaw=self.grasp_yaw, res=0.1)
            if a is None or done:
                self._set_phase("carry")
                return self._run_phase(state, b, q, g)
            return a

        if ph == "carry":
            ax, ay = self._armbase(b)
            tgt = np.array([ax + 0.32 * math.cos(b[2]), ay + 0.32 * math.sin(b[2]), safe_z])
            a, done = self._arm_goto(b, q, tgt, yaw=self.grasp_yaw, res=0.12)
            if a is None or done:
                self._set_phase("drive_place")
                self._plan_drive_place(state, b, obj)
                return self._run_phase(state, b, q, g)
            return a

        if ph == "drive_place":
            if not self.path:
                self._plan_drive_place(state, b, obj)
            wp = self.path[0]
            rot = self.place_rot if len(self.path) == 1 else math.atan2(
                wp[1] - b[1], wp[0] - b[0])
            a, err = self._drive(b, wp, rot)
            if err < 1e-4:
                self.path.pop(0)
                if not self.path:
                    self._set_phase("preplace")
                return self._run_phase(state, b, q, g)
            return a

        if ph == "preplace":
            # closed loop: put the held object above its target spot
            ee = self._ee(b, q)
            off = ee - p
            sp = self.place_pos
            tgt = np.array([sp[0] + off[0], sp[1] + off[1], safe_z])
            a, done = self._arm_goto(b, q, tgt, yaw=self.grasp_yaw, res=0.1)
            if a is None:
                self._abort(state)
                return np.zeros(11)
            if done:
                self._set_phase("lower")
                self.place_gap = 0.002
                return self._run_phase(state, b, q, g)
            return a

        if ph == "lower":
            ee = self._ee(b, q)
            off = ee - p
            sp = self.place_pos
            desired = np.array([sp[0], sp[1], self.ttop + h[2] + self.place_gap])
            tgt = desired + off
            a, done = self._arm_goto(b, q, tgt, yaw=self.grasp_yaw, res=0.03)
            if a is None:
                self.place_gap += 0.002
                self.arm_traj = None
                self.arm_key = None
                if self.place_gap > 0.02:
                    self._abort(state)
                return np.zeros(11)
            if done:
                self._set_phase("open")
                self.open_tries = 0
                return self._run_phase(state, b, q, g)
            return a

        if ph == "open":
            if g < 0.5:
                self._set_phase("retreat")
                return self._run_phase(state, b, q, g)
            self.open_tries = getattr(self, "open_tries", 0) + 1
            if self.open_tries > 2:
                self.open_tries = 0
                self.spot_fail = getattr(self, "spot_fail", 0) + 1
                if self.spot_fail >= 3:
                    self.spot_fail = 0
                    sx, sy = self.place_pos
                    f = self._footprint(state, obj)
                    ny = sy + 0.16 if sy + 0.16 <= self.ty + self.thy - f[1] else sy - 0.16
                    nx = float(np.clip(self.tx + (0.06 if sx < self.tx else -0.06),
                                       self.tx - self.thx + f[0] + 0.08,
                                       self.tx + self.thx - f[0]))
                    self.spot_of[obj.name] = (nx, float(np.clip(
                        ny, self.ty - self.thy + f[1], self.ty + self.thy - f[1])))
                    self._set_phase("drive_place")
                    self._plan_drive_place(state, b, obj)
                    return self._run_phase(state, b, q, g)
                self.place_gap = max(0.0002, self.place_gap - 0.0015)
                self._set_phase("lower")
                return self._run_phase(state, b, q, g)
            a = np.zeros(11)
            a[10] = 1.0
            return a

        if ph == "retreat":
            ee = self._ee(b, q)
            tgt = np.array([ee[0], ee[1], ee[2] + 0.12])
            a, done = self._arm_goto(b, q, tgt, yaw=self.grasp_yaw, res=0.12)
            if a is None or done:
                self.phase = "done"
                self.cur = None
                return np.zeros(11)
            return a

        self.phase = "done"
        self.cur = None
        return np.zeros(11)

    def _plan_drive_place(self, state, b, obj):
        f = self._footprint(state, obj)
        sx, sy = self.spot_of.get(obj.name, (self.tx, self.ty))
        sx = float(np.clip(sx, self.tx - self.thx + f[0], self.tx + self.thx - f[0]))
        sy = float(np.clip(sy, self.ty - self.thy + f[1], self.ty + self.thy - f[1]))
        self.place_pos = (sx, sy)
        gx = self.tx - self.thx - BASE_CLEAR - 0.01
        gy = float(np.clip(sy, self.ty - self.thy, self.ty + self.thy))
        self.place_rot = math.atan2(sy - gy, sx - gx)
        self.path = self._base_path(b[:2], np.array([gx, gy]))
