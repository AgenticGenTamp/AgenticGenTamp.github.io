"""Policy for ConstrainedCupboard3DEnv (holonomic base + Kinova Gen3).

Key facts discovered empirically:
  * action = [dx, dy, dtheta (world frame, gain .87/.87/.99),
              7 arm joint deltas (gain .25, i.e. <=0.025 rad/step),
              gripper setpoint].
  * The arm is ~4x slower than the base, so the policy keeps the arm in a
    small set of canonical configurations and does all positioning with the
    holonomic base.
  * The cupboard is a row of 0.1 m wide bays; each `mujoco_fixture` marks one
    bay centre (x=2.0).  Rods are 0.3 x 0.03 x 0.03 and must be inserted
    lengthwise (along world x).
  * FK/IK model calibrated in kin.py / ctrl.py.
"""
import numpy as np

import kin
import ctrl


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    ALPHA = np.radians(90.0)   # gripper tilt below horizontal
    DX = 0.50                  # forward reach of the canonical config
    GRASP_OFF = 0.125          # grasp point offset from the rod centre (rear)
    Z_GRASP_UP = 0.000         # grasp above the rod centre
    Z_HOVER = 0.10             # hover height above the grasp point
    Z_CARRY = 0.09
    Z_PLACE = 0.036
    Z_FINAL = 0.036            # rod centre height on the bay base plate
    TARGET_DX = 0.10            # target rod centre x relative to the fixture x
    PRE_BACK = 0.28            # base standoff before insertion
    MAXSTEP_BASE = 0.085

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.rng = np.random.default_rng(0)
        self.verbose = False
        self.q_home = np.array([0.0, -0.3491, 3.1416, -2.5482, 0.0,
                                -0.8727, 1.5708])
        self.q_cfg = {}

    # ------------------------------------------------------------------
    def _tool_R(self, psi):
        """Canonical tool orientation for base yaw psi."""
        a = self.ALPHA
        z = np.array([np.cos(a), 0.0, -np.sin(a)])
        x = np.array([0.0, 1.0, 0.0])
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        return ctrl.rotz(psi) @ R

    def _cfg(self, key, z=None, dx=None):
        """Joint config placing the canonical grasp point at (dx, 0, z)."""
        if z is None:
            z = {'hover': 0.13, 'carry': self.Z_CARRY,
                 'place': self.Z_PLACE, 'final': self.Z_FINAL, 'grasp': 0.015}[key]
        if dx is None:
            dx = self.DX
        ck = (round(z, 3), round(dx, 3))
        if ck in self.q_cfg:
            return self.q_cfg[ck]
        R = self._tool_R(0.0)
        T = ctrl.pose_from([dx, 0.0, z - ctrl.MOUNT[2]], R)
        ref = self.q_ref if self.q_ref is not None else self.q_home
        best = None
        for k in range(60):
            s = ref if k == 0 else ref + self.rng.uniform(-2.0, 2.0, 7)
            qd, ep, er = kin.ik2(T, s, tool_z=ctrl.TOOL, q_lim=ctrl.SOFT,
                                 iters=80)
            if ep > 0.003 or er > 0.03:
                continue
            for j in (0, 2, 4, 6):
                qd[j] -= 2 * np.pi * np.round((qd[j] - ref[j]) / (2 * np.pi))
            if np.any(qd < ctrl.SOFT[0] - 1e-6) or np.any(qd > ctrl.SOFT[1] + 1e-6):
                continue
            c = np.max(np.abs(qd - ref))
            if best is None or c < best[0]:
                best = (c, qd)
            if c < 0.35:
                break
        q = None if best is None else best[1]
        self.q_cfg[ck] = q
        return q

    def _base_for(self, grasp_xyz, psi):
        """Base (x, y) placing the canonical grasp point at grasp_xyz."""
        off = ctrl.rotz(psi) @ (ctrl.MOUNT + np.array([self.DX, 0.0, 0.0]))
        return np.array([grasp_xyz[0] - off[0], grasp_xyz[1] - off[1], psi])

    # ------------------------------------------------------------------
    def reset(self, state, info):
        self.t = 0
        self.grip = 0.0
        self.pstep = 0
        self.rod_names = sorted([n for n in state.get_object_names()
                                 if n.startswith('cuboid')],
                                key=lambda s: int(s.split('_')[1]))
        bays = []
        for n in state.get_object_names():
            if n.startswith('cuboid') or n == 'robot':
                continue
            o = state.get_object_from_name(n)
            bays.append((state.get(o, 'x'), state.get(o, 'y')))
        bays.sort(key=lambda p: p[1])
        self.bays = bays
        self.placed = set()
        self.order, self.assign = self._plan(state)
        self.cur = 0
        self.stage = 'to_pick'
        self.T_goal = None
        self.q_null = None
        self.base_goal = None
        self.grasp_rel = None
        self.q_ref = None
        self.q_cfg = {}
        self.q_ref = self._cfg('carry')
        self.q_null = self.q_ref

    def _plan(self, state):
        rods = []
        for i, n in enumerate(self.rod_names):
            o = state.get_object_from_name(n)
            rods.append((state.get(o, 'y'), i))
        rods.sort()
        nb, nr = len(self.bays), len(rods)
        assign = {}
        for k, (_, i) in enumerate(rods):
            assign[i] = int(round(k * (nb - 1) / (nr - 1))) if nr > 1 else nb // 2
        order = [i for _, i in rods]
        return order, assign

    # ------------------------------------------------------------------
    def _rod(self, state, name):
        o = state.get_object_from_name(name)
        p = np.array([state.get(o, 'x'), state.get(o, 'y'), state.get(o, 'z')])
        qw, qz = state.get(o, 'qw'), state.get(o, 'qz')
        yaw = np.arctan2(2 * qw * qz, 1 - 2 * qz * qz)
        return p, yaw

    def _rod_axis(self, yaw):
        """Unit long axis with positive x component, and its heading."""
        u = np.array([-np.sin(yaw), np.cos(yaw)])
        if u[0] < 0:
            u = -u
        return u, np.arctan2(u[1], u[0])

    def _act(self, dbase=None, dq=None, grip=None):
        a = np.zeros(11, dtype=np.float32)
        if dbase is not None:
            d = np.asarray(dbase, dtype=float).copy()
            n = np.linalg.norm(d[:2])
            if n > self.MAXSTEP_BASE:
                d[:2] *= self.MAXSTEP_BASE / n
            a[:3] = np.clip(d / np.array([ctrl.BASE_GAIN, ctrl.BASE_GAIN, 0.994]),
                            -0.1, 0.1)
        if dq is not None:
            a[3:10] = np.clip(np.asarray(dq) / ctrl.ARM_GAIN, -0.1, 0.1)
        a[10] = self.grip if grip is None else grip
        return np.clip(a, self.action_space.low, self.action_space.high)

    def _base_err(self, b):
        if self.base_goal is None:
            return np.zeros(3)
        e = np.array(self.base_goal, dtype=float) - b
        e[2] = wrap(e[2])
        return e

    def _set_goal(self, key, z=None, dx=None):
        if z is None:
            z = {'hover': 0.13, 'carry': self.Z_CARRY,
                 'place': self.Z_PLACE, 'final': self.Z_FINAL, 'grasp': 0.015}[key]
        if dx is None:
            dx = self.DX
        R = self._tool_R(0.0)
        self.T_goal = ctrl.pose_from([dx, 0.0, z - ctrl.MOUNT[2]], R)
        qn = self._cfg(key, z, dx)
        if qn is not None:
            self.q_null = qn

    def _enter(self, stage):
        self.stage = stage
        self.pstep = 0

    def _arm_cmd(self, q, tol=0.004):
        """Closed-loop Cartesian servo toward self.T_goal (arm frame)."""
        if self.T_goal is None:
            return np.zeros(7), True
        T, J = kin.fk_jac(q, ctrl.TOOL)
        ep = self.T_goal[:3, 3] - T[:3, 3]
        er = kin._log_so3(self.T_goal[:3, :3] @ T[:3, :3].T)
        npe, nre = np.linalg.norm(ep), np.linalg.norm(er)
        done = npe < tol and nre < 0.02
        v = ep.copy()
        if npe > 0.03:
            v *= 0.03 / npe
        w = er.copy()
        if nre > 0.15:
            w *= 0.15 / nre
        e = np.concatenate([v, 0.5 * w])
        Jw = np.vstack([J[:3], 0.5 * J[3:]])
        JT = Jw.T @ np.linalg.solve(Jw @ Jw.T + 2e-3 * np.eye(6), np.eye(6))
        dq = JT @ e
        if self.q_null is not None:
            dn = np.clip(self.q_null - q, -0.5, 0.5)
            dq = dq + 0.15 * (dn - JT @ (Jw @ dn))
        lo = ctrl.SOFT[0] - q
        hi = ctrl.SOFT[1] - q
        dq = np.clip(dq, np.minimum(lo, 0), np.maximum(hi, 0))
        return dq, done

    # ------------------------------------------------------------------
    def get_action(self, state):
        self.t += 1
        self.pstep += 1
        try:
            q, b = ctrl.read_robot(state)
            a = getattr(self, '_st_' + self.stage)(state, q, b)
            a = np.asarray(a, dtype=np.float32)
            if not np.all(np.isfinite(a)):
                raise ValueError
            return np.clip(a, self.action_space.low, self.action_space.high)
        except Exception:
            try:
                self.cur += 1
                self._enter('to_pick')
            except Exception:
                pass
            return np.zeros(11, dtype=np.float32)

    def _rodname(self):
        return self.rod_names[self.order[self.cur]]

    def _target(self):
        bx, by = self.bays[self.assign[self.order[self.cur]]]
        return np.array([bx + self.TARGET_DX, by, self.Z_PLACE])

    # ---- stages ------------------------------------------------------
    def _st_done(self, state, q, b):
        return self._act()

    def _st_to_pick(self, state, q, b):
        if self.cur >= len(self.order):
            self._enter('done')
            return self._act()
        p, yaw = self._rod(state, self._rodname())
        u, psi = self._rod_axis(yaw)
        gp = np.array([p[0] - u[0] * self.GRASP_OFF,
                       p[1] - u[1] * self.GRASP_OFF, p[2]])
        self.base_goal = self._base_for(gp, psi)
        self._set_goal('hover', p[2] + self.Z_HOVER)
        dq, arm_ok = self._arm_cmd(q)
        eb = self._base_err(b)
        if (arm_ok and np.linalg.norm(eb[:2]) < 0.004 and abs(eb[2]) < 0.01) \
                or self.pstep > 220:
            self._enter('descend')
        return self._act(eb, dq, 0.0)

    def _st_descend(self, state, q, b):
        p, yaw = self._rod(state, self._rodname())
        self._set_goal('grasp', p[2] + self.Z_GRASP_UP)
        dq, ok = self._arm_cmd(q)
        eb = self._base_err(b)
        if ok or self.pstep > 60:
            self._enter('close')
        return self._act(eb, dq, 0.0)

    def _st_close(self, state, q, b):
        self.grip = 1.0
        if self.pstep >= 12:
            p, yaw = self._rod(state, self._rodname())
            o = state.get_object_from_name(self._rodname())
            Rr = kin.quat_to_mat([state.get(o, f) for f in
                                  ('qw', 'qx', 'qy', 'qz')])
            Trod = np.eye(4)
            Trod[:3, :3] = Rr
            Trod[:3, 3] = p
            Tee = ctrl.ee_world(q, b)
            self.grasp_rel = np.linalg.inv(Tee) @ Trod
            self._enter('lift')
        return self._act(None, None, 1.0)

    def _st_lift(self, state, q, b):
        self._set_goal('carry')
        dq, ok = self._arm_cmd(q)
        if ok or self.pstep > 30:
            self._enter('to_place')
        return self._act(None, dq, 1.0)

    def _st_to_place(self, state, q, b):
        tp = self._target()
        gp = np.array([tp[0] - self.GRASP_OFF, tp[1], self.Z_PLACE])
        bg = self._base_for(gp, 0.0)
        self.base_goal = [bg[0] - self.PRE_BACK, bg[1], 0.0]
        self._set_goal('place')
        dq, arm_ok = self._arm_cmd(q)
        eb = self._base_err(b)
        if (arm_ok and np.linalg.norm(eb[:2]) < 0.004 and abs(eb[2]) < 0.01) \
                or self.pstep > 220:
            self._enter('insert')
        return self._act(eb, dq, 1.0)

    def _st_insert(self, state, q, b):
        tp = self._target()
        gp = np.array([tp[0] - self.GRASP_OFF, tp[1], self.Z_PLACE])
        bg = self._base_for(gp, 0.0)
        p, yaw = self._rod(state, self._rodname())
        # push until the rod itself reaches the target depth
        if p[0] < tp[0] - 0.004 and b[0] < min(bg[0] + 0.22, 1.55):
            self.base_goal = [min(b[0] + 0.03, bg[0] + 0.22, 1.55), bg[1], 0.0]
        else:
            self.base_goal = [b[0], bg[1], 0.0]
        eb = self._base_err(b)
        d = eb.copy()
        d[:2] = np.clip(d[:2], -0.03, 0.03)
        dq, _ = self._arm_cmd(q)
        if (p[0] >= tp[0] - 0.004 and abs(eb[1]) < 0.004) or self.pstep > 60:
            self._enter('release')
        return self._act(d, dq, 1.0)

    def _st_lower(self, state, q, b):
        self._set_goal('final')
        dq, ok = self._arm_cmd(q)
        if ok or self.pstep > 25:
            self._enter('release')
        return self._act(None, dq, 1.0)

    def _st_release(self, state, q, b):
        self.grip = 0.0
        if self.pstep >= 7:
            self._enter('back')
        return self._act(None, None, 0.0)

    def _st_back(self, state, q, b):
        self.base_goal = [b[0] - 0.30 if self.pstep == 1 else self.base_goal[0],
                          b[1], 0.0] if self.pstep == 1 else self.base_goal
        eb = self._base_err(b)
        if np.linalg.norm(eb[:2]) < 0.01 or self.pstep > 20:
            self.cur += 1
            self._enter('to_pick')
        return self._act(eb, None, 0.0)
