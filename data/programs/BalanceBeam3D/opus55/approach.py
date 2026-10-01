import numpy as np
from kin import Kin, _rz

LB, SW, SB1, SB2 = 0, 38, 54, 70


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class GeneratedApproach:
    R_REACH = 0.5
    Z_CARRY = 0.11
    Z_PRE = 0.045
    SIDE = 0.05
    SIDE_MIN = 0.045
    DESC_TOL = 0.025
    CLOSE_T = 2
    REL_T = 1
    LAST_ACROSS = 0.02
    Z_MOVE = 0.06
    SIDE_MAX = 0.07

    def __init__(self, action_space, observation_space, primitives):
        self.kin = Kin(mount=(0.113, 0.0, 0.36), tool=0.128)
        seed = np.array([0, 1.4, np.pi, -1.8, 0, 0.1, np.pi / 2])
        R0 = _rz(np.pi / 2)
        z0 = np.zeros(3)
        self.qC, _ = self.kin.ik(z0, seed, np.array([self.R_REACH, 0, self.Z_CARRY]), R0, iters=300)
        self.qP, _ = self.kin.ik(z0, self.qC, np.array([self.R_REACH, 0, self.Z_PRE]), R0, iters=300)
        self.qG, _ = self.kin.ik(z0, self.qP, np.array([self.R_REACH, 0, 0.005]), R0, iters=300)

    # ---------------------------------------------------------------- plan
    def reset(self, state, info):
        s = np.asarray(state, dtype=float)
        self.grip = 0.0
        self.placed = []
        self.obj = None
        self.phase = "approach"
        self.t = 0
        self.qtarget = None
        self.base_t = None

    # ------------------------------------------------------------ helpers
    def best_pose(self, s, p, yaw, nsym=4, avoid=()):
        best, bc = None, 1e9
        for k in range(nsym):
            th = wrap(yaw + k * 2 * np.pi / nsym)
            h = np.array([np.cos(th), np.sin(th)])
            pos = p[:2] - self.R_REACH * h
            cost = max(np.linalg.norm(pos - s[16:18]) / 0.087, abs(wrap(th - s[18])) / 0.1)
            for q, half in avoid:
                d = q[:2] - p[:2]
                a, b = abs(np.dot(d, h)), abs(h[0] * d[1] - h[1] * d[0])
                if a < half + 0.018 and b < 0.055 + half:
                    cost += 1000
            if cost < bc:
                best, bc = np.array([pos[0], pos[1], th]), cost
        return best

    def grasp_base(self, s, obj):
        avoid = [(s[o:o + 2], s[o + 13] / 2) for o in (LB, SB1, SB2) if o != obj]
        return self.best_pose(s, s[obj:obj + 2], quat_yaw(s[obj + 3:obj + 7]), avoid=avoid)

    def near_seesaw(self, tool, margin):
        c, axis = self.sw_c, self.sw_axis
        rel = tool[:2] - c[:2]
        along, across = abs(np.dot(rel, axis[:2])), abs(axis[0] * rel[1] - axis[1] * rel[0])
        return along < 0.18 + margin and across < 0.035 + margin

    def on_seesaw(self, s, o):
        c, axis = self.seesaw(s)
        rel = s[o:o + 2] - c[:2]
        along, across = abs(np.dot(rel, axis[:2])), abs(axis[0] * rel[1] - axis[1] * rel[0])
        return s[o + 2] > 0.035 and along < 0.17 and across < 0.04

    def next_task(self, s):
        todo = [o for o in (LB, SB1, SB2) if not self.on_seesaw(s, o)]
        self.placed = [o for o in (LB, SB1, SB2) if self.on_seesaw(s, o)]
        if not todo:
            return None
        if LB in todo:
            return LB
        return min(todo, key=lambda o: np.linalg.norm(s[o:o + 2] - s[16:18]))

    def seesaw(self, s):
        c = s[SW:SW + 3].copy()
        yaw = quat_yaw(s[SW + 3:SW + 7])
        return c, np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def place_target(self, s, obj):
        c, axis = self.seesaw(s)
        m = lambda o: s[o + 13] ** 3
        torque = sum(m(o) * np.dot(s[o:o + 3] - c, axis) for o in self.placed)
        if not self.placed:
            off = 0.0
        else:
            n_left = 3 - len(self.placed)
            if n_left > 1:
                # place on side opposite to current torque, moderate offset
                sgn = -np.sign(torque) if abs(torque) > 1e-9 else 1.0
                off = sgn * self.SIDE
            else:
                off = -torque / m(obj)
                sgn = np.sign(off) if abs(off) > 1e-9 else 1.0
                off = sgn * np.clip(abs(off), self.SIDE_MIN, self.SIDE_MAX)
        return c + off * axis

    def place_base(self, s, tgt):
        c, axis = self.seesaw(s)
        return self.best_pose(s, tgt, np.arctan2(axis[1], axis[0]))

    def act(self, s, base_t, q_t, grip):
        eb = base_t - s[16:19]
        eb[2] = wrap(eb[2])
        eq = wrap(q_t - s[19:26])
        a = np.zeros(11)
        a[0:3] = np.clip(eb, -0.1, 0.1)
        a[3:10] = np.clip(eq, -0.1, 0.1)
        a[10] = grip
        return a.astype(np.float32), eb, eq

    def get_action(self, state):
        s = np.asarray(state, dtype=float)
        self.t += 1
        self.sw_c, self.sw_axis = self.seesaw(s)
        if self.phase == "approach" and self.obj is None:
            self.obj = self.next_task(s)
            self.base_t = None
        obj = self.obj
        if obj is None:
            self.phase = "approach"
            a, _, _ = self.act(s, s[16:19].copy(), self.qC, 0.0)
            return a
        ph = self.phase
        if ph == "approach":
            if self.base_t is None:
                self.base_t = self.grasp_base(s, obj)
            # re-target slowly with current block pose
            bt = self.grasp_base(s, obj)
            if np.linalg.norm(bt[:2] - self.base_t[:2]) > 0.003:
                self.base_t = bt
            dist = np.linalg.norm(self.base_t[:2] - s[16:18])
            tool = self.kin.fk(s[16:19], s[19:26])[0]
            near_sw = self.near_seesaw(tool, 0.06)
            qt = self.qC if (dist > 0.15 or near_sw) else self.qP
            a, eb, eq = self.act(s, self.base_t, qt, 0.0)
            if np.abs(eb[:2]).max() < self.DESC_TOL and abs(eb[2]) < 0.04 and np.abs(eq).max() < 0.06 and not near_sw:
                self.set_phase("descend")
            return a
        if ph == "descend":
            if self.qtarget is None:
                p = s[obj:obj + 3].copy()
                p[2] = max(s[obj + 15] / 2 - 0.012, -0.005)
                yaw = quat_yaw(s[obj + 3:obj + 7])
                rel = (wrap(yaw - s[18]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
                self.qtarget, _ = self.kin.ik(self.base_t, self.qG, p, _rz(self.base_t[2] + rel + np.pi / 2))
            a, eb, eq = self.act(s, self.base_t, self.qtarget, 0.0)
            if (np.abs(eq).max() < 0.012 and np.abs(eb[:2]).max() < 0.006 and abs(eb[2]) < 0.01) or self.t > 40:
                self.set_phase("close")
            return a
        if ph == "close":
            a, eb, eq = self.act(s, self.base_t, self.qtarget, 1.0)
            if self.t >= self.CLOSE_T:
                self.set_phase("transport")
                if len(self.placed) >= 2:
                    # last block: success triggers once it is over the beam -> keep heading
                    h = np.array([np.cos(s[18]), np.sin(s[18])])
                    tgt = self.place_target(s, obj)
                    self.base_t = np.array([*(tgt[:2] - self.R_REACH * h), s[18]])
                else:
                    tgt = self.place_target(s, obj)
                    self.base_t = self.place_base(s, tgt)
            return a
        if ph == "transport":
            tool = self.kin.fk(s[16:19], s[19:26])[0]
            ztool = tool[2]
            if ztool < self.Z_CARRY - 0.012 and not self.lifted and (ztool < self.Z_MOVE or self.near_seesaw(tool, 0.12)):
                a, eb, eq = self.act(s, s[16:19].copy(), self.qC, 1.0)
                a[0:3] = 0.0
                return a
            if ztool >= self.Z_CARRY - 0.012:
                self.lifted = True
            held = np.linalg.norm(self.kin.fk(s[16:19], s[19:26])[0] - s[obj:obj + 3]) < 0.04
            if not held:
                self.set_phase("approach")
                self.obj = None
                a, eb, eq = self.act(s, s[16:19].copy(), self.qC, 0.0)
                return a
            a, eb, eq = self.act(s, self.base_t, self.qC, 1.0)
            if np.abs(eb[:2]).max() < 0.01 and abs(eb[2]) < 0.03 and np.abs(eq).max() < 0.03:
                self.set_phase("release")
            return a
        if ph == "release":
            a, eb, eq = self.act(s, self.base_t, self.qC, 0.0)
            if self.t >= self.REL_T:
                self.set_phase("approach")
                self.obj = None
            return a
        raise RuntimeError(ph)

    def set_phase(self, ph):
        self.phase = ph
        self.t = 0
        if ph == "descend":
            self.qtarget = None
        self.lifted = False
