import numpy as np
import armkin as ak
import planar

JN = ['pos_arm_joint%d' % i for i in range(1, 8)]
BKEYS = ('pos_base_x', 'pos_base_y', 'pos_base_rot')
MX, MY, MZ = 0.111, 0.004, 0.358
LAT = planar.LAT
XP_PICK = 0.45
XI = 0.62
BOARD_OFF = 0.542          # top board top surface above cupboard z
CARRY_OFF = BOARD_OFF + 0.040
PLACE_X = 1.50
STAGE_X = 1.16


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ---------- state helpers ----------
    def _rob(self, s):
        return s.get_object_from_name('robot')

    def _q(self, s):
        r = self._rob(s)
        return np.array([float(s.get(r, j)) for j in JN])

    def _b(self, s):
        r = self._rob(s)
        return np.array([float(s.get(r, k)) for k in BKEYS])

    def _cubes(self, s):
        out = []
        for n in sorted(s.get_object_names()):
            o = s.get_object_from_name(n)
            feats = s.type_features.get(o.type, [])
            if 'bb_x' in feats and 'vx' in feats:
                out.append((n, o))
        return out

    def _cpos(self, s, o):
        return np.array([float(s.get(o, 'x')), float(s.get(o, 'y')), float(s.get(o, 'z'))])

    def reset(self, state, info):
        self.t = 0
        self.I = np.zeros(7)
        self.qc = self._q(state)
        self.bd = self._b(state).copy()
        self.grip = 0.0
        self.pose = None
        # cupboard pose
        self.dz = 0.0
        self.cup_x, self.cup_y = 1.5, 0.0
        fixtures = []
        for n in sorted(state.get_object_names()):
            o = state.get_object_from_name(n)
            feats = state.type_features.get(o.type, [])
            if 'qw' in feats and 'bb_x' not in feats and 'pos_gripper' not in feats:
                fixtures.append((n, o))
        pick = None
        for n, o in fixtures:
            if 'cupboard' in n or 'shelf' in n:
                pick = o
                break
        if pick is None and fixtures:
            pick = fixtures[0][1]
        if pick is not None:
            self.dz = float(state.get(pick, 'z'))
            self.cup_x = float(state.get(pick, 'x'))
            self.cup_y = float(state.get(pick, 'y'))
        self.board = self.dz + BOARD_OFF
        self.carry_z = self.dz + CARRY_OFF
        cubes = self._cubes(state)
        self.names = [n for n, _ in cubes]
        # order: nearest first
        base = self.bd[:2]
        pos = {n: self._cpos(state, o) for n, o in cubes}
        self.names.sort(key=lambda n: np.linalg.norm(pos[n][:2] - base))
        yslots = [-0.17, 0.17, -0.085, 0.085, 0.0]
        px = self.cup_x
        slots = []
        for depth in (px, px - 0.11, px - 0.22):
            for y in yslots:
                slots.append((depth, self.cup_y + y))
        self.slots = {}
        for i, n in enumerate(self.names):
            self.slots[n] = slots[i % len(slots)]
        self.idx = 0
        self.phase = 0
        self.pt = 0
        self.attempts = 0
        self.done = False

    # ---------- control ----------
    def _set_arm(self, xp, zp, s):
        key = (round(xp, 5), round(zp, 5), round(s, 5))
        if self.pose == key:
            return True
        q = planar.planar_ik(xp, zp, s, seed=(self.qc[1], self.qc[3]),
                             cur=(self.qc[1], self.qc[3], self.qc[5]))
        if q is None:
            return False
        self.qc = q
        self.pose = key
        return True

    def _base_for_tool(self, X, Y, xp):
        return np.array([X - (xp + MX), Y - (LAT + MY), 0.0])

    def _converged(self, state, qtol=0.02, btol=0.02):
        return (np.linalg.norm(self._q(state) - self.qc) < qtol and
                np.linalg.norm(self._b(state) - self.bd) < btol)

    def _tool_err(self, state):
        """position error of the tool point (world) vs commanded, plus base error."""
        q = self._q(state)
        pa = ak.fk_pos(q, ak.TOOL_OFFSET)
        pc = ak.fk_pos(self.qc, ak.TOOL_OFFSET)
        b = self._b(state)
        eb = self.bd[:2] - b[:2]
        return float(np.linalg.norm(pa - pc)), float(np.linalg.norm(eb))

    def _ok(self, state, ptol, btol):
        pe, be = self._tool_err(state)
        return pe < ptol and be < btol

    def _action(self, state):
        a = np.zeros(11, dtype=np.float32)
        q = self._q(state)
        e = self.qc - q
        self.I = np.clip(self.I + 0.3 * e, -0.1, 0.1)
        a[3:10] = np.clip(2.0 * e + self.I, -0.1, 0.1)
        b = self._b(state)
        eb = self.bd - b
        eb[2] = (eb[2] + np.pi) % (2 * np.pi) - np.pi
        a[0] = np.clip(1.03 * eb[0], -0.1, 0.1)
        a[1] = np.clip(1.03 * eb[1], -0.1, 0.1)
        a[2] = np.clip(0.95 * eb[2], -0.1, 0.1)
        a[10] = self.grip
        return a

    # ---------- plan ----------
    def _advance(self, state):
        """Run the state machine; sets self.qc/self.bd/self.grip."""
        if self.idx >= len(self.names):
            # second pass: retry any cube that is not on the shelf
            if (1000 - self.t) > 340:
                for k, nm in enumerate(self.names):
                    ob = state.get_object_from_name(nm)
                    p = self._cpos(state, ob)
                    if not (p[2] > self.board - 0.06 and p[0] > self.cup_x - 0.17):
                        self.idx = k
                        self.phase = 0
                        self.pt = 0
                        self.attempts = 0
                        break
                else:
                    self.done = True
                    return
                if self.idx >= len(self.names):
                    self.done = True
                    return
            else:
                self.done = True
                return
        name = self.names[self.idx]
        cub = state.get_object_from_name(name)
        cp = self._cpos(state, cub)
        ph = self.phase
        if ph == 0:                      # go above cube
            self.bd = self._base_for_tool(cp[0], cp[1], XP_PICK)
            self._set_arm(XP_PICK, 0.16 - MZ, np.pi)
            self.grip = 0.0
            self._next(state, 150, lambda st: self._ok(st, 0.05, 0.025))
        elif ph == 1:                    # fine align + descend
            self.bd = self._base_for_tool(cp[0], cp[1], XP_PICK)
            self._set_arm(XP_PICK, 0.021 - MZ, np.pi)
            self.grip = 0.0
            self._next(state, 60, lambda st: self._ok(st, 0.006, 0.006))
        elif ph == 2:                    # close gripper
            self.grip = 1.0
            self._next(state, 15, None)
        elif ph == 3:                    # lift straight up
            self._set_arm(XP_PICK, 0.26 - MZ, np.pi)
            self.grip = 1.0
            self._next(state, 50, lambda st: self._ok(st, 0.02, 0.05))
        elif ph == 4:                    # check grasp
            if cp[2] < 0.12:
                self.attempts += 1
                self.phase = 0 if self.attempts < 3 else 0
                self.pt = 0
                if self.attempts >= 4:
                    self.idx += 1
                    self.phase = 0
                    self.attempts = 0
                return
            self.phase = 5
            self.pt = 0
        elif ph == 5:                    # reorient to horizontal carry (base back)
            self._set_arm(XI, self.carry_z - MZ, np.pi / 2)
            self.bd = self._base_for_tool(self.cup_x - 0.60, self.slots[name][1], XI)
            self.grip = 1.0
            self._next(state, 150, lambda st: self._ok(st, 0.02, 0.02))
        elif ph == 6:                    # drive to stage then insert
            self.bd = self._base_for_tool(self.slots[name][0], self.slots[name][1], XI)
            self.grip = 1.0
            self._next(state, 45, lambda st: self._ok(st, 0.05, 0.01))
        elif ph == 7:                    # release
            self.grip = 0.0
            self._next(state, 14, None)
        elif ph == 8:                    # retract base
            self.bd = self._base_for_tool(self.cup_x - 0.48, self.slots[name][1], XI)
            self.grip = 0.0
            self._next(state, 28, lambda st: self._ok(st, 0.2, 0.02))
        elif ph == 9:
            placed = (cp[2] > self.board - 0.06) and (cp[0] > self.cup_x - 0.17)
            budget_left = 1000 - self.t
            if (not placed) and self.attempts < 3 and budget_left > 340:
                self.attempts += 1
                self.phase = 0
                self.pt = 0
                return
            self.idx += 1
            self.phase = 0
            self.pt = 0
            self.attempts = 0
            return
        else:
            self.done = True

    def _next(self, state, maxsteps, cond):
        self.pt += 1
        if self.pt >= maxsteps or (cond is not None and self.pt > 3 and cond(state)):
            self.phase += 1
            self.pt = 0

    def get_action(self, state):
        self.t += 1
        if not self.done:
            self._advance(state)
        return self._action(state)
