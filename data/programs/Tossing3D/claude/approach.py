import numpy as np
import robot

MX = 0.122      # arm mount x offset from base origin
T_TOOL = 0.171  # tool offset to grasp point
MZ = 0.394      # arm mount height
DT = 0.1
GX = 0.42       # nominal grasp x in arm frame
G = 9.81
ZRIM = 0.20     # bin rim height used as landing plane
BASE_X_MAX = 0.95

QREL = np.array([0.0, -0.105, np.pi, 0.259, 0.0, 0.051, np.pi / 2])
QDIR = np.array([0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0])
VCARR = 1.8
BACK = 0.3
QCOCK = QREL - QDIR * BACK
D0 = 2.45
YBIAS = -0.007
XBIAS = -0.13
RELK = 1
# calibrated: landing x offset from base origin vs swing speed w
WTAB2 = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
DTAB2 = np.array([1.10, 1.634, 2.130, 2.657, 3.210, 3.764, 4.277])
RDOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
YAW_OFF = 0.0
NEIGH_CONV = 0.0


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])



def _zeros():
    return np.zeros(18, dtype=np.float32)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    # ---------- state helpers ----------
    def _robot(self, s):
        o = s.get_object_from_name('robot')
        b = np.array([float(s.get(o, f)) for f in
                      ('pos_base_x', 'pos_base_y', 'pos_base_rot')])
        q = np.array([float(s.get(o, 'pos_arm_joint%d' % (i + 1))) for i in range(7)])
        return b, q

    def _pos(self, s, name):
        o = s.get_object_from_name(name)
        return np.array([float(s.get(o, f)) for f in ('x', 'y', 'z')])

    def _pv(self, s, name):
        o = s.get_object_from_name(name)
        return (np.array([float(s.get(o, f)) for f in ('x', 'y', 'z')]),
                np.array([float(s.get(o, f)) for f in ('vx', 'vy', 'vz')]))

    # ---------- lifecycle ----------
    def reset(self, state, info):
        names = sorted([n for n in state.get_object_names() if n.startswith('cube_')])
        self.cubes = names
        bins = sorted([n for n in state.get_object_names()
                       if n.startswith('bin')])
        if not bins:
            bins = sorted([o.name for o in state.get_objects(
                state.get_object_from_name(names[0]).type)
                if not o.name.startswith('cube')]) if names else []
        self.binname = bins[0] if bins else None
        self.binpos = (self._pos(state, self.binname)
                       if self.binname else np.array([3.0, 0.0, 0.0]))
        self.misses = {}
        self.todo = list(names)
        self.phase = 'pick'
        self.timer = 0
        self.qt = None
        self.grip = 0.0
        self.cur = None
        self.attempt = 0

    # ---------- low level ----------
    def _drive(self, b, tx, ty, tyaw=0.0):
        a = _zeros()
        a[0] = np.clip(tx - b[0], -0.1, 0.1)
        a[1] = np.clip(ty - b[1], -0.1, 0.1)
        er = (tyaw - b[2] + np.pi) % (2 * np.pi) - np.pi
        a[2] = np.clip(er, -0.1, 0.1)
        a[10] = self.grip
        return a, (abs(tx - b[0]) < 0.005 and abs(ty - b[1]) < 0.005 and abs(er) < 0.01)

    def _arm(self, q, qt, rate=0.1, kv=6.0, vmax=10.0, vcart=None):
        e = qt - q
        dq = np.clip(e, -rate, rate)
        vc = np.clip(kv * e, -vmax, vmax)
        if vcart is not None:
            Jm = robot.jac(q, T_TOOL)[:3]
            sp = float(np.linalg.norm(Jm @ dq)) / DT
            if sp > vcart:
                dq = dq * (vcart / sp)
            sp2 = float(np.linalg.norm(Jm @ vc))
            if sp2 > vcart:
                vc = vc * (vcart / sp2)
        a = _zeros()
        a[3:10] = dq
        a[11:18] = vc
        a[10] = self.grip
        return a, float(np.max(np.abs(e))) < 0.012

    # ---------- main ----------
    def get_action(self, state):
        b, q = self._robot(state)
        self.timer += 1
        ph = self.phase
        f = getattr(self, '_ph_' + ph)
        return f(state, b, q)

    def _go(self, ph):
        self.phase = ph
        self.timer = 0

    # phases
    def _ph_pick(self, state, b, q):
        if not self.todo:
            self._go('done')
            return _zeros()
        # choose nearest remaining cube on the robot's side
        best = None
        for n in self.todo:
            p = self._pos(state, n)
            if p[0] > 1.25:      # already across the barrier / in bin
                continue
            d = np.hypot(p[0] - b[0], p[1] - b[1])
            if best is None or d < best[0]:
                best = (d, n)
        if best is None:
            self.todo = []
            self._go('done')
            return _zeros()
        self.cur = best[1]
        self.grip = 0.0
        self.qt = None
        self._go('drive_grasp')
        return _zeros()

    def _ph_drive_grasp(self, state, b, q):
        c = self._pos(state, self.cur)
        a, done = self._drive(b, c[0] - MX - GX, c[1], 0.0)
        if done or self.timer > 140:
            self._go('pregrasp')
        return a

    def _ph_pregrasp(self, state, b, q):
        if self.qt is None:
            c = self._pos(state, self.cur)
            gx = c[0] - b[0] - MX
            gy = c[1] - b[1]
            zg = c[2] - 0.010 - MZ
            o = state.get_object_from_name(self.cur)
            yaw = 2.0 * np.arctan2(float(state.get(o, 'qz')),
                                   float(state.get(o, 'qw')))
            psi = (yaw + YAW_OFF + np.pi / 4) % (np.pi / 2) - np.pi / 4
            # if another cube is close, pick the 90-deg variant whose finger
            # axis is most perpendicular to the neighbour direction
            nb = None
            nbd = 0.16
            for nm in self.cubes:
                if nm == self.cur:
                    continue
                pp = self._pos(state, nm)
                dd = float(np.hypot(pp[0] - c[0], pp[1] - c[1]))
                if dd < nbd:
                    nbd = dd
                    nb = np.arctan2(pp[1] - c[1], pp[0] - c[0])
            if nb is not None:
                cands = [psi, psi + np.pi / 2]
                psi = max(cands,
                          key=lambda u: abs(np.sin(u + NEIGH_CONV - nb)))
            R = _rz(psi) @ RDOWN
            self.qgrasp = robot.ik(np.array([gx, gy, zg]), R, q, tool=T_TOOL)
            self.qt = robot.ik(np.array([gx, gy, zg + 0.16]), R,
                               self.qgrasp, tool=T_TOOL)
            self.qlift = robot.ik(np.array([gx, gy, zg + 0.32]), R,
                                  self.qgrasp, tool=T_TOOL)
        a, done = self._arm(q, self.qt, rate=0.1, kv=6.0, vmax=10.0)
        if done or self.timer > 60:
            self.qt = None
            self._go('descend')
        return a

    def _ph_descend(self, state, b, q):
        a, done = self._arm(q, self.qgrasp, rate=0.07, kv=3.5, vmax=1.8)
        if done or self.timer > 60:
            self.qt = None
            self._go('close')
        return a

    def _ph_close(self, state, b, q):
        self.grip = 1.0
        a = _zeros()
        a[10] = 1.0
        if self.timer > 6:
            self.qt = None
            self._go('lift')
        return a

    def _ph_lift(self, state, b, q):
        a, done = self._arm(q, self.qlift, rate=0.13, kv=4.0, vmax=4.0, vcart=VCARR)
        if done or self.timer > 60:
            self.qt = None
            p = self._pos(state, self.cur)
            if p[2] < 0.15:
                self.attempt += 1
                self.grip = 0.0
                self._go('pick')
            else:
                self._plan_throw(state, b)
                self._go('cock')
        return a

    def _plan_throw(self, state, b):
        binp = self._pos(state, self.binname)
        self.binpos = binp
        self.qc = QCOCK.copy()
        aim = binp[0] + XBIAS
        bx = float(np.clip(aim - D0, -0.8, BASE_X_MAX))
        D = aim - bx
        self.w = float(np.interp(D, DTAB2, WTAB2))
        self.base_throw = (bx, binp[1] + YBIAS)

    def _ph_cock(self, state, b, q):
        a, done = self._arm(q, self.qc, rate=0.13, kv=4.0, vmax=4.0, vcart=VCARR)
        dr, ddone = self._drive(b, self.base_throw[0], self.base_throw[1], 0.0)
        a[0:3] = dr[0:3]
        if self._pos(state, self.cur)[2] < 0.15 and self.timer > 5:
            self.grip = 0.0
            self.attempt += 1
            self._go('pick')
            return _zeros()
        if done or self.timer > 220:
            self._go('drive_throw')
        return a

    def _ph_drive_throw(self, state, b, q):
        a, done = self._drive(b, self.base_throw[0], self.base_throw[1], 0.0)
        aa, _ = self._arm(q, self.qc, rate=0.06, kv=2.0, vmax=0.6)
        if self.timer > 40 and self._pos(state, self.cur)[2] < 0.15:
            self.grip = 0.0
            self.attempt += 1
            self._go('pick')
            return _zeros()
        a[3:10] = aa[3:10]
        a[11:18] = aa[11:18]
        if (done and float(np.max(np.abs(self.qc - q))) < 0.008) or self.timer > 120:
            self.released = False
            self.relcount = 0
            self._go('swing')
        return a

    def _ph_swing(self, state, b, q):
        a = _zeros()
        a[3:10] = np.clip(QDIR * self.w * DT, -0.1, 0.1)
        a[11:18] = QDIR * self.w
        k = self.timer - 1
        a[10] = 1.0 if k < RELK else 0.0
        if k >= RELK + 3:
            self.grip = 0.0
            self.qt = None
            self._go('settle')
        return a

    def _ph_settle(self, state, b, q):
        self.grip = 0.0
        a = _zeros()
        if self.timer > 8:
            if self.cur in self.todo:
                p = self._pos(state, self.cur)
                d = float(np.hypot(p[0] - self.binpos[0], p[1] - self.binpos[1]))
                self.misses = getattr(self, 'misses', {})
                n = self.misses.get(self.cur, 0)
                if d < 0.12 or p[0] > 1.25 or n >= 2:
                    self.todo.remove(self.cur)
                else:
                    self.misses[self.cur] = n + 1
            self.attempt = 0
            self._go('pick')
        return a

    def _ph_done(self, state, b, q):
        a = _zeros()
        a[10] = self.grip
        return a
