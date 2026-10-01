"""SweepSimple3D approach: fast pick-and-place of every cube into the target
box on the floor north of the kitchen island.

Target (found empirically): x in [0.075, 0.925], y in [0.92, 1.43], z < 0.1,
shared by all cubes; episode terminates when all cubes are inside.

Strategy: the arm holds a fixed "gripper down" pose at reach D in front of the
base; xy positioning and cube-yaw alignment are done with the (fast) holonomic
base.  Only small joint motions are needed for descend / lift.
"""
import numpy as np
import kin

J = ['pos_arm_joint%d' % i for i in range(1, 8)]
BOX = (0.075, 0.925, 0.92, 1.43)
D = 0.55
Z_HOVER = 0.045
Z_GRASP = 0.012
ARM_GAIN = 1.5
BASE_GAIN = 1.0 / 0.87
PICK_HEADING = -np.pi / 2
DROP_HEADING = -np.pi / 2
ISL_X = 1.0   # tip keep-out: island occupies x < ISL_X, y < ISL_Y
ISL_Y = 0.97
ISL_BX = 0.935   # island faces (for base footprint check)
ISL_BY = 0.92
BASE_HALF = 0.26
BASE_MARGIN = 0.04
PRE_GRIP = 0.6
PAIR_DIST = 0.04
LIFT_OK = 0.025
WIPER_FIRST = True
TR_ERR = 0.02
TR_DQ = 0.05
CARRY_ERR = 0.012
REL_STEPS = 2
CLOSE_STEPS = 3
W_CLOSE_STEPS = 5
W_HOVER = 0.45
W_GRASP = 0.30
W_DROP = 0.36
W_DROP_XY = (1.8, -0.35)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _slots():
    xs = np.arange(0.76, 0.14, -0.09)
    ys = [1.18, 1.30, 1.06]
    out = []
    for x in xs:
        for y in ys:
            out.append((x, y))
    # add a finer interleaved grid for many cubes
    for x in np.arange(0.715, 0.15, -0.09):
        for y in [1.24, 1.12, 1.36, 1.0]:
            out.append((x, y))
    return out


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        q_h, _, _ = kin.ik_multi((0, 0, 0), kin.Q_HOME, [D, 0, Z_HOVER], 'down',
                                 yaw=np.pi / 2, yaw_sym=np.pi / 2)
        q_g, _, _ = kin.ik((0, 0, 0), q_h, [D, 0, Z_GRASP], 'down',
                           yaw=np.pi / 2, yaw_sym=np.pi / 2)
        self.q_hover = kin.wrap_to(q_h, kin.Q_HOME)
        self.q_grasp = kin.wrap_to(q_g, self.q_hover)
        self.slots = _slots()
        self.q_wh = self._cfg(W_HOVER, self.q_hover)
        self.q_wg = self._cfg(W_GRASP, self.q_wh)
        self.q_wd = self._cfg(W_DROP, self.q_wh)

    def _cfg(self, z, q0):
        q, _, _ = kin.ik((0, 0, 0), q0, [D, 0, z], 'down', yaw=0.0, yaw_sym=np.pi / 2)
        return kin.wrap_to(q, q0)

    # ------------------------------------------------------------ helpers
    def _read(self, state):
        R = state.get_object_from_name('robot')
        self.base = np.array([state.get(R, 'pos_base_x'), state.get(R, 'pos_base_y'),
                              state.get(R, 'pos_base_rot')])
        self.q = np.array([state.get(R, j) for j in J])
        self.cubes = {}
        self.others = {}
        for o in state.get_objects(self._mov_type(state)):
            v = np.array([state.get(o, f) for f in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')])
            if o.name.startswith('cube'):
                self.cubes[o.name] = v
            else:
                self.others[o.name] = v

    def _mov_type(self, state):
        if getattr(self, '_mt', None) is None:
            for n in state.get_object_names():
                if n.startswith('cube'):
                    self._mt = state.get_object_from_name(n).type
                    break
        return self._mt

    @staticmethod
    def _in_box(p, m=0.04):
        return (BOX[0] + m < p[0] < BOX[1] - m and BOX[2] + m < p[1] < BOX[3] - m
                and p[2] < 0.08)

    def _arm_action(self, q_t):
        dq = kin.wrap_to(q_t, self.q) - self.q
        return np.clip(ARM_GAIN * dq, -0.1, 0.1), np.max(np.abs(dq))

    def _base_action(self, bt):
        d = np.array(bt, float) - self.base
        d[2] = _wrap(d[2])
        a = np.zeros(3)
        a[0:2] = np.clip(BASE_GAIN * d[0:2], -0.1, 0.1)
        a[2] = np.clip(d[2], -0.1, 0.1)
        return a, np.hypot(d[0], d[1]), abs(d[2])

    def _grasp_heading(self, name, gxy=None, pair=None):
        """Choose base heading th and wrist offset delta for grasping cube,
        minimising the estimated time (wrist / base rotation / travel)."""
        c = self.cubes[name].copy()
        yaw = 2 * np.arctan2(c[6], c[3])
        ks = (0, 1)
        if gxy is not None:
            c[:2] = gxy
            dv = self.cubes[pair][:2] - self.cubes[name][:2]
            yaw = np.arctan2(dv[1], dv[0])
            ks = (0,)
        d_cur = self.q[6] - self.q_hover[6]
        th_cur = self.base[2]
        tip = self._tip_xy()
        t_travel = np.hypot(*(c[:2] - tip)) / 0.087
        best = None
        for k in ks:
            cl = yaw + k * np.pi / 2           # world closing-axis direction
            ax = np.array([np.cos(cl), np.sin(cl)])
            pe = np.array([-ax[1], ax[0]])
            conflict = 0.0
            for n2, c2 in self.cubes.items():
                if n2 == name or n2 == pair:
                    continue
                dv = c2[:2] - c[:2]
                ua, up = abs(dv @ ax), abs(dv @ pe)
                if ua < 0.05 and up < 0.035:
                    conflict += (1.0 - ua / 0.05) * (1.0 - up / 0.035) * 4.0
            if self.last_k.get(name) == k:
                conflict += 1.0 * self.fails.get(name, 0)
            for dth in np.linspace(-1.2, 1.2, 25):
                th = _wrap(PICK_HEADING + dth)
                bp = c[:2] - D * np.array([np.cos(th), np.sin(th)])
                if not self._base_ok(bp, th):
                    continue
                delta = cl - th
                delta = d_cur + (delta - d_cur + np.pi / 2) % np.pi - np.pi / 2
                t = max(abs(delta - d_cur) / 0.025, abs(_wrap(th - th_cur)) / 0.1, t_travel)
                t += abs(dth) / 0.1        # turn back toward drop heading later
                cost = 40 * conflict + t
                if best is None or cost < best[0]:
                    best = (cost, th, delta, k)
        if best is None:
            return PICK_HEADING, d_cur
        self.last_k[name] = best[3]
        return best[1], best[2]

    @staticmethod
    def _base_ok(bp, th=None):
        r = BASE_HALF * (1.0 if th is None else abs(np.cos(th)) + abs(np.sin(th))) + BASE_MARGIN
        if not (bp[0] > ISL_BX + r or bp[1] > ISL_BY + r):
            return False
        return -2.3 < bp[0] < 1.95 - r and -2.5 < bp[1] < 2.15 - r

    def _pick_base(self, name, th):
        c = self.cubes[name]
        return np.array([c[0] - D * np.cos(th), c[1] - D * np.sin(th), th])

    def _next_slot(self):
        for s in self.slots:
            if s in self.used_slots:
                continue
            if any(np.hypot(c[0] - s[0], c[1] - s[1]) < 0.06 for c in self.cubes.values()):
                continue
            return s
        return (0.5, 1.18)

    # ------------------------------------------------------------ API
    def reset(self, state, info):
        self.phase = 'select'
        self._read(state)
        self.wiper = None
        for n, v in self.others.items():
            upright = abs(v[4]) < 0.1 and abs(v[5]) < 0.1
            near = any(np.hypot(*(c[:2] - v[:2])) < 1.0 for c in self.cubes.values())
            if upright and near and WIPER_FIRST:
                self.wiper = n
                self.phase = 'w_transit'
                break
        self.target = None
        self.th = None
        self.count = 0
        self.grip = 0.0
        self.used_slots = []
        self.fails = {}
        self.last_k = {}
        self.slot = None
        self.qh_t = self.q_hover.copy()
        self.delta = 0.0

    def _select(self):
        best = None
        tip = self.base[:2] + D * np.array([np.cos(self.base[2]), np.sin(self.base[2])])
        for n, c in self.cubes.items():
            if self._in_box(c):
                continue
            d = np.hypot(*(c[:2] - tip)) + 0.5 * self.fails.get(n, 0)
            if best is None or d < best[0]:
                best = (d, n)
        return None if best is None else best[1]

    def get_action(self, state):
        self._read(state)
        a = np.zeros(11, np.float32)
        for _ in range(4):
            r = self._step(a)
            if r is not None:
                return r
        a[10] = self.grip
        return a

    def _set(self, phase):
        self.phase = phase
        self.count = 0

    def _tip_xy(self):
        return self.base[:2] + D * np.array([np.cos(self.base[2]), np.sin(self.base[2])])

    @staticmethod
    def _seg_hits_island(p, q, n=12):
        for t in np.linspace(0, 1, n):
            x, y = p + t * (q - p)
            if x < ISL_X and y < ISL_Y and x > -0.5 and y > -1.0:
                return True
            if (x < ISL_X + 0.08 and y < ISL_Y + 0.08 and x > -0.5 and y > -1.0
                    and np.hypot(x - p[0], y - p[1]) > 0.12 and np.hypot(x - q[0], y - q[1]) > 0.12):
                return True
        return False

    def _route(self, goal):
        """Next tip waypoint toward goal avoiding the island (NE-corner detour)."""
        tip = self._tip_xy()
        goal = np.asarray(goal, float)
        if not self._seg_hits_island(tip, goal):
            return goal, True
        w = np.array([max(tip[0], goal[0], ISL_X + 0.1), max(tip[1], goal[1], ISL_Y + 0.1)])
        if tip[0] >= ISL_X or tip[1] >= ISL_Y:
            w = np.array([max(tip[0], goal[0]), max(tip[1], goal[1])])
            if w[0] < ISL_X + 0.05 and w[1] < ISL_Y + 0.05:
                w = np.array([max(w[0], ISL_X + 0.1), max(w[1], ISL_Y + 0.1)])
        if np.hypot(*(w - tip)) < 0.03:
            return goal, True
        return w, False

    def _move_tip(self, a, tip_goal, heading):
        """Base action placing the tip (reach D) at tip_goal with heading;
        rotation happens about the tip, limited to 0.1 rad/step."""
        dth = _wrap(heading - self.base[2])
        th_cmd = self.base[2] + np.clip(dth, -0.1, 0.1)
        bt = np.array([tip_goal[0] - D * np.cos(th_cmd), tip_goal[1] - D * np.sin(th_cmd), th_cmd])
        d = bt[:2] - self.base[:2]
        # base-level island avoidance: detour via NE corner of inflated island
        lim = BASE_HALF
        for t in np.linspace(0, 1, 8):
            p = self.base[:2] + t * d
            if p[0] < ISL_BX + lim and p[1] < ISL_BY + lim:
                w = np.array([max(self.base[0], bt[0], ISL_BX + lim + 0.02),
                              max(self.base[1], bt[1], ISL_BY + lim + 0.02)])
                if self.base[0] >= ISL_BX + lim or self.base[1] >= ISL_BY + lim:
                    w = np.array([max(self.base[0], bt[0]), max(self.base[1], bt[1])])
                if np.hypot(*(w - self.base[:2])) > 0.02:
                    d = w - self.base[:2]
                break
        a[0:2] = np.clip(BASE_GAIN * d, -0.1, 0.1)
        a[2] = np.clip(dth, -0.1, 0.1)
        return np.hypot(*(np.asarray(tip_goal) - self._tip_xy())), abs(dth)

    def _step(self, a):
        """Return an action, or None to re-dispatch after a phase change."""
        ph = self.phase
        self.count += 1
        if ph.startswith('w_'):
            return self._wiper_step(a, ph)
        if ph == 'select':
            n = self._select()
            if n is None:
                self.grip = 0.0
                arm, _ = self._arm_action(self.q_hover)
                a[3:10] = arm
                a[10] = 0.0
                return a
            self.target = n
            self.gxy, self.pair, self.pre = self.cubes[n][:2].copy(), None, PRE_GRIP
            best = None
            for n2, v in self.cubes.items():
                dd = np.hypot(*(v[:2] - self.gxy))
                if n2 != n and dd < PAIR_DIST and v[2] < 0.03 and not self._in_box(v) and (best is None or dd < best[0]):
                    best = (dd, n2)
            if best is not None and (self.fails.get(n, 0) % 2 == 1 or best[0] < 0.03):
                self.pair = best[1]
                self.gxy = 0.5 * (self.gxy + self.cubes[self.pair][:2])
                self.pre = 0.0
                self.th, self.delta = self._grasp_heading(n, self.gxy, self.pair)
            else:
                self.th, self.delta = self._grasp_heading(n)
            self.qh_t = self.q_hover.copy()
            self.qh_t[6] += self.delta
            self.qg_t = self.q_grasp.copy()
            self.qg_t[6] += self.delta
            self._set('transit')
            return None
        c = self.cubes.get(self.target)
        if ph == 'transit':
            self.grip = 0.0
            wp, final = self._route(self.gxy)
            hd = self.base[2] if not final else self.th
            if not final and abs(_wrap(self.base[2] - DROP_HEADING)) < 0.8:
                hd = DROP_HEADING
            err, dth = self._move_tip(a, wp, hd)
            arm, dq = self._arm_action(self.qh_t)
            a[3:10] = arm
            a[10] = 0.0
            if final and err < TR_ERR and dth < 0.02 and dq < TR_DQ:
                self._set('descend')
            elif self.count > 90:
                self.fails[self.target] = self.fails.get(self.target, 0) + 1
                self._set('select')
            return a
        if ph == 'descend':
            err, dth = self._move_tip(a, self.gxy, self.th)
            arm, dq = self._arm_action(self.qg_t)
            a[3:10] = arm
            a[10] = self.pre
            if (dq < 0.01 and err < 0.006) or self.count > 30:
                self._set('close')
            return a
        if ph == 'close':
            self.grip = 1.0
            arm, dq = self._arm_action(self.qg_t)
            a[3:10] = arm
            a[10] = 1.0
            if self.count >= CLOSE_STEPS:
                self._set('lift')
            return a
        if ph == 'lift':
            arm, dq = self._arm_action(self.qh_t)
            a[3:10] = arm
            a[10] = 1.0
            if c[2] > LIFT_OK or dq < 0.01 or self.count > 20:
                if c[2] > LIFT_OK:
                    self.slot = self._next_slot()
                    self.hold_xy = self._tip_xy()
                    self._set('turn')
                else:
                    self.fails[self.target] = self.fails.get(self.target, 0) + 1
                    self._set('reopen')
            return a
        if ph == 'reopen':
            self.grip = 0.0
            arm, dq = self._arm_action(self.qh_t)
            a[3:10] = arm
            a[10] = 0.0
            if self.count >= 4:
                self._set('select')
            return a
        if ph == 'turn':
            err, dth = self._move_tip(a, self.hold_xy, DROP_HEADING)
            arm, dq = self._arm_action(self.qh_t)
            a[3:10] = arm
            a[10] = 1.0
            if c[2] < 0.018 and self.count > 3:
                self.fails[self.target] = self.fails.get(self.target, 0) + 1
                self._set('reopen')
            elif dth < 0.05:
                self._set('carry')
            return a
        if ph == 'carry':
            wp, final = self._route(self.slot)
            err, dth = self._move_tip(a, wp, DROP_HEADING)
            arm, dq = self._arm_action(self.qh_t)
            a[3:10] = arm
            a[10] = 1.0
            if c[2] < 0.018 and not self._in_box(c, m=0.0):
                self.fails[self.target] = self.fails.get(self.target, 0) + 1
                self._set('reopen')
                return a
            if final and err < CARRY_ERR and dth < 0.05:
                self._set('release')
            elif self.count > 200:
                self._set('release')
            return a
        if ph == 'release':
            self.grip = 0.0
            arm, dq = self._arm_action(self.qh_t)
            a[3:10] = arm
            a[10] = 0.0
            if self.count >= REL_STEPS:
                self.used_slots.append(self.slot)
                self._set('select')
            return a
        self._set('select')
        return None

    def _wiper_step(self, a, ph):
        w = self.others[self.wiper]
        if ph == 'w_transit':
            wp, final = self._route(w[:2])
            err, dth = self._move_tip(a, wp, PICK_HEADING)
            arm, dq = self._arm_action(self.q_wh)
            a[3:10] = arm
            a[10] = 0.0
            if final and err < 0.008 and dth < 0.03 and dq < 0.02:
                self._set('w_descend')
            elif self.count > 200:
                self._set('select')
            return a
        if ph == 'w_descend':
            self._move_tip(a, self.w_xy if hasattr(self, 'w_xy') and False else w[:2], PICK_HEADING)
            arm, dq = self._arm_action(self.q_wg)
            a[3:10] = arm
            a[10] = 0.0
            if dq < 0.006 or self.count > 40:
                self._set('w_close')
            return a
        if ph == 'w_close':
            arm, dq = self._arm_action(self.q_wg)
            a[3:10] = arm
            a[10] = 1.0
            if self.count >= W_CLOSE_STEPS:
                self._set('w_lift')
            return a
        if ph == 'w_lift':
            arm, dq = self._arm_action(self.q_wd)
            a[3:10] = arm
            a[10] = 1.0
            if w[2] > 0.035 or dq < 0.02 or self.count > 40:
                if w[2] > 0.025:
                    self._set('w_carry')
                else:
                    self._set('w_fail')
            return a
        if ph == 'w_turn':
            err, dth = self._move_tip(a, self.hold_xy, 0.0)
            arm, dq = self._arm_action(self.q_wh)
            a[3:10] = arm
            a[10] = 1.0
            if dth < 0.05:
                self._set('w_carry')
            return a
        if ph == 'w_carry':
            err, dth = self._move_tip(a, W_DROP_XY, PICK_HEADING)
            arm, dq = self._arm_action(self.q_wd)
            a[3:10] = arm
            a[10] = 1.0
            if (err < 0.03 and dth < 0.05) or self.count > 150:
                self._set('w_fail')
            return a
        if ph == 'w_lower':
            self._move_tip(a, W_DROP_XY, PICK_HEADING)
            arm, dq = self._arm_action(self.q_wd)
            a[3:10] = arm
            a[10] = 1.0
            if dq < 0.01 or self.count > 30:
                self._set('w_fail')
            return a
        # w_fail / release: open and continue with cubes
        arm, dq = self._arm_action(self.q_wd)
        a[3:10] = arm
        a[10] = 0.0
        self.grip = 0.0
        if self.count >= 5:
            self._set('select')
        return a
