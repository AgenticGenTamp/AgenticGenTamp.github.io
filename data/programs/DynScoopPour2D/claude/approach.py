import numpy as np

LIM = np.array([0.03, 0.03, 0.098174, 0.08, 0.015])
PI = np.pi


def _wrap(a):
    return (a + PI) % (2 * PI) - PI


class GeneratedApproach:
    """Grasp the L-hook, use it as a dustpan: sweep small objects against the
    left wall, scoop them, carry over the middle wall, tip them out on the right."""

    # geometry constants (world 3.5 x 3.0, middle wall x in [1.70,1.80], top y=1.50)
    WALL_L = 1.70
    WALL_R = 1.80
    WALL_TOP = 1.50
    CROSS_X = 1.66      # x at which we must already be high (left of wall)
    CARRY_Y = 2.30      # base y when carrying the loaded pan across (recomputed)
    DUMP_X = 2.08
    SWEEP_X = 0.70      # left end of sweep (pan lip hits left wall)

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ---------- reading ----------
    def _find(self, state, key, fallback):
        for n in sorted(state.get_object_names()):
            o = state.get_object_from_name(n)
            try:
                tn = o.type.name
            except Exception:
                tn = ''
            if key in tn:
                return o
        return state.get_object_from_name(fallback)

    def _read(self, state):
        R = getattr(self, 'R', None)
        if R is None:
            R = self._find(state, 'robot', 'robot')
        self.R = R
        self.p = np.array([state.get(R, 'x'), state.get(R, 'y'), state.get(R, 'theta'),
                           state.get(R, 'arm_joint'), state.get(R, 'finger_gap')], dtype=float)
        H = getattr(self, 'H', None)
        if H is None:
            H = self._find(state, 'hook', 'hook')
        self.H = H
        self.hp = np.array([state.get(H, 'x'), state.get(H, 'y'), state.get(H, 'theta')], dtype=float)
        self.held = float(state.get(H, 'held')) > 0.5

    def _smalls(self, state):
        out = []
        for n in state.get_object_names():
            o = state.get_object_from_name(n)
            try:
                tn = o.type.name
            except Exception:
                tn = ''
            if 'small' in tn or n.startswith('small'):
                out.append(o)
        return out

    def _rel(self):
        th = self.p[2]
        d = self.hp[:2] - self.p[:2]
        c, s = np.cos(th), np.sin(th)
        return c * d[0] + s * d[1], -s * d[0] + c * d[1], _wrap(self.hp[2] - th)

    # ---------- motion primitive ----------
    def _act(self, x=None, y=None, th=None, arm=None, gap=None, rate=True):
        p = self.p
        tx = x if x is not None else p[0]
        ty = y if y is not None else p[1]
        dx, dy = tx - p[0], ty - p[1]
        if rate:
            n = max(abs(dx) / LIM[0], abs(dy) / LIM[1], 1e-9)
            if n > 1.0:
                dx, dy = dx / n, dy / n
        dth = _wrap((th if th is not None else p[2]) - p[2])
        da = (arm if arm is not None else p[3]) - p[3]
        dg = (gap if gap is not None else p[4]) - p[4]
        self._err = np.array([tx - p[0], ty - p[1], dth, da, dg])
        return np.clip(np.array([dx, dy, dth, da, dg]), -LIM, LIM).astype(np.float32)

    def _at(self, tol=0.012, tolth=0.02):
        e = self._err
        return abs(e[0]) < tol and abs(e[1]) < tol and abs(e[2]) < tolth

    # ---------- lifecycle ----------
    def reset(self, state, info):
        self.R = None
        self.H = None
        self._read(state)
        self.far = float(self.hp[0]) > 3.30
        self.phase = self._entry()
        self.tphase = 0
        self.t = 0
        self.rel_f = 0.72
        self.rel_t = 0.0
        self.hook_x0 = float(self.hp[0])
        self.post_top = float(self.hp[1]) + float(state.get(self.H, 'length_side1'))
        self.retry = 0.0
        self.trip = 0
        self.sx = 1.45
        self.n_small = len(self._smalls(state))

    def _entry(self):
        far = float(self.hp[0]) > 3.30
        if not far:
            self.hook_x0 = float(self.hp[0])
        return 'fapproach' if far else 'approach'

    def _go(self, ph):
        self.phase = ph
        self.tphase = 0

    def get_action(self, state):
        self._read(state)
        self.t += 1
        self.tphase += 1
        if self.held:
            f, l, rt = self._rel()
            self.rel_f, self.rel_t = f, rt
        return getattr(self, '_ph_' + self.phase)(state)

    def _pan_th(self):
        return -PI / 2 - self.rel_t

    def _pan_y(self, d=0.0):
        return self.rel_f * np.cos(d) + 0.5 * np.sin(d) + 0.04

    TILT = 0.55
    RAMP = 0.25

    def _carry_y(self):
        return float(min(2.72, self.rel_f * np.cos(self.TILT) + 1.60))

    def _sweep_start(self, state):
        xs = [float(state.get(o, 'x')) for o in self._smalls(state)]
        left = [x for x in xs if x < self.WALL_L]
        if not left:
            return 1.45
        left.sort()
        mx = left[-1]
        return float(min(1.45, max(0.70, mx + 0.10)))

    # ---------- phases: get the hook ----------
    # ---- far hook (spawns too close to the right wall for a top grasp) ----
    def _ph_fapproach(self, state):
        a = self._act(x=3.29, y=2.30, th=-PI / 2, arm=0.4, gap=0.25)
        e = self._err
        if (self.p[0] > 3.283 and abs(e[1]) < 0.03 and abs(e[2]) < 0.02
                and self.p[3] > 0.39 and self.p[4] > 0.24) or self.tphase > 260:
            self._go('ftilt')
        return a

    def _ph_ftilt(self, state):
        if self.p[1] > 1.23:
            return self._act(x=3.29, y=1.20, arm=0.4, gap=0.25)
        a = self._act(x=3.29, y=1.20, th=-PI / 2 + 0.09, arm=0.4, gap=0.25)
        if abs(_wrap(self.p[2] - (-PI / 2 + 0.09))) < 0.03 or self.tphase > 60:
            self._go('fdescend')
        return a

    def _ph_fdescend(self, state):
        ty = float(self.hp[1]) + 0.46 + 0.46 * np.cos(0.098)
        a = self._act(x=3.29, y=ty, th=self.p[2], arm=0.4, gap=0.25)
        if abs(self.p[1] - ty) < 0.015 or self.tphase > 60:
            self._go('close')
        return a

    def _ph_approach(self, state):
        tx = min(self.hook_x0 + self.retry, 3.28)
        ty = 1.30 if (abs(self.p[0] - tx) < 0.05 and abs(_wrap(self.p[2] + PI / 2)) < 0.05) else 2.0
        a = self._act(x=tx, y=ty, th=-PI / 2, arm=0.2, gap=0.25)
        if (self._at(0.02) and self.p[4] > 0.24) or self.tphase > 250:
            self._go('descend')
        return a

    def _ph_descend(self, state):
        ty = self.post_top + self.p[3] + 0.02
        a = self._act(x=min(self.hook_x0 + self.retry, 3.28), y=ty, th=-PI / 2, gap=0.25)
        if abs(self.p[1] - ty) < 0.01 or self.tphase > 120:
            self._go('close')
        return a

    def _ph_close(self, state):
        if self.held:
            self._go('lift')
            return self._act(gap=0.08)
        if self.p[4] <= 0.0815 or self.tphase > 40:
            self.retry = -0.06 if self.retry >= 0 else 0.04
            self.hook_x0 = float(self.hp[0])
            self.post_top = float(self.hp[1]) + float(state.get(self.H, 'length_side1'))
            self._go(self._entry())
            return self._act(gap=0.25)
        return self._act(gap=0.08)

    def _ph_lift(self, state):
        a = self._act(y=self._carry_y(), th=self._pan_th(), arm=0.2, gap=0.08)
        if not self.held:
            self._go(self._entry())
        elif abs(self.p[1] - self._carry_y()) < 0.02 or self.tphase > 120:
            self._go('cross_left')
        return a

    # ---------- phases: the scoop cycle ----------
    def _ph_cross_left(self, state):
        # get to the descent column, left of the wall, still high
        if self.tphase == 1:
            self.sx = self._sweep_start(state)
        a = self._act(x=self.sx, y=max(self.p[1], self._carry_y()), th=self._pan_th(), gap=0.08)
        if not self.held:
            self._go(self._entry())
        elif self._at(0.02) or self.tphase > 200:
            self._go('lower')
        return a

    def _ph_lower(self, state):
        # press a bit below floor contact: reseats the hook in the gripper
        ty = self._pan_y(self.RAMP) - 0.14
        a = self._act(x=self.sx, y=ty, th=self._pan_th() + self.RAMP, gap=0.08)
        if self.tphase == 1:
            self.stall = 0
            self.ylast = self.p[1]
        if abs(self.p[1] - self.ylast) < 0.004:
            self.stall += 1
        else:
            self.stall = 0
        self.ylast = self.p[1]
        if not self.held:
            self._go(self._entry())
        elif (abs(self.p[1] - ty) < 0.02 and abs(self.p[0] - self.sx) < 0.02) \
                or (self.stall > 3 and self.tphase > 25 and abs(self.p[0] - self.sx) < 0.05) \
                or self.tphase > 90:
            self._go('sweep')
        return a

    def _ph_sweep(self, state):
        if self.tphase == 1:
            nleft = sum(1 for o in self._smalls(state)
                        if float(state.get(o, 'x')) < self.WALL_L)
            self.swx = self.SWEEP_X if nleft >= 14 else 0.55
            self.dopress = nleft < 14
        ty = self._pan_y(self.RAMP)
        a = self._act(x=self.swx, y=ty, th=self._pan_th() + self.RAMP, gap=0.08)
        if not self.held:
            self._go(self._entry())
        elif self.p[0] < self.swx + 0.02 or self.tphase > 90:
            self._go('press' if self.dopress else 'level')
        return a

    def _ph_press(self, state):
        ty = self._pan_y(self.RAMP)
        a = self._act(x=0.40, y=ty, th=self._pan_th() + self.RAMP, gap=0.08, rate=False)
        if self.tphase > 10:
            self._go('level')
        return a

    def _ph_level(self, state):
        a = self._act(y=self._pan_y(), th=self._pan_th(), gap=0.08)
        if self.tphase > 5:
            self._go('tilt')
        return a

    def _ph_tilt(self, state):
        # tip the lip up so the load is retained
        a = self._act(th=self._pan_th() - self.TILT, gap=0.08)
        if not self.held:
            self._go(self._entry())
        elif abs(_wrap(self.p[2] - (self._pan_th() - self.TILT))) < 0.03 or self.tphase > 20:
            self._go('rise')
        return a

    SWING_X = 1.60

    def _swing_y(self):
        r = float(np.sqrt(self.rel_f ** 2 + 0.26))
        return float(min(2.72, max(2.40, 1.5 + r + 0.07)))

    def _ph_rise(self, state):
        a = self._act(x=self.SWING_X, y=self._swing_y(), th=self._pan_th() - self.TILT, gap=0.08)
        if not self.held:
            self._go(self._entry())
        elif self._at(0.02) or self.tphase > 150:
            self.swing_acc = 0.0
            self.th_prev = self.p[2]
            self._go('swing')
        return a

    def _ph_swing(self, state):
        # whip the pan over the top clockwise: the load is released on the right
        self.swing_acc += abs(_wrap(self.p[2] - self.th_prev))
        self.th_prev = self.p[2]
        if not self.held:
            self._go(self._entry())
            return self._act()
        if self.swing_acc > 2 * PI - 0.35 or self.tphase > 90:
            self.trip += 1
            self._go('cross_left')
            return self._act(th=self._pan_th(), gap=0.08)
        return np.array([0.0, 0.0, -LIM[2], 0.0, 0.0], dtype=np.float32)

