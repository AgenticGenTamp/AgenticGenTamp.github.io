import math
import numpy as np

WALL_L, WALL_R, WALL_TOP = 1.70, 1.80, 1.50
MAXS = np.array([0.03, 0.03, 0.098, 0.08, 0.015])
TOL = np.array([2e-3, 2e-3, 5e-3, 2e-3, 2e-3])
TH0 = -math.pi / 2


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def robot_pts(x, y, th, aj, gap):
    d = np.array([math.cos(th), math.sin(th)])
    n = np.array([-d[1], d[0]])
    c = np.array([x, y])
    pts = []
    for al in (aj, aj + 0.04):
        for ac in (-0.125, 0.125):
            pts.append(c + al * d + ac * n)
    for sg in (-1, 1):
        for al in (aj, aj + 0.12):
            for ac in (gap / 2 - 0.02, gap / 2 + 0.02):
                pts.append(c + al * d + sg * ac * n)
    # sample along base edge
    for t in np.linspace(-0.125, 0.125, 5):
        pts.append(c + (aj + 0.02) * d + t * n)
    return np.array(pts)


def collides(x, y, th, aj, gap, m=0.008):
    # base circle
    if x < 0.2 + m or x > 3.3 - m or y < 0.2 + m or y > 2.8 - m:
        return True
    # base vs middle wall
    cx = min(max(x, WALL_L), WALL_R)
    cy = min(y, WALL_TOP)
    if (x - cx) ** 2 + (y - cy) ** 2 < (0.2 + m) ** 2:
        return True
    P = robot_pts(x, y, th, aj, gap)
    if np.any(P[:, 0] < m) or np.any(P[:, 0] > 3.5 - m) or np.any(P[:, 1] < m) or np.any(P[:, 1] > 3.0 - m):
        return True
    inw = (P[:, 0] > WALL_L - m) & (P[:, 0] < WALL_R + m) & (P[:, 1] < WALL_TOP + m)
    if np.any(inw):
        return True
    return False


class GeneratedApproach:
    PHI_A = 0.5        # right-side scoop tilt
    LOW_A = 0.42
    SCOOP_X = 2.06
    VERT_MAX_X = 3.30
    HOOK_A_X = 1.73
    PHI_B = 0.25       # left-side scoop tilt
    LOW_B = 0.72
    ARM_B = 0.4
    START_BX = 1.14
    CARRY_Y = 2.55
    DUMP_X = 2.9
    DUMP_TH = -1.3
    PHI_CARRY = 0.9
    DWELL = 4
    FIRST = 'B'
    B_CAP = 10
    WIN_SHIFT = 0.0
    XO0 = 3.499
    LIFT_DY = 0.5
    CARRY_ARM = None

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    def reset(self, state, info):
        self.phase = 'up'
        self.ps = 0
        self.dump_th = None
        self.bx = None
        self.n_scoops = 0
        self.dth = 0.0
        self.last_pose = None
        self.last_a = np.zeros(5)
        self.stuck = 0
        self.grasp_try = 0
        self.gpose = None

    def _robot(self, s):
        r = s.get_object_from_name('robot')
        return dict(x=s.get(r, 'x'), y=s.get(r, 'y'), th=s.get(r, 'theta'),
                    arm=s.get(r, 'arm_joint'), gap=s.get(r, 'finger_gap'))

    def _hook(self, s):
        h = s.get_object_from_name('hook')
        return dict(x=s.get(h, 'x'), y=s.get(h, 'y'), th=s.get(h, 'theta'),
                    held=s.get(h, 'held'))

    def _smalls(self, s):
        out = []
        for n in s.get_object_names():
            if n.startswith('small'):
                o = s.get_object_from_name(n)
                out.append((s.get(o, 'x'), s.get(o, 'y')))
        return out

    def _toward(self, r, x=None, y=None, th=None, arm=None, gap=None):
        a = np.zeros(5)
        if x is not None:
            a[0] = x - r['x']
        if y is not None:
            a[1] = y - r['y']
        if th is not None:
            a[2] = wrap(th - r['th'])
        if arm is not None:
            a[3] = arm - r['arm']
        if gap is not None:
            a[4] = gap - r['gap']
        done = bool(np.all(np.abs(a) < TOL))
        return np.clip(a, -MAXS, MAXS), done

    def _grasp_pose(self, h):
        if h['x'] <= self.VERT_MAX_X:
            return h['x'] - 0.026, h['y'] + 0.74
        xo = min(self.XO0 if self.grasp_try == 0 else 3.499, h['x'] + 0.015)
        return min(3.3, xo - 0.20568), h['y'] + 0.71376

    def _gth(self, h):
        return TH0 if h['x'] <= self.VERT_MAX_X else TH0 + 0.2

    def R(self, hook_th):
        return hook_th - self.dth

    def _go(self, name):
        self.phase = name
        self.ps = 0

    def _choose_bx(self, state):
        pts = [p for p in self._smalls(state) if p[0] < WALL_L]
        r, h = self._robot(state), self._hook(state)
        off = h['x'] - r['x']
        best, bx = -1, self.START_BX
        for rx in np.arange(0.45, self.START_BX + 1e-6, 0.05):
            hx = rx + off
            lo, hi = hx - 0.53 + self.WIN_SHIFT, hx - 0.01 + self.WIN_SHIFT
            c = sum(1 for (x, y) in pts if lo <= x <= hi)
            if c > best + 0.5:
                best, bx = c, rx
        return float(bx)

    def get_action(self, state):
        r = self._robot(state)
        pose = np.array([r['x'], r['y'], r['th'], r['arm'], r['gap']])
        if self.last_pose is not None and np.max(np.abs(pose - self.last_pose)) < 1e-7 \
                and np.max(np.abs(self.last_a)) > 1e-6:
            self.stuck += 1
        else:
            self.stuck = 0
        a = self._get_action(state)
        if self.phase in ('up', 'above_hook', 'descend_hook', 'close', 'regrasp'):
            out = np.clip(a, -MAXS, MAXS)
        else:
            out = self._safe(state, a, self.stuck)
        self.last_pose = pose
        self.last_a = np.array(out, float)
        return out

    def _safe(self, state, a, stuck=0):
        r = self._robot(state)
        a = np.clip(np.asarray(a, float), -MAXS, MAXS)
        def ok(b):
            return not collides(r['x'] + b[0], r['y'] + b[1], r['th'] + b[2], np.clip(r['arm'] + b[3], 0.2, 0.4), np.clip(r['gap'] + b[4], 0.08, 0.25))
        if collides(r['x'], r['y'], r['th'], r['arm'], r['gap'], m=0.0):
            return a
        cands = [a]
        b = a.copy(); b[2] = 0; b[3] = 0; cands.append(b)
        b = a.copy(); b[0] = 0; cands.append(b)
        b = a.copy(); b[1] = 0; cands.append(b)
        b = a.copy(); b[0] = 0; b[2] = 0; b[3] = 0; cands.append(b)
        b = a.copy(); b[1] = 0; b[2] = 0; b[3] = 0; cands.append(b)
        good = [b for b in cands if ok(b)] or [a]
        if stuck > 0:
            good = [b for b in cands if np.max(np.abs(b)) > 1e-6]
            return good[stuck % len(good)]
        return good[0]

    def _get_action(self, state):
        r = self._robot(state)
        h = self._hook(state)
        self.ps += 1
        if self.phase not in ('up', 'above_hook', 'descend_hook', 'close', 'regrasp') and h['held'] < 0.5:
            self._go('regrasp')
        if self.phase == 'close' and self.ps > 30:
            self.grasp_try += 1
            self._go('regrasp')
        for _ in range(20):
            p = self.phase
            if p != 'descend_hook' and p != 'close':
                self.gpose = None
            if p == 'regrasp':  # open and back off upward
                a, d = self._toward(r, y=min(2.2, r['y'] + 0.4), gap=0.25)
                if self.ps > 9:
                    self._go('up'); continue
                return a
            if p == 'up':
                if r['y'] >= 1.95:
                    self._go('above_hook'); continue
                a, d = self._toward(r, y=2.2)
                return a
            if p == 'above_hook':
                gx, gy = self._grasp_pose(h)
                ty = 1.95 if r['x'] < 2.15 else gy + 0.25
                a, d = self._toward(r, x=gx, y=ty, th=self._gth(h), arm=0.2, gap=0.25)
                if r['x'] < 2.15 and r['y'] < 1.9:
                    a[0] = 0.0
                if d:
                    self._go('descend_hook'); continue
                return a
            if p == 'descend_hook':
                if self.gpose is None:
                    self.gpose = self._grasp_pose(h) + (self._gth(h),)
                gx, gy, gth = self.gpose
                a, d = self._toward(r, x=gx, y=gy, th=gth, arm=0.2, gap=0.25)
                if gth != TH0 and r['y'] < gy + 0.12:
                    a[1] = max(a[1], -0.005)
                if d:
                    self._go('close'); continue
                return a
            if p == 'close':
                if h['held'] > 0.5:
                    self.dth = wrap(h['th'] - r['th'])
                    first = self.FIRST
                    if first == 'auto':
                        n = len(self._smalls(state))
                        first = 'B' if (n + 1) // 2 <= self.B_CAP else 'A'
                    self._go('A_low' if first == 'A' else 'B_prep'); continue
                a = np.zeros(5); a[4] = -0.015
                return a
            # ---- right-side scoop A
            if p == 'A_low':
                a, d = self._toward(r, y=self.LOW_A, th=self.R(TH0 - self.PHI_A), arm=0.2)
                if d or self.ps > 60:
                    self._go('A_left'); continue
                return a
            if p == 'A_left':
                tx = max(self.SCOOP_X, r['x'] + self.HOOK_A_X - h['x'])
                a, d = self._toward(r, x=tx, y=self.LOW_A, th=self.R(TH0 - self.PHI_A))
                if abs(a[0]) < 1e-9 and self.ps > 3:
                    d = True
                if d or self.ps > 80:
                    self._go('A_lift'); continue
                return a
            if p == 'A_lift':
                self.lift_y0 = r['y']
                self._go('carry'); continue
            # ---- carry & dump
            if p == 'carry':
                ty = min(2.75, max(self.CARRY_Y, r['y'] + 1.62 - h['y']))
                a, d = self._toward(r, x=self.DUMP_X, y=ty)
                if h['y'] > 0.12:
                    a[2] = np.clip(wrap(self.R(TH0 - self.PHI_CARRY) - r['th']), -MAXS[2], MAXS[2])
                ok_x = (h['y'] >= 1.6) or (h['x'] + 0.03 < 1.62)
                if r['y'] < self.lift_y0 + self.LIFT_DY:
                    ok_x = False
                if self.CARRY_ARM is not None and h['y'] > 0.3:
                    a[3] = np.clip(self.CARRY_ARM - r['arm'], -MAXS[3], MAXS[3])
                if r['y'] < 1.75 and r['x'] + 0.03 > 1.48 and r['x'] < 1.9:
                    ok_x = False
                if not ok_x:
                    a[0] = 0.0
                if r['x'] >= self.DUMP_X - 0.002 and h['y'] >= 1.55:
                    self.dump_th = self.R(self.DUMP_TH)
                    self._go('dump'); continue
                if self.ps > 150:
                    self.dump_th = self.R(self.DUMP_TH)
                    self._go('dump'); continue
                return a
            if p == 'dump':
                a, d = self._toward(r, th=self.dump_th)
                if d or self.ps > 30:
                    self._go('dwell'); continue
                return a
            if p == 'dwell':
                if self.ps > self.DWELL:
                    self.n_scoops += 1
                    self._go('B_prep'); continue
                return np.zeros(5)
            # ---- left-side scoop B
            if p == 'B_prep':  # rotate so hook points right, move over wall
                a, d = self._toward(r, x=self.START_BX, y=1.78, th=self.R(0.0), arm=self.ARM_B)
                if r['y'] < 1.8 and r['x'] > self.START_BX + 0.01:
                    a[1] = max(a[1], 0.0)
                if d or self.ps > 120:
                    self._go('B_down'); continue
                return a
            if p == 'B_down':
                a, d = self._toward(r, x=self.START_BX, y=self.LOW_B, th=self.R(0.0), arm=self.ARM_B)
                if d or self.ps > 60:
                    self._go('B_swing'); continue
                return a
            if p == 'B_swing':
                a, d = self._toward(r, th=self.R(TH0 - self.PHI_B), arm=self.ARM_B)
                if wrap(r['th'] + self.dth) > -1.2:
                    a[3] = 0.0
                if d or self.ps > 40:
                    self.bx = self._choose_bx(state)
                    self._go('B_left'); continue
                return a
            if p == 'B_left':
                a, d = self._toward(r, x=self.bx, y=self.LOW_B, th=self.R(TH0 - self.PHI_B))
                if d or self.ps > 60:
                    self._go('B_lift'); continue
                return a
            if p == 'B_lift':
                self.lift_y0 = r['y']
                self._go('carry'); continue
            return np.zeros(5)
        return np.zeros(5)
