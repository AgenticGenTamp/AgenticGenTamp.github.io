"""Pick-and-place approach for Rearrange3D: put boxed drink and can next to bowl.

Action semantics (found empirically):
  a[0:3]  base delta (world frame), applied gain ~0.87
  a[3:10] arm joint deltas; internal joint target += 0.25*a (integrating)
  a[10]   gripper absolute (0 open, 1 closed); acts within ~2 steps
Goal (found empirically): object centre within ~0.17m (xy) of bowl centre and
resting on the counter.  Episode terminates when both objects satisfy it.
"""
import numpy as np
from kin import fk_world, ik_arm, rotz, MOUNT, READY

BOWL, DRINK, CAN = 0, 16, 32
OBJS = (DRINK, CAN)
ROB = 93
Q = slice(96, 103)
DEBUG = False
FORCE = {}

PLACE_DIST = 0.115      # centre-centre distance from bowl for placement
OK_DIST = 0.128         # considered satisfied below this
GRASP_H = 0.055          # closed fingertip height above object bottom
VMAX = 0.024            # joint speed (rad/step) for interpolation (limit .025)
PRE_TOL = 0.01
DESC_V = 0.02
PRE_CLR = 0.02
ORBIT_GAP = 0.10
PUSH_STROKE = 0.2
BASE_X = -0.5          # base x to move to (table edge ~ -0.066 limit)


def rdown(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])


def yaw_of(qw, qx, qy, qz):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space=None, observation_space=None, primitives=None):
        self.action_space = action_space

    # ------------------------------------------------------------ kinematics
    def _to_arm(self, pw):
        b = self.base_t
        return rotz(-b[2]) @ (np.asarray(pw, float) - np.array([b[0], b[1], 0.0])) - MOUNT

    def _ik(self, pw, psi, q0):
        b = self.base_t
        q, err = ik_arm(self._to_arm(pw), rotz(-b[2]) @ rdown(psi), q0, iters=150, rest=READY)
        return q

    def _fk(self, q):
        return fk_world(self.base_t, q)[0]

    def _tip_yaw(self, q):
        R = fk_world(self.base_t, q)[1]
        return np.arctan2(R[1, 0], R[0, 0])

    # ------------------------------------------------------------ episode
    def reset(self, state, info=None):
        s = np.asarray(state, float)
        self.QT = s[Q].copy()
        self.base_t = s[ROB:ROB + 3].copy()
        self.base_t[0] = max(self.base_t[0], BASE_X)
        self.table_z = float(np.median([s[o + 2] - s[o + 15] / 2 for o in (BOWL, DRINK, CAN)]))
        self.bias = np.zeros(7)
        self.t = 0
        self.grip = 0.0
        self.segs = []
        self.seg = None
        self.plan_q = s[Q].copy()
        self.plan_p = self._fk(self.plan_q)
        self.plan_psi = self._tip_yaw(self.plan_q)
        self.cur_obj = DRINK
        self.done = False
        self.placed = set()
        self.push_n = 0
        self.hist = {o: [] for o in OBJS}
        self.cls = {}
        self.away = None
        self.last_bowl = None

    # ------------------------------------------------------------ goal logic
    def _satisfied(self, s, o):
        if o in FORCE:
            return o in self.placed
        d = np.linalg.norm(s[o:o + 2] - s[BOWL:BOWL + 2])
        on = s[o + 2] - s[o + 15] / 2 < self.table_z + 0.02
        return d < OK_DIST and on

    def _embedded(self, s, o):
        return s[o + 2] - s[o + 15] / 2 < self.table_z - 0.015

    def _fit_orbit(self, o):
        h = np.array(self.hist[o][-60:])
        x, y = h[:, 0], h[:, 1]
        A = np.c_[x, y, np.ones(len(x))]
        sol = np.linalg.lstsq(A, -(x * x + y * y), rcond=None)[0]
        c = -sol[:2] / 2
        r = np.sqrt(max(c @ c - sol[2], 0.0))
        return c, r

    def _classify(self, s, o):
        """For embedded objects: 'wait' (unknown yet), 'stuck', or ('orbit', c, r)."""
        if o in self.cls:
            return self.cls[o]
        if len(self.hist[o]) < 55:
            return 'wait'
        h = np.array(self.hist[o])
        mv = np.linalg.norm(h[-1] - h[-20])
        c, r = self._fit_orbit(o)
        if mv > 0.03 and 0.03 < r < 0.25:
            self.cls[o] = ('orbit', c, r)
        elif mv < 0.01:
            self.cls[o] = 'stuck'
        else:
            return 'wait'
        return self.cls[o]

    def _plan_push(self, s, T):
        """Push the bowl towards xy target T."""
        self.base_t = s[ROB:ROB + 3].copy()
        self.plan_q = self.QT - self.bias
        self.plan_p = self._fk(self.plan_q)
        self.plan_psi = self._tip_yaw(self.plan_q)
        B = s[BOWL:BOWL + 2]
        if self.last_bowl is not None and np.linalg.norm(B - self.last_bowl) < 0.004:
            self.push_n = 99  # bowl blocked: give up pushing
        self.last_bowl = B.copy()
        self.push_n += 1
        tz = self.table_z
        v = T - B
        L = np.linalg.norm(v)
        u = v / (L + 1e-9)
        start = B - u * 0.09
        end = B + u * min(L, PUSH_STROKE) - u * 0.07
        psi = np.arctan2(u[1], u[0]) + np.pi / 2
        psi = min([psi + k * np.pi for k in (-2, -1, 0, 1, 2)], key=lambda c: abs(c - self.plan_psi))
        hov = tz + 0.07
        self.grip = 1.0
        if self.plan_p[2] < hov - 0.01:
            up = self.plan_p.copy()
            up[2] = hov
            self._add_lin(up, self.plan_psi, 1.0, tol=0.05, step=0.05)
        self._add_lin([start[0], start[1], hov], psi, 1.0, tol=0.02, step=0.06)
        self._add_lin([start[0], start[1], tz + 0.02], psi, 1.0, tol=0.01, step=0.03)
        self._add_lin([end[0], end[1], tz + 0.02], psi, 1.0, tol=0.03, step=0.02, vmax=0.015)
        self._add_lin([end[0], end[1], hov], psi, 1.0, tol=0.05, step=0.05)
        self.segs.append(dict(kind='check'))
        return True

    def _choose_target(self, s, o, other_target):
        bowl = s[BOWL:BOWL + 2]
        p = s[o:o + 2]
        d = p - bowl
        ang0 = np.arctan2(d[1], d[0]) if self.away is None else self.away
        others = [s[x:x + 2] for x in OBJS if x != o]
        for da in np.linspace(0, np.pi, 19):
            for sgn in (1, -1):
                a = ang0 + sgn * da
                tgt = bowl + PLACE_DIST * np.array([np.cos(a), np.sin(a)])
                if not (0.1 <= tgt[0] <= 0.62 and -0.45 <= tgt[1] <= 0.45):
                    continue
                if other_target is not None and np.linalg.norm(tgt - other_target) < 0.10:
                    continue
                if any(np.linalg.norm(tgt - u) < 0.10 for u in others):
                    continue
                return tgt
        return bowl + PLACE_DIST * d / (np.linalg.norm(d) + 1e-9)

    def _grasp_psi(self, s, o):
        cur = self.plan_psi
        if o == CAN:
            return cur
        yaw = yaw_of(*s[o + 3:o + 7])
        cands = [yaw + k * np.pi / 2 for k in range(4)]
        return min(cands, key=lambda c: abs(wrap(c - cur)))

    # ------------------------------------------------------------ planning
    def _add_lin(self, p1, psi, grip, tol=None, step=0.03, hold=0, vmax=VMAX):
        """Cartesian straight line from plan_p to p1 (with psi) as joint waypoints."""
        p0 = self.plan_p
        p1 = np.asarray(p1, float)
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
        q = self.plan_q
        qs = []
        for k in range(1, n + 1):
            p = p0 + (p1 - p0) * k / n
            q = self._ik(p, psi, q)
            qs.append(q)
        self.segs.append(dict(kind='path', qs=qs, grip=grip, tol=tol, hold=hold, vmax=vmax))
        self.plan_q = q
        self.plan_p = p1
        self.plan_psi = psi

    def _add_grip(self, grip, n):
        self.segs.append(dict(kind='grip', grip=grip, n=n))

    def _plan(self, s):
        """Plan next pick-and-place. Returns False if nothing to do."""
        todo = [o for o in OBJS if not self._satisfied(s, o)]
        if not todo:
            return False
        self.away = None
        for o in list(todo):
            if not self._embedded(s, o):
                continue
            cl = self._classify(s, o)
            if cl == 'wait':
                self.segs.append(dict(kind='wait', grip=self.grip, n=4))
                return True
            todo.remove(o)
            if cl == 'stuck':
                continue
            c, r = cl[1], cl[2]
            B = s[BOWL:BOWL + 2]
            u = (B - c) / (np.linalg.norm(B - c) + 1e-9)
            self.away = np.arctan2(u[1], u[0])
            if np.linalg.norm(B - c) > r + ORBIT_GAP + 0.015 and self.push_n < 12:
                return self._plan_push(s, c + u * (r + ORBIT_GAP))
        if not todo:
            if self.plan_p is not None and self.segs == []:
                self.segs.append(dict(kind='wait', grip=0.0, n=10))
                return True
            return False
        tip = self._fk(s[Q])
        if len(todo) == 2:
            todo.sort(key=lambda o: np.linalg.norm(s[o:o + 2] - tip[:2]))
        o = todo[0]
        last = len(todo) == 1
        other = [x for x in OBJS if x != o][0]
        other_t = s[other:other + 2] if self._satisfied(s, other) else None
        tgt = self._choose_target(s, o, other_t)
        if o in FORCE:
            tgt = s[BOWL:BOWL + 2] + np.asarray(FORCE[o])
        self.cur_obj, self.cur_tgt, self.cur_last = o, tgt, last
        self.placed.add(o)

        self.base_t = s[ROB:ROB + 3].copy()
        self.plan_q = self.QT - self.bias
        self.plan_p = self._fk(self.plan_q)
        self.plan_psi = self._tip_yaw(self.plan_q)
        c = s[o:o + 3].copy()
        hz = s[o + 15] / 2
        tz = self.table_z
        psi = self._grasp_psi(s, o)
        top = c[2] + hz
        pre = np.array([c[0], c[1], max(top + PRE_CLR, tz + 0.13)])
        grasp = np.array([c[0], c[1], c[2] - hz + GRASP_H])
        if self.plan_p[2] < pre[2] - 0.01:
            up = self.plan_p.copy()
            up[2] = pre[2]
            self._add_lin(up, self.plan_psi, self.grip, tol=0.05, step=0.05)
        self._add_lin(pre, psi, 0.0, tol=PRE_TOL, step=0.06)
        self._add_lin(grasp, psi, 0.0, tol=0.004, step=0.02, hold=1, vmax=DESC_V)
        self._add_grip(1.0, 3)
        rel_z = GRASP_H
        self.place_top = tz + 2 * hz
        carry_bot = tz + 0.03
        seg_a, seg_b = c[:2], tgt
        for x in (BOWL, other):
            px = s[x:x + 2]
            ab = seg_b - seg_a
            tt = np.clip(np.dot(px - seg_a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            if np.linalg.norm(seg_a + tt * ab - px) < 0.1:
                carry_bot = max(carry_bot, s[x + 2] + s[x + 15] / 2 + 0.02)
        lift = np.array([c[0], c[1], carry_bot + rel_z])
        self._add_lin(lift, psi, 1.0, tol=0.05, step=0.03)
        above = np.array([tgt[0], tgt[1], carry_bot + rel_z])
        self._add_lin(above, psi, 1.0, tol=0.04, step=0.03)
        place = np.array([tgt[0], tgt[1], tz + 0.004 + rel_z])
        self._add_lin(place, psi, 1.0, tol=0.006, step=0.02, hold=1)
        if last:
            self.segs.append(dict(kind='wait', grip=1.0, n=10))
        self._add_grip(0.0, 2)
        up = place.copy()
        up[2] = tz + 2 * hz + 0.02
        self._add_lin(up, psi, 0.0, tol=0.05, step=0.05)
        self.segs.append(dict(kind='check'))
        return True

    # ------------------------------------------------------------ control
    def _start_seg(self, s):
        sg = self.seg
        sg['k'] = 0
        if sg['kind'] == 'path':
            q0 = self.QT - self.bias
            pts = [q0] + sg['qs']
            traj = []
            for i in range(1, len(pts)):
                d = np.max(np.abs(pts[i] - pts[i - 1]))
                m = max(1, int(np.ceil(d / sg['vmax'])))
                for j in range(1, m + 1):
                    traj.append(pts[i - 1] + (pts[i] - pts[i - 1]) * j / m)
            sg['traj'] = traj

    def get_action(self, state):
        s = np.asarray(state, float)
        self.t += 1
        for o in OBJS:
            self.hist[o].append(s[o:o + 2].copy())
        a = np.zeros(11, np.float32)
        db = self.base_t[:2] - s[ROB:ROB + 2]
        a[0:2] = np.clip(db / 0.87, -0.1, 0.1)
        a[2] = np.clip(wrap(self.base_t[2] - s[ROB + 2]), -0.1, 0.1)

        guard = 0
        while self.seg is None and guard < 5:
            guard += 1
            if not self.segs:
                if not self._plan(s):
                    a[10] = self.grip
                    return a
            self.seg = self.segs.pop(0)
            if self.seg['kind'] == 'check':
                self.seg = None
                continue
            self._start_seg(s)
            if DEBUG:
                print('t', self.t, 'seg', self.seg['kind'], 'obj', s[self.cur_obj:self.cur_obj + 3].round(3),
                      'tip', self._fk(s[Q]).round(3), 'base', s[ROB:ROB+3].round(3))
        sg = self.seg
        if sg is None:
            a[10] = self.grip
            return a
        sg['k'] += 1
        q = s[Q]
        qd = None
        settle = False
        done = False
        self.grip = sg.get('grip', self.grip)
        if sg['kind'] in ('grip', 'wait'):
            done = sg['k'] >= sg['n']
        else:
            tr = sg['traj']
            k = sg['k'] - 1
            qd = tr[min(k, len(tr) - 1)]
            settle = k >= len(tr) - 1
            if k >= len(tr) - 1:
                err = np.max(np.abs(tr[-1] - q))
                if (sg['tol'] is None or err < sg['tol']) and k >= len(tr) - 1 + sg['hold']:
                    done = True
                if k > len(tr) + 60:
                    done = True
        if qd is not None:
            e = qd - q
            if settle:
                self.bias = np.clip(self.bias + 0.15 * e, -0.15, 0.15)
            cmd = qd + self.bias
            a[3:10] = np.clip(4.0 * (cmd - self.QT), -0.1, 0.1)
        a[10] = self.grip
        self.QT += 0.25 * a[3:10].astype(float)
        if done:
            self.seg = None
        return a
