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

PLACE_DIST = 0.105      # centre-centre distance from bowl for placement
OK_DIST = 0.13         # considered satisfied below this (Linf if SQUARE)
DESC_TOL = 0.015
DESC_HOLD = 1
GRIP_N = 2
PLACE_TOL = 0.012
RELEASE_N = 1
PRE_MIN = 0.10
MERGE_DESC = False
FIRST_CARRY = 0.015
CYL = True
LOW_TOL = 0.004
IDLE_MAX = 15
RETIME = True           # uniform time scaling along joint path
SQUARE = True           # goal region is a square (max-norm) around the bowl
GRIP_OFF = 0.0
TURN_BASE = True
YAW_MIX = 0.5
MOVE_BASE = True
JOINT_TRANSIT = True
CARRY_BASE = True
PLACE_DZ = 0.004
LAST_CARRY = 0.015
BIAS_K = 0.25
R0 = 0.5
BX_MAX = -0.165
ORDER = 1.0
GRASP_H = 0.07
VMAX = 0.024            # joint speed (rad/step) for interpolation (limit .025)
PRE_TOL = 0.01
DESC_V = 0.024
PRE_CLR = 0.01
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
        pt, Rt = self._to_arm(pw), rotz(-b[2]) @ rdown(psi)
        q, err = ik_arm(pt, Rt, q0, iters=150, rest=READY)
        if err > 0.004:
            best = (err, q)
            for d7 in (0.0, np.pi / 2, -np.pi / 2, np.pi, -np.pi):
                sd = np.array(READY, float)
                sd[6] = np.clip(sd[6] + d7, -2.9, 2.9)
                q2, e2 = ik_arm(pt, Rt, sd, iters=200, rest=READY)
                if e2 < best[0] - 1e-6:
                    best = (e2, q2)
                if e2 < 0.002:
                    break
            q = best[1]
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
        self.base_cmd = self.base_t.copy()
        self.table_z = float(np.median([s[o + 2] - self._geom(s, o)[0] for o in (BOWL, DRINK, CAN)]))
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
        self.idle = 0
        self.ok = OK_DIST
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
        dv = s[o:o + 2] - s[BOWL:BOWL + 2]
        d = np.max(np.abs(dv)) if SQUARE else np.linalg.norm(dv)
        on = s[o + 2] - self._geom(s, o)[0] < self.table_z + 0.02
        return d < self.ok and on

    def _embedded(self, s, o):
        return s[o + 2] - self._geom(s, o)[0] < self.table_z - 0.015

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
                u = np.array([np.cos(a), np.sin(a)])
                rr = PLACE_DIST / np.max(np.abs(u)) if SQUARE else PLACE_DIST
                tgt = bowl + rr * u
                if not (0.1 <= tgt[0] <= 0.62 and -0.45 <= tgt[1] <= 0.45):
                    continue
                if other_target is not None and np.linalg.norm(tgt - other_target) < 0.10:
                    continue
                if any(np.linalg.norm(tgt - u) < 0.10 for u in others):
                    continue
                return tgt
        return bowl + PLACE_DIST * d / (np.linalg.norm(d) + 1e-9)

    def _geom(self, s, o):
        """half height (world), grasp closing angle (or None = any), is_upright"""
        w, x, y, z = s[o + 3:o + 7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        ext = s[o + 13:o + 16]
        hz = 0.5 * float(np.sum(np.abs(R[2]) * ext))
        if o == CAN and CYL:
            cz = abs(R[2, 2])
            hz = 0.5 * (cz * ext[2] + np.sqrt(max(0.0, 1 - cz * cz)) * max(ext[0], ext[1]))
        vi = int(np.argmax(np.abs(R[2])))
        hor = [j for j in range(3) if j != vi]
        j = min(hor, key=lambda k: ext[k])
        phi = np.arctan2(R[1, j], R[0, j])
        return hz, phi, vi == 2

    def _grasp_psi(self, s, o, phi, upright):
        cur = self.plan_psi
        if o == CAN and upright:
            return cur
        step = np.pi / 2 if upright else np.pi
        cands = [phi + GRIP_OFF + k * step for k in range(-4, 5)]
        return min(cands, key=lambda c: abs(wrap(c - cur)))

    # ------------------------------------------------------------ planning
    def _add_lin(self, p1, psi, grip, tol=None, step=0.03, hold=0, vmax=VMAX, joint=False):
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
        if joint:
            qs = qs[-1:]
        self.segs.append(dict(kind='path', qs=qs, grip=grip, tol=tol, hold=hold, vmax=vmax,
                              base=self.base_t.copy()))
        self.plan_q = q
        self.plan_p = p1
        self.plan_psi = psi

    def _add_grip(self, grip, n):
        self.segs.append(dict(kind='grip', grip=grip, n=n))

    def _plan(self, s):
        """Plan next pick-and-place. Returns False if nothing to do."""
        todo = [o for o in OBJS if not self._satisfied(s, o)]
        if not todo:
            # believed done but episode continues -> belief too optimistic
            self.idle += 1
            if self.idle > IDLE_MAX:
                self.idle = 0
                self.ok = max(0.09, self.ok - 0.01)
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
            todo.sort(key=lambda o: ORDER * np.linalg.norm(s[o:o + 2] - tip[:2]))
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
        aim = s[o:o + 2] * (1 - YAW_MIX) + tgt * YAW_MIX
        if MOVE_BASE and CARRY_BASE:
            self.base_t = np.array([min(s[o] - R0, BX_MAX), s[o + 1], 0.0])
        elif MOVE_BASE:
            self.base_t = np.array([min(aim[0] - R0, BX_MAX), aim[1], 0.0])
        elif TURN_BASE:
            self.base_t[2] = np.arctan2(aim[1] - self.base_t[1], aim[0] - self.base_t[0])
        self.plan_q = self.QT - self.bias
        self.plan_p = self._fk(self.plan_q)
        self.plan_psi = self._tip_yaw(self.plan_q)
        c = s[o:o + 3].copy()
        hz, phi, upright = self._geom(s, o)
        tz = self.table_z
        psi = self._grasp_psi(s, o, phi, upright)
        top = c[2] + hz
        pre = np.array([c[0], c[1], max(top + PRE_CLR, tz + PRE_MIN)])
        rel_z = GRASP_H if 2 * hz > 0.09 else max(0.012, min(2 * hz - 0.025, hz - 0.008))
        grasp = np.array([c[0], c[1], c[2] - hz + rel_z])
        need_lift = False
        a0, b0 = self.plan_p[:2], pre[:2]
        for x in (BOWL, DRINK, CAN):
            if x == o:
                continue
            px = s[x:x + 2]
            ab = b0 - a0
            tt = np.clip(np.dot(px - a0, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            if np.linalg.norm(a0 + tt * ab - px) < 0.1 and \
                    s[x + 2] + self._geom(s, x)[0] + 0.02 > self.plan_p[2]:
                need_lift = True
        if abs(self.plan_p[2] - pre[2]) > 0.01 and np.linalg.norm(a0 - s[o:o + 2]) < 0.08:
            need_lift = True
        if need_lift and self.plan_p[2] < pre[2] - 0.01:
            up = self.plan_p.copy()
            up[2] = pre[2]
            self._add_lin(up, self.plan_psi, self.grip, tol=0.05, step=0.05)
        self._add_lin(pre, psi, 0.0, tol=PRE_TOL, step=0.06, joint=JOINT_TRANSIT)
        self._add_lin(grasp, psi, 0.0, tol=DESC_TOL if 2 * hz > 0.09 else LOW_TOL, step=0.02, hold=DESC_HOLD, vmax=DESC_V)
        if MERGE_DESC and self.segs[-2]['kind'] == 'path':
            dsc = self.segs.pop()
            self.segs[-1]['qs'] = self.segs[-1]['qs'] + dsc['qs']
            self.segs[-1]['tol'], self.segs[-1]['hold'] = dsc['tol'], dsc['hold']
        self._add_grip(1.0, GRIP_N)
        self.place_top = tz + 2 * hz
        carry_bot = tz + (LAST_CARRY if last else FIRST_CARRY)
        seg_a, seg_b = c[:2], tgt
        for x in (BOWL, other):
            px = s[x:x + 2]
            ab = seg_b - seg_a
            tt = np.clip(np.dot(px - seg_a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            if np.linalg.norm(seg_a + tt * ab - px) < 0.1:
                carry_bot = max(carry_bot, s[x + 2] + self._geom(s, x)[0] + 0.02)
        lift = np.array([c[0], c[1], carry_bot + rel_z])
        self._add_lin(lift, psi, 1.0, tol=0.05, step=0.03)
        above = np.array([tgt[0], tgt[1], carry_bot + rel_z])
        if MOVE_BASE and CARRY_BASE:
            nb = self.base_t.copy()
            nb[:2] += tgt - c[:2]
            nb[0] = np.clip(nb[0], BASE_X, BX_MAX)
            self.plan_p = self.plan_p + np.array([nb[0] - self.base_t[0], nb[1] - self.base_t[1], 0.0])
            self.base_t = nb
            self._add_lin(above, psi, 1.0, tol=0.04, step=0.03, joint=True)
        else:
            self._add_lin(above, psi, 1.0, tol=0.04, step=0.03)
        place = np.array([tgt[0], tgt[1], tz + (PLACE_DZ if last else 0.004) + rel_z])
        self._add_lin(place, psi, 1.0, tol=PLACE_TOL, step=0.02, hold=1)
        if last:
            self.segs.append(dict(kind='wait', grip=1.0, n=10))
        self._add_grip(0.0, RELEASE_N)
        up = place.copy()
        up[2] = tz + 2 * hz + 0.02
        self._add_lin(up, psi, 0.0, tol=0.05, step=0.05)
        self.segs.append(dict(kind='check'))
        return True

    # ------------------------------------------------------------ control
    def _start_seg(self, s):
        sg = self.seg
        sg['k'] = 0
        if 'base' in sg:
            self.base_cmd = sg['base']
        if sg['kind'] == 'path':
            q0 = self.QT - self.bias
            pts = [q0] + sg['qs']
            traj = []
            if RETIME:
                P = np.array(pts)
                dl = np.max(np.abs(np.diff(P, axis=0)), axis=1)
                L = np.concatenate([[0.0], np.cumsum(dl)])
                T = max(1, int(np.ceil(L[-1] / sg['vmax'] - 1e-9)))
                for j in range(1, T + 1):
                    x = L[-1] * j / T
                    i = min(int(np.searchsorted(L, x, side='left')), len(L) - 1)
                    i = max(i, 1)
                    f = 1.0 if dl[i - 1] < 1e-12 else (x - L[i - 1]) / dl[i - 1]
                    traj.append(P[i - 1] + (P[i] - P[i - 1]) * min(max(f, 0.0), 1.0))
            else:
                for i in range(1, len(pts)):
                    d = np.max(np.abs(pts[i] - pts[i - 1]))
                    m = max(1, int(np.ceil(d / sg['vmax'])))
                    for j in range(1, m + 1):
                        traj.append(pts[i - 1] + (pts[i] - pts[i - 1]) * j / m)
            sg['traj'] = traj
            if DEBUG:
                print('   seg dq', np.abs(pts[-1] - pts[0]).round(2), 'n', len(traj))

    def get_action(self, state):
        s = np.asarray(state, float)
        self.t += 1
        for o in OBJS:
            self.hist[o].append(s[o:o + 2].copy())
        a = np.zeros(11, np.float32)
        bc = self.base_cmd
        db = (bc[:2] - s[ROB:ROB + 2]) / 0.87
        m = np.max(np.abs(db))
        a[0:2] = db * (0.1 / m) if m > 0.1 else db
        a[2] = np.clip(wrap(bc[2] - s[ROB + 2]), -0.1, 0.1)

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
                berr = np.linalg.norm(sg['base'][:2] - s[ROB:ROB + 2]) if 'base' in sg else 0.0
                if (sg['tol'] is None or err < sg['tol']) and k >= len(tr) - 1 + sg['hold'] and berr < 0.01:
                    done = True
                if k > len(tr) + 60:
                    done = True
        if qd is not None:
            e = qd - q
            if settle:
                self.bias = np.clip(self.bias + BIAS_K * e, -0.15, 0.15)
            cmd = qd + self.bias
            a[3:10] = np.clip(4.0 * (cmd - self.QT), -0.1, 0.1)
        a[10] = self.grip
        self.QT += 0.25 * a[3:10].astype(float)
        if done:
            if DEBUG:
                print('   done', sg['kind'], 'k', sg['k'], 'n', len(sg.get('traj', [])), 'tol', sg.get('tol'))
            self.seg = None
        return a
