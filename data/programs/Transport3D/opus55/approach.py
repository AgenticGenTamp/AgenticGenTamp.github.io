"""Transport3D approach: pick every cube and the box and place them on the table.

Calibrated (empirically) kinematic model:
  * Kinova Gen3 arm (URDF kinematics, end_effector_link), mounted at base frame
    offset (M_OFF, 0, H).  Base frame = (pos_base_x, pos_base_y, pos_base_rot).
  * Base deltas are in world frame; only the arm / held objects collide.
  * Closing the gripper grasps an object between the fingers; opening only
    releases if the held object rests on a surface.
"""
import math
import numpy as np

M_OFF = 0.12
H = 0.2748
PI = math.pi
JNAMES = ['joint_%d' % i for i in range(1, 8)]
JLIM = np.array([1e9, 2.35, 1e9, 2.6, 1e9, 2.18, 1e9])
MAXD = 0.2


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _T(xyz, R):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = xyz
    return M


def rotz(q):
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


_JOINTS = [((0, 0, 0.15643), (PI, 0, 0)), ((0, 0.005375, -0.12838), (PI / 2, 0, 0)),
           ((0, -0.21038, -0.006375), (-PI / 2, 0, 0)), ((0, 0.006375, -0.21038), (PI / 2, 0, 0)),
           ((0, -0.20843, -0.006375), (-PI / 2, 0, 0)), ((0, 0.00017505, -0.10593), (PI / 2, 0, 0)),
           ((0, -0.10593, -0.00017505), (-PI / 2, 0, 0))]
_JT = [_T(x, _rpy(*r)) for x, r in _JOINTS]
_TEE = _T((0, 0, -0.061525), _rpy(PI, 0, 0))
RDOWN = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.]])


def fk_full(q):
    M = np.eye(4)
    axes, origins = [], []
    for i in range(7):
        M = M @ _JT[i]
        axes.append(M[:3, 2].copy())
        origins.append(M[:3, 3].copy())
        Rz = np.eye(4)
        Rz[:3, :3] = rotz(q[i])
        M = M @ Rz
    M = M @ _TEE
    return M, axes, origins


def fk(q):
    return fk_full(q)[0]


def rot_err(Rc, Rt):
    E = Rt @ Rc.T
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def ik(p, R, q0, iters=60):
    q = np.array(q0, dtype=float)
    best = None
    for _ in range(iters):
        M, axes, origins = fk_full(q)
        ep = p - M[:3, 3]
        er = rot_err(M[:3, :3], R)
        ne, nr = np.linalg.norm(ep), np.linalg.norm(er)
        if best is None or ne + 0.1 * nr < best[0]:
            best = (ne + 0.1 * nr, q.copy(), ne, nr)
        if ne < 2e-4 and nr < 2e-3:
            break
        J = np.zeros((6, 7))
        for i in range(7):
            J[:3, i] = np.cross(axes[i], M[:3, 3] - origins[i])
            J[3:, i] = axes[i]
        e = np.concatenate([ep, 0.5 * er])
        J[3:] *= 0.5
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
        n = np.max(np.abs(dq))
        if n > 0.4:
            dq *= 0.4 / n
        q = np.clip(q + dq, -JLIM, JLIM)
    return best[1], best[2], best[3]


def wrap(a):
    return (a + PI) % (2 * PI) - PI


def quat_yaw(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def quat_mat(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------------------------ utils
    def _objs(self, s):
        out = {}
        for n in s.get_object_names():
            out[n] = s.get_object_from_name(n)
        return out

    def _robot(self):
        s = self.state
        r = self.objs['robot']
        base = np.array([s.get(r, 'pos_base_x'), s.get(r, 'pos_base_y'), s.get(r, 'pos_base_rot')])
        q = np.array([s.get(r, j) for j in JNAMES])
        return base, q

    def _grasped(self):
        return self.state.get(self.objs['robot'], 'grasp_active') > 0.5

    def _obj(self, name):
        s = self.state
        o = self.objs[name]
        g = lambda f: float(s.get(o, f))
        return dict(x=g('pose_x'), y=g('pose_y'), z=g('pose_z'),
                    q=(g('pose_qx'), g('pose_qy'), g('pose_qz'), g('pose_qw')),
                    hx=g('half_extent_x'), hy=g('half_extent_y'), hz=g('half_extent_z'),
                    ga=g('grasp_active'))

    def _ee_world(self, base, q):
        M = fk(q)
        Rb = rotz(base[2])
        p = np.array([base[0], base[1], 0.]) + Rb @ (M[:3, 3] + np.array([M_OFF, 0, 0])) + np.array([0, 0, H])
        return p, Rb @ M[:3, :3]

    def _world_to_arm(self, base, p):
        Rb = rotz(base[2])
        return Rb.T @ (np.asarray(p) - np.array([base[0], base[1], H])) - np.array([M_OFF, 0, 0])

    def _act(self, dbase=None, dq=None, grip=0.0):
        a = np.zeros(11, dtype=np.float32)
        if dbase is not None:
            a[:3] = np.clip(dbase, -MAXD, MAXD)
        if dq is not None:
            a[3:10] = dq
        a[10] = grip
        return a

    @staticmethod
    def _scale(d, m=MAXD):
        n = np.max(np.abs(d)) if len(d) else 0.0
        if n > m:
            d = d * (m / n)
        return d

    # --------------------------------------------------------------- episode
    def reset(self, state, info):
        self.state = state
        self.objs = self._objs(state)
        tab = self._obj('table')
        self.table = (tab['x'] - tab['hx'], tab['x'] + tab['hx'], tab['y'] - tab['hy'], tab['y'] + tab['hy'],
                      tab['z'] + tab['hz'])
        self.gen = self._plan()
        self.steps = 0

    def get_action(self, state):
        self.state = state
        self.objs = self._objs(state)
        self.steps += 1
        try:
            a = next(self.gen)
        except StopIteration:
            a = self._act()
        return np.asarray(a, dtype=np.float32)

    # --------------------------------------------------------- geometry/plan
    def _in_obst(self, p, infl):
        x0, x1, y0, y1, _ = self.table
        return x0 - infl < p[0] < x1 + infl and y0 - infl < p[1] < y1 + infl

    def _seg_hits(self, a, b, infl):
        x0, x1, y0, y1, _ = self.table
        x0 -= infl; x1 += infl; y0 -= infl; y1 += infl
        # Liang-Barsky
        dx, dy = b[0] - a[0], b[1] - a[1]
        t0, t1 = 0.0, 1.0
        for p, qv in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
            if abs(p) < 1e-12:
                if qv < 0:
                    return False
            else:
                t = qv / p
                if p < 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
                if t0 > t1:
                    return False
        return t1 - t0 > 1e-6

    NAV_INFL = 0.2
    FORCE_BOX_FIRST = False
    CUBE_H = 0.05
    CLOSE_LONG = False
    CORNER_LAYOUT = False
    DEBUG = False
    GRIP_WITH_MOVE = False
    BOX_NEAR = True
    NO_LIFT_LAST = True
    LAST_LIFT = 0.085
    DEST_CHOICE = True
    TABLE_BIAS = 0.0
    ARM_COST = True
    PLACE_DZ = 0.0005
    HIGH_CARRY = False
    DD_MAX = PI / 6
    HOLD_T = 0.4
    HOLD_N = 1
    HOLD_M = 0.035
    SLOTS_OVERRIDE = None
    PLACE_EPS = 0.003

    def _path(self, a, b):
        infl = self.NAV_INFL
        a = np.array(a[:2], float); b = np.array(b[:2], float)
        if not self._seg_hits(a, b, infl - 1e-3):
            return [b]
        x0, x1, y0, y1, _ = self.table
        e = infl + 0.03
        corners = [np.array(c) for c in ((x0 - e, y0 - e), (x1 + e, y0 - e), (x1 + e, y1 + e), (x0 - e, y1 + e))]
        nodes = [a] + corners + [b]
        n = len(nodes)
        dist = [1e18] * n; prev = [-1] * n; dist[0] = 0; done = [False] * n
        for _ in range(n):
            u = min((i for i in range(n) if not done[i]), key=lambda i: dist[i])
            done[u] = True
            for v in range(n):
                if done[v] or v == u:
                    continue
                if self._seg_hits(nodes[u], nodes[v], infl - 1e-3):
                    continue
                d = dist[u] + np.linalg.norm(nodes[u] - nodes[v])
                if d < dist[v]:
                    dist[v] = d; prev[v] = u
        if dist[n - 1] >= 1e17:
            return [b]
        path = []
        v = n - 1
        while v != 0:
            path.append(nodes[v]); v = prev[v]
        return path[::-1]

    def _path_len(self, a, b):
        pts = [np.array(a[:2], float)] + self._path(a, b)
        return sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))

    # ----------------------------------------------------------- primitives
    def _arm_R(self, yaw_a):
        return rotz(yaw_a) @ RDOWN

    def _move(self, base_goal=None, q_goal=None, tol=1e-4, max_steps=200, split=False, q_seq=None, need=None):
        """Drive base (along a table-avoiding path) and arm joints concurrently."""
        waypoints = None
        stuck = 0
        k = 0
        pdiff = None
        for it in range(max_steps):
            base, q = self._robot()
            db = np.zeros(3)
            if base_goal is not None:
                if waypoints is None:
                    waypoints = self._path(base, base_goal)
                while len(waypoints) > 1 and np.linalg.norm(waypoints[0] - base[:2]) < 1e-3:
                    waypoints.pop(0)
                tgt = waypoints[0]
                db[:2] = tgt - base[:2]
                if len(waypoints) > 1 and np.max(np.abs(db[:2])) < MAXD:
                    # reach waypoint exactly then continue next step
                    pass
                db[2] = wrap(base_goal[2] - base[2])
            dq = np.zeros(7)
            if q_goal is not None:
                use = q_seq is not None and k < len(q_seq) and not split
                qt = q_seq[k] if use else q_goal
                dq = self._scale(qt - q)
                if use:
                    diff = float(np.max(np.abs(qt - q)))
                    nz = max(need[k + 1:k + 1 + self.HOLD_N]) if (need is not None and self.HOLD_N and k + 1 < len(need)) else 0.0
                    ez = fk(q)[2, 3] + H
                    if ez < nz - self.HOLD_M and diff > MAXD + 1e-6 and (pdiff is None or pdiff - diff > 0.02):
                        db[:] = 0.0  # arm lags the schedule: hold the base
                        if self.DEBUG: print("HOLD", k, round(ez, 3), round(nz, 3), round(diff, 3), np.round(base, 2))
                        pdiff = diff
                    else:
                        k += 1
                        pdiff = None
            done_b = base_goal is None or (np.max(np.abs(np.r_[base_goal[:2] - base[:2], db[2]])) < tol)
            done_q = q_goal is None or np.max(np.abs(q_goal - q)) < tol
            if done_b and done_q:
                return True
            yield self._act(db, dq)
            base2, q2 = self._robot()
            if np.max(np.abs(base2 - base)) < 1e-9 and np.max(np.abs(q2 - q)) < 1e-9:
                stuck += 1
                if stuck >= 2:
                    if split or base_goal is None or q_goal is None:
                        return False
                    ok = yield from self._move(None, q_goal, tol, max_steps, split=True)
                    if not ok:
                        return False
                    ok = yield from self._move(base_goal, None, tol, max_steps, split=True)
                    return ok
            else:
                stuck = 0
        return False

    def _lin(self, p_goal, R, seg=0.04, tol=5e-4, grip_last=0.0):
        """Straight-line EE motion (arm frame) with fixed orientation.

        A dense joint-space path is computed by IK along the line; each step
        advances as far along it as the joint-delta limit allows."""
        base, q = self._robot()
        p0 = fk(q)[:3, 3]
        dist = np.linalg.norm(p_goal - p0)
        n = max(1, int(math.ceil(dist / 0.01)))
        path = []
        qc = q.copy()
        for i in range(1, n + 1):
            wp = p0 + (p_goal - p0) * i / n
            qc, e1, e2 = ik(wp, R, qc, iters=30 if i < n else 80)
            if e1 > 1e-3 or e2 > 1e-2:
                qc, e1, e2 = ik(wp, R, qc, iters=100)
            path.append(qc.copy())
        k = -1  # index of the last path point reached
        fails = 0
        for _ in range(4 * n + 10):
            base, q = self._robot()
            if k >= len(path) - 1 and np.max(np.abs(path[-1] - q)) < 1e-4:
                return True
            j = max(k, 0)
            while j + 1 < len(path) and np.max(np.abs(path[j + 1] - q)) <= MAXD - 1e-6:
                j += 1
            if fails:
                j = min(k + 1, len(path) - 1)
            d = self._scale(path[j] - q)
            if np.max(np.abs(d)) < 1e-4:
                k = j
                continue
            yield self._act(None, d, grip=(grip_last if (j == len(path) - 1 and np.max(np.abs(path[j] - q - d)) < 1e-6) else 0.0))
            base2, q2 = self._robot()
            if np.max(np.abs(q2 - q)) < 1e-9:
                fails += 1
                if fails >= 2:
                    return False
                continue
            fails = 0
            if np.max(np.abs(path[j] - q2)) < 1e-4:
                k = j
        return True

    # ------------------------------------------------------------ main plan
    def _plan(self):
        cubes = sorted(n for n in self.objs if n.startswith('cube'))
        boxes = sorted(n for n in self.objs if n.startswith('box'))
        box = boxes[0] if boxes else None
        # slots inside the box (box-local coordinates), along the long axis
        self.in_box_slots = []
        if box is not None:
            b = self._obj(box)
            long_is_y = b['hy'] >= b['hx']
            hl = max(b['hx'], b['hy'])
            n_along = max(1, int((2 * (hl - 0.045)) // 0.08) + 1)
            span = (n_along - 1) * 0.08
            for i in range(n_along):
                v = -span / 2 + i * 0.08
                self.in_box_slots.append(np.array([0.0, v]) if long_is_y else np.array([v, 0.0]))
            # fill the middle first
            self.in_box_slots.sort(key=lambda p: np.linalg.norm(p))
            if self.SLOTS_OVERRIDE is not None:
                self.in_box_slots = [np.array(p) for p in self.SLOTS_OVERRIDE]
            self.in_box_slots = [(p, self.CLOSE_LONG) for p in self.in_box_slots]
            hs = min(b['hx'], b['hy'])
            cs = hs - 0.03 - 0.035
            rl = hl - 0.03 - 0.05
            if self.CORNER_LAYOUT and len(cubes) > len(self.in_box_slots) and cs >= 0.025 and self.SLOTS_OVERRIDE is None:
                # 2x2 corners (fingers close along the long axis) + centre (short axis)
                sl = []
                for a in (1, -1):
                    for c in (1, -1):
                        sl.append((np.array([c * cs, a * rl]) if long_is_y else np.array([a * rl, c * cs]), True))
                sl.append((np.zeros(2), False))
                self.in_box_slots = sl
        x0, x1, y0, y1, ztop = self.table
        xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        self.table_slots = []
        dxs = (0.0, -0.09, 0.09)
        if len(cubes) > len(self.in_box_slots) + 12:
            dxs = (0.0, -0.075, 0.075, -0.15, 0.15)
        for dy in (0.3, 0.22):
            for sg in (1, -1):
                for dx in dxs:
                    p = np.array([xc + dx, yc + sg * dy])
                    if x0 + 0.04 < p[0] < x1 - 0.04 and y0 + 0.04 < p[1] < y1 - 0.04:
                        self.table_slots.append(p)
        failures = {}
        remaining = list(cubes)
        box_done = False
        if box is not None:
            bo = self._obj(box)
            if self.FORCE_BOX_FIRST or any(self._clear(bo, [self._obj(c)['x'], self._obj(c)['y']]) < 0.075 for c in cubes):
                # a cube hugs the box: slide the box away on the floor first
                for _ in range(3):
                    ok = yield from self._transport(box, 'floor', None)
                    if ok:
                        break
                    yield from self._recover()
        while remaining:
            best = None
            cand = list(remaining)
            if len(cand) > 8:
                b0, _ = self._robot()
                cand.sort(key=lambda n: np.hypot(self._obj(n)['x'] - b0[0], self._obj(n)['y'] - b0[1]))
                cand = cand[:8]
            for n in cand:
                for dst in (('box', 'table') if (box is not None and self.in_box_slots) else ('table',)):
                    c = self._est(n, dst, box)
                    if best is None or c < best[0]:
                        best = (c, n, dst)
            _, name, dst = best
            if not self.DEST_CHOICE:
                base, q = self._robot()
                remaining.sort(key=lambda n: np.hypot(self._obj(n)['x'] - base[0], self._obj(n)['y'] - base[1]))
                name = remaining[0]
                dst = 'box' if (box is not None and self.in_box_slots) else 'table'
            ok = yield from self._transport(name, dst, box)
            if ok:
                remaining.remove(name)
            else:
                failures[name] = failures.get(name, 0) + 1
                if failures[name] > 3:
                    remaining.remove(name)
                yield from self._recover()
        for bname in boxes:
            if box_done and bname == box and not self.FORCE_BOX_FIRST:
                continue
            for _ in range(4):
                ok = yield from self._transport(bname, 'table', None)
                if ok:
                    break
                yield from self._recover()
        while True:
            yield self._act()

    def _est(self, name, dest, box):
        """Estimated step cost of transporting `name` to `dest` from the current state."""
        opts, _ = self._grasp_options(name)
        if not opts:
            return 1e9
        cg, bpose, gp, ya, ez, reach = opts[0]
        t = np.array([0.0, 0.0, 0.035])
        if dest == 'box':
            po = []
            for sl, cl in self.in_box_slots:
                po, _ = self._place_options_box(t, box, sl, cl, vbase=bpose)
                if po:
                    break
        else:
            x0, x1, y0, y1, ztop = self.table
            tgt = self.table_slots[0] if self.table_slots else np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
            po = self._place_options_table(t, ya, tgt, vbase=bpose)
        if not po:
            return 1e9
        return cg + po[0][0] + (self.TABLE_BIAS if dest == 'table' else 0.0)

    def _recover(self):
        if self._grasped():
            yield self._act(grip=1.0)
        if self._grasped():
            # still holding: lower straight down onto whatever is below and release
            base, q = self._robot()
            r = self.objs['robot']
            s = self.state
            ew, _ = self._ee_world(base, q)
            x0, x1, y0, y1, ztop = self.table
            below = ztop if self._in_obst(ew[:2], 0.0) else 0.0
            held = None
            for n in self.objs:
                if n.startswith('cube') or n.startswith('box'):
                    o = self._obj(n)
                    if abs(o['x'] - ew[0]) < 0.12 and abs(o['y'] - ew[1]) < 0.12 and o['z'] > o['hz'] + 0.01:
                        held = o
            if held is not None:
                drop = held['z'] - held['hz'] - below - 0.002
                p = fk(q)[:3, 3]
                R = fk(q)[:3, :3]
                yield from self._lin(p - np.array([0, 0, drop]), R, seg=0.1)
                for _ in range(8):
                    yield self._act(grip=1.0)
                    if not self._grasped():
                        break
                    base, q = self._robot()
                    p = fk(q)[:3, 3]
                    yield from self._lin(p - np.array([0, 0, 0.003]), R)
        base, q = self._robot()
        p = fk(q)[:3, 3]
        R = fk(q)[:3, :3]
        yield from self._lin(np.array([p[0], p[1], max(p[2], 0.35)]), R, seg=0.1)

    def _grasp_options(self, name, box_grasp_long=True):
        """Candidate (cost, base_pose, grasp_point, yaw_a, ee_z, reach)."""
        o = self._obj(name)
        base, q = self._robot()
        psi = quat_yaw(*o['q'])
        is_box = max(o['hx'], o['hy']) > 0.06
        cands = []
        if is_box:
            for k in range(4):
                th = wrap(psi + k * PI / 2)
                h_n = o['hx'] if k % 2 == 0 else o['hy']
                h_p = o['hy'] if k % 2 == 0 else o['hx']
                if box_grasp_long and h_n > h_p + 1e-6:
                    continue  # grasp a long wall (normal along the short axis)
                n = np.array([math.cos(th), math.sin(th)])
                gp = np.array([o['x'], o['y']]) - (h_n - 0.002) * n
                cands.append((th, gp, PI / 2, o['z'] + o['hz']))
        else:
            ang0 = math.atan2(o['y'] - base[1], o['x'] - base[0])
            for k in range(24):
                th = wrap(ang0 + ((k + 1) // 2) * (1 if k % 2 else -1) * PI / 12)
                ya = wrap(psi - th - PI / 2)
                ya = (ya + PI / 4) % (PI / 2) - PI / 4
                gp = np.array([o['x'], o['y']])
                cands.append((th, gp, ya, o['z'] + 0.035))
        others = [self._obj(n) for n in self.objs if n != name and n != 'robot' and (n.startswith('cube') or n.startswith('box'))]
        out = []
        ya_cur = self._ya_cur(q)
        for th, gp, ya, ez in cands:
            if self._in_obst(gp, 0.07):
                continue
            pen = 0.0
            if is_box:
                cub = [ob for ob in others if ob['hz'] < 0.06]
                if cub and min(math.hypot(ob['x'] - gp[0], ob['y'] - gp[1]) for ob in cub) < 0.09:
                    pen += 2.0
            if not is_box and others:
                # pick finger yaw whose finger tips stay clear of neighbours
                best = None
                for yy in (ya, wrap(ya + PI / 2) if ya < 0 else wrap(ya - PI / 2)):
                    ca = th + yy + PI / 2
                    cdir = np.array([math.cos(ca), math.sin(ca)])
                    clr = min(self._clear(ob, gp + sgn * 0.065 * cdir) for ob in others for sgn in (1, -1))
                    if best is None or clr > best[0]:
                        best = (clr, yy)
                ya = best[1]
                if best[0] < 0.01:
                    pen += 2.0
                # arm segment clearance (mount -> grasp point) against tall objects
                for ob in others:
                    if ob['hz'] < 0.06:
                        continue
                    for f in (0.55, 0.7, 0.85):
                        p = gp - f * 0.3 * np.array([math.cos(th), math.sin(th)])
                        if self._clear(ob, p) < 0.04:
                            pen += 1.0
                            break
            if not is_box or self.BOX_NEAR:
                ya = self._near(ya, PI, ya_cur)
            for reach in (0.5, 0.42, 0.58):
                bp = gp - (M_OFF + reach) * np.array([math.cos(th), math.sin(th)])
                if self._in_obst(bp, self.NAV_INFL + 0.01):
                    continue
                cost = self._mcost(base, bp, th) + pen
                if self.ARM_COST:
                    qh = self._carry_q(reach, ez + 0.03 - H, ya, q)
                    cost = max(cost, float(np.max(np.abs(qh - q)))) + 0.03 * float(np.max(np.abs(qh - q)))
                out.append((cost, np.array([bp[0], bp[1], th]), gp, ya, ez, reach))
                break
        out.sort(key=lambda c: c[0])
        return out, is_box

    @staticmethod
    def _ya_cur(q):
        R = fk(q)[:3, :3]
        return math.atan2(-R[0, 0], R[1, 0])

    @staticmethod
    def _near(ya, period, ref):
        return ya + period * round((ref - ya) / period)

    def _sim_base(self, base, goal, max_steps=200):
        """Replicate the base controller of _move: list of base poses per step."""
        b = np.array(base, float)
        poses = [b.copy()]
        wps = list(self._path(b, goal))
        for _ in range(max_steps):
            while len(wps) > 1 and np.linalg.norm(wps[0] - b[:2]) < 1e-3:
                wps.pop(0)
            d = np.zeros(3)
            d[:2] = wps[0] - b[:2]
            d[2] = wrap(goal[2] - b[2])
            if np.max(np.abs(d)) < 1e-4:
                break
            b = b + np.clip(d, -MAXD, MAXD)
            poses.append(b.copy())
        return poses

    def _q_sched(self, poses, reach, exclude, held_r, drop, zmin, ya, q):
        """Per-step arm targets: suffix-max of needed height along the sweep."""
        raw = [self._need_z([p], reach, exclude, held_r, drop) for p in poses]
        self._raw_need = raw
        zs = [max(r, zmin) for r in raw]
        for i in range(len(zs) - 2, -1, -1):
            zs[i] = max(zs[i], zs[i + 1])
        cache = {}
        seq = []
        qc = q
        for z in zs[1:]:
            key = round(z, 3)
            if key not in cache:
                cache[key] = self._carry_q(reach, z - H, ya, qc)
                qc = cache[key]
            seq.append(cache[key])
        return seq, zs

    def _need_z(self, poses, reach, exclude, held_r, drop):
        """World EE height needed so gripper/held object clears obstacles along a sweep."""
        x0, x1, y0, y1, ztop = self.table
        obs = [self._obj(n) for n in self.objs
               if n not in exclude and (n.startswith('cube') or n.startswith('box'))]
        clr = drop + 0.03
        z = 0.0
        for b in poses:
            c, s_ = math.cos(b[2]), math.sin(b[2])
            p = np.array([b[0], b[1]]) + (M_OFF + reach) * np.array([c, s_])
            if self._in_obst(p, held_r + 0.04):
                z = max(z, ztop + clr)
            for ob in obs:
                if self._clear(ob, p) < held_r + 0.03:
                    z = max(z, ob['z'] + ob['hz'] + clr)
        return z

    def _mcost(self, base, bxy, th):
        """Approximate step cost of a base move (translation/rotation concurrent)."""
        pts = [np.asarray(base[:2])] + list(self._path(base, np.array([bxy[0], bxy[1], th])))
        tr = sum(np.max(np.abs(pts[i + 1] - pts[i])) for i in range(len(pts) - 1))
        dth = abs(wrap(th - base[2]))
        return max(tr, dth) + 0.03 * dth + 0.03 * tr

    def _clear(self, ob, p):
        """Signed-ish distance from xy point p to object's footprint rectangle."""
        psi = quat_yaw(*ob['q'])
        d = np.array([p[0] - ob['x'], p[1] - ob['y']])
        c, s_ = math.cos(psi), math.sin(psi)
        lx, ly = c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]
        dx, dy = abs(lx) - ob['hx'], abs(ly) - ob['hy']
        if dx < 0 and dy < 0:
            return max(dx, dy)
        return math.hypot(max(dx, 0), max(dy, 0))

    def _floor_shift(self, name):
        """Displacement moving object `name` to a free floor spot (base translates only)."""
        o = self._obj(name)
        base, q = self._robot()
        others = [self._obj(n) for n in self.objs if n != name and n.startswith('cube')]
        x0, x1, y0, y1, ztop = self.table
        tc = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        best = None
        for dist in (0.3, 0.4, 0.5):
            for k in range(16):
                a = 2 * PI * k / 16
                d = dist * np.array([math.cos(a), math.sin(a)])
                o2 = dict(o); o2['x'] = o['x'] + d[0]; o2['y'] = o['y'] + d[1]
                c2 = np.array([o2['x'], o2['y']])
                if self._in_obst(c2, max(o['hx'], o['hy']) + 0.08):
                    continue
                if any(self._clear(o2, [c['x'], c['y']]) < 0.13 for c in others):
                    continue
                if self._in_obst(base[:2] + d, self.NAV_INFL + 0.01):
                    continue
                if self._seg_hits(base[:2], base[:2] + d, self.NAV_INFL):
                    continue
                cost = np.linalg.norm(c2 - tc) + dist
                if best is None or cost < best[0]:
                    best = (cost, d)
            if best is not None:
                return best[1]
        return None

    def _carry_q(self, reach, zc, ya, q0):
        qt, e1, e2 = ik(np.array([reach, 0., zc]), self._arm_R(ya), q0)
        return qt

    def _place_options_table(self, off_t, ya, slot, vbase=None):
        x0, x1, y0, y1, ztop = self.table
        base, q = self._robot()
        if vbase is not None:
            base = vbase
        out = []
        for th in (0.0, PI / 2, PI, -PI / 2):
            off_w = rotz(th) @ self._arm_R(ya) @ off_t
            ee_xy = slot - off_w[:2]
            if th == 0.0:
                mount = x0 - 0.1; bxy = np.array([mount - M_OFF, ee_xy[1]]); rch = ee_xy[0] - mount
            elif th == PI:
                mount = x1 + 0.1; bxy = np.array([mount + M_OFF, ee_xy[1]]); rch = mount - ee_xy[0]
            elif th == PI / 2:
                mount = y0 - 0.1; bxy = np.array([ee_xy[0], mount - M_OFF]); rch = ee_xy[1] - mount
            else:
                mount = y1 + 0.1; bxy = np.array([ee_xy[0], mount + M_OFF]); rch = mount - ee_xy[1]
            if not (0.22 <= rch <= 0.62):
                continue
            cost = self._mcost(base, bxy, th)
            out.append((cost, th, bxy, ee_xy, rch, off_w, ya))
        out.sort(key=lambda c: c[0])
        return out

    def _place_options_box(self, off_t, box, slot_local, close_long=False, vbase=None):
        b = self._obj(box)
        psi = quat_yaw(*b['q'])
        Rb = rotz(psi)[:2, :2]
        slot = np.array([b['x'], b['y']]) + Rb @ slot_local
        # closing axis must be along the box short axis
        close_ang = psi if b['hx'] <= b['hy'] else psi + PI / 2
        if close_long:
            close_ang += PI / 2
        base, q = self._robot()
        if vbase is not None:
            base = vbase
        ang0 = math.atan2(slot[1] - base[1], slot[0] - base[0])
        ya_cur = self._ya_cur(q)
        out = []
        ths = [wrap(ang0 + ((k + 1) // 2) * (1 if k % 2 else -1) * PI / 12) for k in range(24)]
        if np.linalg.norm(slot_local) > 0.01:
            sdir0 = math.atan2(b['y'] - slot[1], b['x'] - slot[0])
            ths = [wrap(sdir0 + m * PI / 36) for m in range(-6, 7)]
        for th in ths:
            ya = wrap(close_ang - th - PI / 2)
            if np.linalg.norm(slot_local) > 0.01:
                ya = (ya + PI / 2) % PI - PI / 2  # wrist orientation matters in tight slots
            else:
                ya = self._near(ya, PI, ya_cur)
            off_w = rotz(th) @ self._arm_R(ya) @ off_t
            ee_xy = slot - off_w[:2]
            for rch in (0.5, 0.42, 0.58, 0.36, 0.3):
                bxy = ee_xy - (M_OFF + rch) * np.array([math.cos(th), math.sin(th)])
                if self._in_obst(bxy, self.NAV_INFL + 0.01):
                    continue
                cost = self._mcost(base, bxy, th)
                if np.linalg.norm(slot_local) > 0.01:
                    sdir = math.atan2(b['y'] - slot[1], b['x'] - slot[0])
                    dd = abs(wrap(th - sdir))
                    if dd > self.DD_MAX + 1e-6:
                        continue
                    cost += 0.3 * dd
                out.append((cost, th, bxy, ee_xy, rch, off_w, ya))
                break
        out.sort(key=lambda c: c[0])
        return out, b['z'] - b['hz'] + 0.019

    def _transport(self, name, dest, box):
        x0, x1, y0, y1, ztop = self.table
        o = self._obj(name)
        opts, is_box = self._grasp_options(name)
        if not opts:
            opts, is_box = self._grasp_options(name, box_grasp_long=False)
        if not opts:
            return False
        _, bpose, gp, ya, ez, reach = opts[0]
        R = self._arm_R(ya)
        base, q = self._robot()
        drop = (ez - (o['z'] - o['hz']))
        held_r = math.hypot(o['hx'], o['hy']) + 0.02
        # approach height: clear everything along the base sweep (gripper only)
        poses = self._sim_base(base, bpose)
        q_seq, zs = self._q_sched(poses, reach, {name}, 0.05, 0.07, ez + (0.1 if is_box else 0.02), ya, q)
        q_carry = q_seq[-1] if q_seq else self._carry_q(reach, zs[-1] - H, ya, q)
        # 1. drive to the grasp pose with arm in carry pose
        ok = yield from self._move(bpose, q_carry, q_seq=q_seq, need=self._raw_need)
        if not ok:
            return False
        # 2. descend to pre-grasp then grasp
        base, q = self._robot()
        pg = self._world_to_arm(base, [gp[0], gp[1], ez + 0.08])
        pgr = self._world_to_arm(base, [gp[0], gp[1], ez])
        yield from self._lin(pgr, R, seg=0.04)
        yield self._act(grip=-1.0)
        if not self._grasped():
            yield self._act(grip=1.0)
            yield from self._lin(pg, R)
            return False
        # 3. lift (placement is planned first so the lift height is known)
        # 4. placement choice
        o = self._obj(name)
        r = self.objs['robot']
        s = self.state
        t = np.array([s.get(r, 'grasp_tf_x'), s.get(r, 'grasp_tf_y'), s.get(r, 'grasp_tf_z')])
        hz = o['hz']
        slot_idx = None
        po = []
        if dest == 'box':
            for i, (sl, cl) in enumerate(self.in_box_slots):
                po, surf = self._place_options_box(t, box, sl, cl)
                if po:
                    slot_idx = i
                    break
            if not po:
                dest = 'table'
        if dest == 'box':
            pass
        elif dest == 'floor':
            base, q = self._robot()
            off_w = rotz(base[2]) @ R @ t
            d = self._floor_shift(name)
            if d is None:
                return False
            ee_w = self._ee_world(base, q)[0]
            po = [(0.0, base[2], base[:2] + d, ee_w[:2] + d, fk(q)[0, 3], off_w, ya)]
            surf = 0.0
        else:
            x0, x1, y0, y1, ztop = self.table
            tgt = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
            if not is_box and self.table_slots:
                tgt = self.table_slots[0]
            po = self._place_options_table(t, ya, tgt)
            surf = ztop
        if not po:
            return False
        _, th, bxy, ee_xy, rch, off_w, ya2 = po[0]
        if self.DEBUG:
            print("PLACE", "base", np.round(self._robot()[0], 3), "box", (np.round([self._obj(box)["x"], self._obj(box)["y"]], 3) if box else None), name, dest, "th", round(th, 3), "ya2", round(ya2, 3), "close_world", round(wrap(th + ya2 + PI / 2), 3), "t", np.round(t, 4), "off_w", np.round(off_w, 4), "ee_xy", np.round(ee_xy, 3), "surf", round(surf, 4))
        R2 = self._arm_R(ya2)
        base, q = self._robot()
        goal2 = np.array([bxy[0], bxy[1], th])
        poses2 = self._sim_base(base, goal2)
        ez_place = surf + hz + self.PLACE_DZ - off_w[2]
        z_car = max(self._need_z(poses2, rch, {name}, held_r, drop), ez_place + 0.01)
        if dest == 'floor':
            z_car = max(z_car, drop + 0.06)
        if self.HIGH_CARRY:
            z_car = max(0.62, ztop + 0.12 + drop)
            if dest == 'box':
                bb = self._obj(box)
                z_car = max(z_car, bb['z'] + bb['hz'] + 0.06 + drop)
        z_start = max(self._need_z(poses2[:1], fk(q)[0, 3], {name}, held_r, drop), ez + 0.03)
        p_now = fk(q)[:3, 3]
        yield from self._lin(np.array([p_now[0], p_now[1], max(z_start, min(z_car, z_start + 0.05)) - H]), R, seg=0.08)
        base, q = self._robot()
        zmin2 = ez_place + 0.01 if dest != 'floor' else max(ez_place + 0.01, drop + 0.06)
        q_seq2, zs2 = self._q_sched(poses2, rch, {name}, held_r, drop, zmin2, ya2, q)
        q_carry2 = q_seq2[-1] if q_seq2 else self._carry_q(rch, zs2[-1] - H, ya2, q)
        ok = yield from self._move(goal2, q_carry2, q_seq=q_seq2, need=self._raw_need)
        if not ok:
            return False
        # 5. lower to place
        base, q = self._robot()
        pp = self._world_to_arm(base, [ee_xy[0], ee_xy[1], ez_place])
        yield from self._lin(pp, R2, seg=0.08, grip_last=(1.0 if self.GRIP_WITH_MOVE else 0.0))
        for k in range(10):
            yield self._act(grip=1.0)
            if not self._grasped():
                break
            base, q = self._robot()
            p_now = fk(q)[:3, 3]
            yield from self._lin(p_now - np.array([0, 0, 0.002]), R2, seg=0.08)
        if self._grasped():
            return False
        if slot_idx is not None:
            self.in_box_slots.pop(slot_idx)
        elif dest == 'table' and not is_box and self.table_slots:
            self.table_slots.pop(0)
        if is_box and dest == "table" and self.NO_LIFT_LAST:
            base, q = self._robot()
            p_now = fk(q)[:3, 3]
            yield from self._lin(p_now + np.array([0, 0, self.LAST_LIFT]), R2, seg=0.1)
            return True
        base, q = self._robot()
        p_now = fk(q)[:3, 3]
        z_up = self._need_z([base], p_now[0], set(), 0.05, 0.07)
        if self.HIGH_CARRY:
            z_up = 0.62
        yield from self._lin(np.array([p_now[0], p_now[1], max(p_now[2] + 0.03, z_up - H)]), R2, seg=0.1)
        return True
