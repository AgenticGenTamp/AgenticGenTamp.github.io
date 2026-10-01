"""Tidybot + Kinova Gen3 policy for ConstrainedCupboard3D.

Hypothesis: rods must stand upright inside the tall compartments of the cupboard:
columns with a low board (above board) and columns with a high board (below board).
Pipeline per rod: top-down pick -> reorient upright (closed-loop tilt correction)
-> drive base forward to insert -> release -> retreat -> push deeper with fingertips.
"""
import numpy as np
from cupkin import fk, ik_best

MX, MY, MZ = 0.1199, 0.0, 0.3949
TOOL = 0.15
FRONT_X = 1.85
L_BOARD = 0.145
TOP_UNDER = 0.54


def quat_yaw(qv):
    w, x, y, z = qv
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_to_R(qv):
    w, x, y, z = qv
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def Rdown(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def world_to_arm(b, pw):
    c, s = np.cos(b[2]), np.sin(b[2]); d = pw[:2] - b[:2]
    return np.array([c * d[0] + s * d[1] - MX, -s * d[0] + c * d[1] - MY, pw[2] - MZ])


def arm_to_world(b, pa):
    c, s = np.cos(b[2]), np.sin(b[2]); x, y = pa[0] + MX, pa[1] + MY
    return np.array([b[0] + c * x - s * y, b[1] + s * x + c * y, pa[2] + MZ])


def vertical_R(R):
    y = R[:, 1]
    t = np.array([0, 0, 1.0 if y[2] >= 0 else -1.0])
    v = np.cross(y, t); s = np.linalg.norm(v); c = float(np.dot(y, t))
    if s < 1e-9:
        return R
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + K + K @ K * ((1 - c) / s ** 2)) @ R


def column_types(n_cols):
    if n_cols <= 3:
        return ['L', 'H', 'M'][:n_cols]
    if n_cols <= 6:
        return (['L'] * 3 + ['H'] * 3)[:n_cols]
    return (['L'] * 3 + ['H'] * 3 + ['M'] * 2 + ['T'] * 3)[:n_cols]


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.state = None
        self.gen = None
        self.grip = 0.0

    # ------------------------------------------------------------------ state
    def _types(self, state):
        rt = mt = ft = None
        for t in self.observation_space.types:
            if t.name == 'mujoco_tidybot_robot': rt = t
            elif t.name == 'mujoco_movable_object': mt = t
            elif t.name == 'mujoco_fixture': ft = t
        return rt, mt, ft

    def q(self):
        s = self.state
        return np.array([s.get(self.robot, 'pos_arm_joint%d' % i) for i in range(1, 8)])

    def base(self):
        s = self.state; r = self.robot
        return np.array([s.get(r, 'pos_base_x'), s.get(r, 'pos_base_y'), s.get(r, 'pos_base_rot')])

    def rod(self, name):
        s = self.state; o = self.rod_objs[name]
        return np.array([s.get(o, k) for k in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']])

    def ee_world(self):
        b = self.base(); T = fk(self.q(), TOOL)
        return arm_to_world(b, T[:3, 3]), Rz(b[2]) @ T[:3, :3]

    def grasp_tf(self, name):
        pe, Re = self.ee_world(); r = self.rod(name)
        return Re.T @ (r[:3] - pe), Re.T @ quat_to_R(r[3:7])

    # ------------------------------------------------------------------ api
    def reset(self, state, info):
        self.state = state
        rt, mt, ft = self._types(state)
        self.robot = state.get_objects(rt)[0]
        self.rod_objs = {o.name: o for o in state.get_objects(mt)}
        fx = state.get_objects(ft) if ft is not None else []
        cols = {}
        for o in fx:
            try:
                idx = int(o.name.split('_')[-1])
            except ValueError:
                continue
            cols[idx] = (state.get(o, 'x'), state.get(o, 'y'))
        types = column_types(len(cols))
        hcols = [cols[i] for i in sorted(cols) if i < len(types) and types[i] == 'H']
        if not hcols:
            hcols = [cols[i] for i in sorted(cols)]
        n = len(self.rod_objs)
        if n <= len(hcols):
            self.slots = [(c[0], c[1]) for c in hcols]
        else:
            self.slots = [(c[0], c[1] + d) for d in (-0.022, 0.022) for c in hcols]
            self.slots += [(c[0], c[1]) for c in hcols]
        self.grip = 0.0
        self.push_to = None
        self.gen = self._plan()

    def get_action(self, state):
        self.state = state
        try:
            a = next(self.gen)
        except StopIteration:
            a = self._act()
        return np.asarray(a, dtype=np.float32)

    # ------------------------------------------------------------------ motion
    def _act(self, dq=None, db=None):
        a = np.zeros(11)
        if db is not None: a[0:3] = np.clip(db, -0.1, 0.1)
        if dq is not None: a[3:10] = np.clip(dq, -0.1, 0.1)
        a[10] = self.grip
        return a

    def goto(self, qt=None, bt=None, steps=60, tol=0.01):
        for t in range(steps):
            dq = db = None; done = True
            if qt is not None:
                e = qt - self.q(); e[[0, 2, 4, 6]] = (e[[0, 2, 4, 6]] + np.pi) % (2 * np.pi) - np.pi
                dq = e; done &= np.abs(e).max() < tol
            if bt is not None:
                e = bt - self.base(); e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi
                db = e; done &= np.abs(e).max() < tol
            if done and t > 0:
                return
            yield self._act(dq, db)

    def wait(self, n):
        for _ in range(n):
            yield self._act()

    def move_ee(self, pw, Rw, bt=None, steps=80, tol=0.005):
        b = self.base() if bt is None else np.asarray(bt, float)
        qt, ok = ik_best(self.q(), world_to_arm(b, pw), Rz(b[2]).T @ Rw, tool=TOOL, n_random=10)
        yield from self.goto(qt=qt, bt=b if bt is not None else None, steps=steps, tol=tol)

    # ------------------------------------------------------------------ plan
    def _plan(self):
        done = set()
        si = 0
        while si < len(self.slots):
            remaining = [n for n in self.rod_objs if n not in done]
            if not remaining:
                break
            b = self.base()
            name = min(remaining, key=lambda n: np.linalg.norm(self.rod(n)[:2] - b[:2]))
            if getattr(self, 'rod_slot', None):
                name = sorted(remaining)[0]
            ok = yield from self._pick(name)
            if not ok:
                done.add(name)
                continue
            sl = self.rod_slot[name] if getattr(self, 'rod_slot', None) else self.slots[si]
            yield from self._insert(name, sl[1], zc=sl[2] if len(sl) > 2 else 0.075)
            done.add(name)
            si += 1
        for rep in range(2):
            for name in sorted(self.rod_objs):
                r = self.rod(name)
                if 1.6 < r[0] < 1.87 and r[2] < 0.1:
                    saved = self.push_to
                    self.push_to = 1.90
                    yield from self._push(name, r[1])
                    self.push_to = saved
        while True:
            yield self._act()

    def _pick(self, name, off=0.10):
        for attempt in range(2):
            r = self.rod(name)
            R = quat_to_R(r[3:7]); u = R[:, 1].copy(); u[2] = 0; u /= np.linalg.norm(u)
            b = self.base()
            # grasp the end nearer to the robot (rod center ahead of grasp)
            ends = [r[:3] - off * u, r[:3] + off * u]
            k = int(np.argmin([np.linalg.norm(p[:2] - b[:2]) for p in ends]))
            gp = ends[k]
            self.rod_sign = 1.0 if k == 0 else -1.0
            gyaw = np.arctan2(u[0], -u[1])  # Rdown(yaw) has ee y = (sin, -cos) = u
            d = gp[:2] - b[:2]; th = np.arctan2(d[1], d[0])
            bt = np.array([gp[0] - 0.55 * np.cos(th), gp[1] - 0.55 * np.sin(th), th])
            # choose yaw (+/- pi) closest to the base heading for a natural wrist
            cands = [gyaw, gyaw + np.pi]
            gyaw = min(cands, key=lambda y: abs((y - th - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi))
            Rw = Rdown(gyaw)
            self.grip = 0.0
            yield from self.move_ee(np.array([gp[0], gp[1], 0.12]), Rw, bt=bt, steps=120)
            yield from self.move_ee(np.array([gp[0], gp[1], 0.02]), Rw, steps=50, tol=0.004)
            self.grip = 1.0
            yield from self.wait(6)
            yield from self.move_ee(np.array([gp[0], gp[1], 0.14]), Rw, steps=40, tol=0.02)
            if self.rod(name)[2] > 0.06:
                return True
            self.grip = 0.0
            yield from self.wait(4)
        return False

    def _rod_goal(self, name, p):
        G = self.grasp_tf(name)
        Rr = quat_to_R(self.rod(name)[3:7])
        s = self.rod_sign
        ny = np.array([s, 0, 0]); nz = np.array([0, 0, 1.0]); nx = np.cross(ny, nz)
        Rt = np.column_stack([nx, ny, nz])
        Re = Rt @ G[1].T
        pe = p - Re @ G[0]
        return pe, Re

    def _insert(self, name, cy, zc=0.075, xs=1.5):
        # rod center must end up ahead (+x) of the grasp point
        pe, Re = self._rod_goal(name, np.array([xs, cy, zc + 0.05]))
        bt = np.array([pe[0] - MX - 0.55, cy, 0.0])
        yield from self.move_ee(pe, Re, bt=bt, steps=150, tol=0.01)
        for it in range(2):
            pe, Re = self._rod_goal(name, np.array([xs, cy, zc]))
            yield from self.move_ee(pe, Re, steps=40, tol=0.004)
        qhold = self.q().copy(); b0 = self.base().copy()
        target = 1.90 - self.rod(name)[0]
        prev = b0[0]; stall = 0
        for k in range(60):
            e = qhold - self.q(); e[[0, 2, 4, 6]] = (e[[0, 2, 4, 6]] + np.pi) % (2 * np.pi) - np.pi
            db = np.array([min(0.04, b0[0] + target - self.base()[0]), b0[1] - self.base()[1], -self.base()[2]])
            yield self._act(e, db)
            bx = self.base()[0]
            if self.rod(name)[0] >= 1.89:
                break
            stall = stall + 1 if bx - prev < 0.002 else 0
            prev = bx
            if stall > 6:
                break
        self.grip = 0.0
        yield from self.wait(6)
        pe, Re = self.ee_world()
        yield from self.move_ee(pe + np.array([0, 0, 0.06]), Re, steps=20, tol=0.02)
        if self.push_to is None:
            yield from self.goto(bt=self.base() - np.array([0.25, 0, 0]), steps=15, tol=0.03)
            return
        yield from self.goto(bt=self.base() - np.array([0.12, 0, 0]), steps=10, tol=0.03)
        yield from self._push(name, cy)

    def _push(self, name, cy):
        r = self.rod(name)
        ar = np.radians(45); zz = np.array([np.cos(ar), 0, -np.sin(ar)]); xx = np.array([0, 1., 0])
        Rp = np.column_stack([xx, np.cross(zz, xx), zz])
        self.grip = 1.0
        xr = r[0] - 0.15
        zp = max(r[2] - 0.005, 0.03)
        yield from self.move_ee(np.array([xr - 0.08, cy, zp + 0.05]), Rp,
                                bt=np.array([xr - 0.08 - MX - 0.6, cy, 0.0]), steps=120, tol=0.01)
        yield from self.move_ee(np.array([xr - 0.04, cy, zp]), Rp, steps=40, tol=0.004)
        qhold = self.q().copy(); b0 = self.base().copy()
        prev = b0[0]; stall = 0
        for k in range(80):
            e = qhold - self.q(); e[[0, 2, 4, 6]] = (e[[0, 2, 4, 6]] + np.pi) % (2 * np.pi) - np.pi
            db = np.array([0.03, b0[1] - self.base()[1], -self.base()[2]])
            yield self._act(e, db)
            if self.rod(name)[0] >= self.push_to:
                break
            bx = self.base()[0]
            stall = stall + 1 if bx - prev < 0.002 else 0
            prev = bx
            if stall > 6:
                break
        pe, Re = self.ee_world()
        yield from self.move_ee(pe + np.array([-0.03, 0, 0.08]), Re, steps=20, tol=0.02)
        yield from self.goto(bt=self.base() - np.array([0.2, 0, 0]), steps=15, tol=0.03)
        self.grip = 0.0
