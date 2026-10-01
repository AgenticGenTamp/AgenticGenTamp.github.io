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
        slots = []
        for idx in sorted(cols):
            t = types[idx] if idx < len(types) else 'M'
            if t == 'L':
                slots.append((0, cols[idx][0], cols[idx][1], L_BOARD))
        for idx in sorted(cols):
            t = types[idx] if idx < len(types) else 'M'
            if t == 'H':
                slots.append((1, cols[idx][0], cols[idx][1], 0.0))
        self.slots = [(s[1], s[2], s[3]) for s in slots]
        self.grip = 0.0
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
        placed = set()
        slot_i = 0
        while slot_i < len(self.slots):
            remaining = [n for n in self.rod_objs if n not in placed]
            if not remaining:
                break
            b = self.base()
            name = min(remaining, key=lambda n: np.linalg.norm(self.rod(n)[:2] - b[:2]))
            sx, sy, sz = self.slots[slot_i]
            ok = yield from self._pick(name)
            if not ok:
                placed.add(name)
                continue
            yield from self._insert_vertical(name, sy, sz)
            placed.add(name)
            slot_i += 1
        while True:
            yield self._act()

    def _pick(self, name):
        for attempt in range(2):
            r = self.rod(name); yaw = quat_yaw(r[3:7])
            b = self.base(); d = r[:2] - b[:2]; th = np.arctan2(d[1], d[0])
            bt = np.array([r[0] - 0.55 * np.cos(th), r[1] - 0.55 * np.sin(th), th])
            self.grip = 0.0
            yield from self.goto(bt=bt, steps=80)
            Rw = Rdown(yaw)
            yield from self.move_ee(np.array([r[0], r[1], 0.15]), Rw, steps=80)
            yield from self.move_ee(np.array([r[0], r[1], 0.02]), Rw, steps=60)
            self.grip = 1.0
            yield from self.wait(8)
            yield from self.move_ee(np.array([r[0], r[1], 0.25]), Rw, steps=60)
            if self.rod(name)[2] > 0.1:
                return True
            self.grip = 0.0
            yield from self.wait(5)
        return False

    def _insert_vertical(self, name, cy, zb):
        REACH = 0.65
        zc = zb + 0.15 + 0.015
        gx0 = 1.6
        Rv = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        yield from self.move_ee(np.array([gx0, cy, max(zc, 0.25) + 0.05]), Rv,
                                bt=np.array([gx0 - MX - REACH, cy, 0.0]), steps=200)
        for it in range(3):
            G = self.grasp_tf(name)
            Rrod = vertical_R(quat_to_R(self.rod(name)[3:7]))
            Re = Rrod @ G[1].T
            pe = np.array([gx0, cy, zc]) - Re @ G[0]
            yield from self.move_ee(pe, Re, steps=80, tol=0.003)
        qhold = self.q().copy(); b0 = self.base().copy()
        # insert: rod center to x ~ FRONT_X + 0.025
        r = self.rod(name)
        dx = (FRONT_X + 0.025) - r[0]
        yield from self.goto(qt=qhold, bt=b0 + np.array([dx, 0, 0]), steps=60, tol=0.003)
        self.grip = 0.0
        yield from self.wait(12)
        for k in range(10):
            yield from self.goto(bt=self.base() - np.array([0.01 * (k + 1), 0, 0]), steps=4)
        yield from self.goto(bt=self.base() - np.array([0.2, 0, 0]), steps=30)
        # push the rod bottom deeper with closed fingertips
        r = self.rod(name)
        if r[0] > FRONT_X - 0.02 and abs(r[1] - cy) < 0.05:
            self.grip = 1.0
            Rp = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            pz = max(zb + 0.04, 0.2)
            yield from self.move_ee(np.array([r[0] - 0.25, cy, pz + 0.05]), Rp,
                                    bt=np.array([r[0] - 0.25 - MX - REACH, cy, 0.0]), steps=150)
            yield from self.move_ee(np.array([r[0] - 0.25, cy, pz]), Rp, steps=60, tol=0.003)
            qhold = self.q().copy(); b0 = self.base().copy()
            for k in range(1, 40):
                yield from self.goto(qt=qhold, bt=b0 + np.array([0.01 * k, 0, 0]), steps=6, tol=0.003)
                if self.base()[0] < b0[0] + 0.01 * k - 0.02:
                    break
            yield from self.goto(bt=self.base() - np.array([0.3, 0, 0]), steps=40)
            self.grip = 0.0
