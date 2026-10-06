"""Scripted grasp-and-toss policy for Tossing3D (variable number of cubes).

Strategy per cube:
  1. drive base so the cube is in front of the arm, aligned with a cube face;
  2. tilted top grasp (deep grasp so the weak gripper holds through the swing);
  3. lift, move arm to an overhead wind-up pose (leaning back);
  4. drive base to a throw pose facing the bin;
  5. overhand swing (joint-velocity targets on j2/j4/j6), release; the
     swing speed is picked from a calibration table (distance -> speed).
"""
import numpy as np
import mujoco

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from kinder.envs.dynamic3d.task_families import Tossing3DEnv
        env = Tossing3DEnv(num_objects=1, scene_bg=False)
        _MODEL = env._object_centric_env._robot_env.sim.model.mj_model
    return _MODEL


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Kin:
    def __init__(self):
        self.m = _get_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.base_adr = [m.jnt_qposadr[jid(n)] for n in ("robot_joint_x", "robot_joint_y", "robot_joint_th")]
        self.arm_adr = [m.jnt_qposadr[jid(f"robot_joint_{i}")] for i in range(1, 8)]
        self.arm_dof = [m.jnt_dofadr[jid(f"robot_joint_{i}")] for i in range(1, 8)]
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def fk(self, base, q):
        d = self.d
        for a, v in zip(self.base_adr, base):
            d.qpos[a] = v
        for a, v in zip(self.arm_adr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, base, q0, pos, rot, iters=300, tol=1e-4, rot_w=0.3, qnom=None, null_w=0.1):
        q = np.array(q0, dtype=float)
        qnom = q.copy() if qnom is None else np.array(qnom, dtype=float)
        lo = np.array([-1e9, -2.2, -1e9, -2.5, -1e9, -2.05, -1e9])
        hi = -lo
        for _ in range(iters):
            p, R = self.fk(base, q)
            ep = pos - p
            er = 0.5 * (np.cross(R[:, 0], rot[:, 0]) + np.cross(R[:, 1], rot[:, 1]) + np.cross(R[:, 2], rot[:, 2]))
            e = np.concatenate([ep, rot_w * er])
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, self.arm_dof], rot_w * self.jacr[:, self.arm_dof]])
            Jp = J.T @ _spd_inv(J @ J.T + 1e-3 * np.eye(6))
            dq = Jp @ e + null_w * (np.eye(7) - Jp @ J) @ (qnom - q)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, lo, hi)
        p, R = self.fk(base, q)
        return q, float(np.linalg.norm(pos - p))


def _spd_inv(A):
    # small SPD inverse via MuJoCo Cholesky (avoids BLAS threading overhead)
    n = A.shape[0]
    L = np.array(A, dtype=np.float64).copy()
    mujoco.mju_cholFactor(L, 0.0)
    inv = np.zeros((n, n))
    e = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        e[:] = 0.0
        e[i] = 1.0
        mujoco.mju_cholSolve(x, L, e)
        inv[:, i] = x
    return inv


def quat_to_mat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def tilt_rot(hd, al):
    z = np.array([np.cos(hd) * np.sin(al), np.sin(hd) * np.sin(al), -np.cos(al)])
    x = np.array([-np.sin(hd), np.cos(hd), 0.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


RETRACT = np.array([0, -0.349, 3.142, -2.548, 0, -0.873, 1.571])
QNOM = np.array([0, 1.7, np.pi, -1.7, 0, 1.2, np.pi / 2])
WINDUP = np.array([0, -1.0, np.pi, 0.6, 0, 0.6, np.pi / 2])
THROW_DIR = np.array([0, 1, 0, -1, 0, -1, 0], dtype=float)
THROW_N = 5
THROW_R = 3
# Calibration: swing speed V -> forward landing distance (base frame) at z=0.1
CAL_V = np.array([3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.5])
CAL_D = np.array([1.551, 1.767, 1.976, 2.175, 2.362, 2.532, 2.681, 2.808, 2.915, 3.031])
GRASP_TILT = 0.9
GRASP_R = 0.65  # horizontal distance arm-mount -> cube
ARM_OFF = 0.12
THROW_D_PREF = 2.4
THROW_X_MAX = 0.9
AIM_BIAS = -0.01
SPEED_DESCEND = 0.8
SPEED_LIFT = 0.8
N_CLOSE = 6
N_POST = 3


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.kin = Kin()
        self.state = None
        self.gen = None
        self.fails = {}

    # ------------------------------------------------------------ state io
    def _robot(self):
        s = self.state
        r = s.get_object_from_name("robot")
        base = np.array([s.get(r, f) for f in ("pos_base_x", "pos_base_y", "pos_base_rot")])
        q = np.array([s.get(r, f"pos_arm_joint{i}") for i in range(1, 8)])
        return base, q

    def _obj(self, name):
        s = self.state
        o = s.get_object_from_name(name)
        return np.array([s.get(o, f) for f in ("x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz")])

    def _cubes(self):
        return sorted(n for n in self.state.get_object_names() if n.startswith("cube_"))

    def _bin(self):
        return self._obj("bin_0")

    def _in_bin(self, name):
        c = self._obj(name)
        b = self._bin()
        R = quat_to_mat(*b[3:7])
        rel = R.T @ (c[:3] - b[:3])
        return abs(rel[0]) <= 0.11 and abs(rel[1]) <= 0.11 and -0.01 <= rel[2] <= 0.11

    # ------------------------------------------------------------ actions
    def _act(self, bt=None, qt=None, g=0.0, vel=None, dq=None, bstep=0.1):
        base, q = self._robot()
        a = np.zeros(18)
        if qt is None:
            qt = q
        e = qt - q
        a[3:10] = np.clip(e, -0.1, 0.1)
        a[11:18] = np.clip(e / 0.3, -3, 3)
        if vel is not None:
            a[11:18] = vel
            a[3:10] = dq
        if bt is not None:
            db = bt - base
            db[2] = wrap(db[2])
            a[0:3] = np.clip(db, -bstep, bstep)
        a[10] = g
        return a.astype(np.float32)

    def _go(self, bt, qt, g, max_n, tol=0.01, min_n=1, bstep=0.1, btol=0.005):
        for i in range(max_n):
            base, q = self._robot()
            db = bt - base
            db[2] = wrap(db[2])
            if i >= min_n and np.abs(qt - q).max() < tol and np.abs(db).max() < btol:
                return
            yield self._act(bt, qt, g, bstep=bstep)

    def _slow(self, bt, qt, g, speed=1.0, extra=6, bstep=0.1):
        base, q0 = self._robot()
        N = max(1, int(np.ceil(np.abs(qt - q0).max() / (speed * 0.1))))
        for i in range(1, N + 1):
            qi = q0 + (qt - q0) * i / N
            base, q = self._robot()
            e = qi - q
            a = self._act(bt, None, g, bstep=bstep)
            a[3:10] = np.clip(e, -0.1, 0.1)
            a[11:18] = (qt - q0) / N / 0.1 + e / 0.3
            yield a
        yield from self._go(bt, qt, g, extra, tol=0.005, bstep=bstep)

    # ------------------------------------------------------------ planning
    def _others(self, name):
        return [self._obj(n)[:2] for n in self._cubes() if n != name and not self._in_bin(n)]

    def _grasp_plan(self, name):
        c = self._obj(name)
        base, q = self._robot()
        Rc = quat_to_mat(*c[3:7])
        up = np.argmax(np.abs(Rc[2, :]))
        yaws = []
        for j in range(3):
            if j == up:
                continue
            ax = Rc[:, j]
            yaw = np.arctan2(ax[1], ax[0])
            yaws += [wrap(yaw), wrap(yaw + np.pi)]
        others = self._others(name)
        L = GRASP_R + ARM_OFF
        best, bc = None, 1e9
        for hd in yaws:
            bt = np.array([c[0] - L * np.cos(hd), c[1] - L * np.sin(hd), hd])
            cost = abs(hd)
            # after failed attempts, rotate preference among faces
            k = self.fails.get(name, 0)
            if k:
                cost = abs(wrap(hd - k * np.pi / 2))
            if bt[0] > 1.0 or abs(bt[1]) > 2.65 or bt[0] < -1.6:
                cost += 100
            for o in others:
                if np.hypot(*(o - bt[:2])) < 0.35:
                    cost += 10
            if cost < bc:
                best, bc = (bt, hd), cost
        bt, hd = best
        return c, bt, hd

    def _throw_plan(self, name=None):
        b = self._bin()
        bx, by = b[0], b[1]
        others = self._others(name) if name is not None else []
        best, bc = None, 1e9
        for x in np.arange(-0.6, 0.91, 0.05):
            for dy in (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75):
                y = by + dy
                if abs(y) > 2.6:
                    continue
                D = np.hypot(bx - x, by - y)
                if D < CAL_D[0] + 0.1 or D > CAL_D[-1] - 0.1:
                    continue
                cost = abs(D - THROW_D_PREF) + 0.3 * abs(dy)
                for o in others:
                    if np.hypot(o[0] - x, o[1] - y) < 0.45:
                        cost += 10
                if cost < bc:
                    bc, best = cost, (x, y)
        if best is None:
            best = (min(bx - THROW_D_PREF, 0.9), float(np.clip(by, -2.6, 2.6)))
        x, y = best
        D = np.hypot(bx - x, by - y) - AIM_BIAS
        hd = np.arctan2(by - y, bx - x)
        V = float(np.interp(D, CAL_D, CAL_V))
        return np.array([x, y, hd]), V

    def _cube_dist_to_pinch(self, name):
        base, q = self._robot()
        p, _ = self.kin.fk(base, q)
        return float(np.linalg.norm(self._obj(name)[:3] - p))

    def _pick_cube(self):
        best, bd = None, 1e9
        base, _ = self._robot()
        for n in self._cubes():
            if self._in_bin(n):
                continue
            c = self._obj(n)
            if c[0] > 1.3:  # beyond the barrier: unreachable
                continue
            d = np.hypot(c[0] - base[0], c[1] - base[1])
            if d < bd:
                best, bd = n, d
        return best

    def _plan(self):
        while True:
            name = self._pick_cube()
            if name is None:
                while True:
                    yield self._act(None, None, 0.0)
            yield from self._handle_cube(name)

    def _handle_cube(self, name):
        kin = self.kin
        c, bt, hd = self._grasp_plan(name)
        Rg = tilt_rot(hd, GRASP_TILT)
        qa, _ = kin.ik(bt, QNOM, c[:3] - 0.12 * Rg[:, 2] + np.array([0, 0, 0.03]), Rg, qnom=QNOM)
        base, q = self._robot()
        far = np.hypot(*(bt[:2] - base[:2])) > 0.4
        self.phase = 'nav'
        if far:
            # keep the arm tucked while travelling, unfold near the target
            yield from self._go(bt, RETRACT, 0.0, 40, tol=0.05, btol=0.3)
        yield from self._go(bt, qa, 0.0, 60, tol=0.02)
        c = self._obj(name)
        qb, _ = kin.ik(bt, qa, c[:3] + 0.01 * Rg[:, 2], Rg, qnom=qa)
        self.phase = 'descend'
        yield from self._slow(bt, qb, 0.0, SPEED_DESCEND, extra=4)
        self.phase = 'close'
        for _ in range(N_CLOSE):
            yield self._act(bt, qb, 1.0)
        qc, _ = kin.ik(bt, qb, c[:3] + np.array([0, 0, 0.2]), Rg, qnom=qb)
        self.phase = 'lift'
        yield from self._slow(bt, qc, 1.0, SPEED_LIFT, extra=1)
        if self._cube_dist_to_pinch(name) > 0.06:
            # failed grasp: open, back off, retry with another face
            self.fails[name] = self.fails.get(name, 0) + 1
            for _ in range(5):
                yield self._act(bt, qc, 0.0)
            return
        tb, V = self._throw_plan(name)
        self.phase = 'windup'
        yield from self._slow(tb, WINDUP, 1.0, 1.0, extra=2, bstep=0.05)
        self.phase = 'navthrow'
        yield from self._go(tb, WINDUP, 1.0, 80, tol=0.01, bstep=0.05)
        tb, V = self._throw_plan(name)
        self.phase = 'navfine'
        yield from self._go(tb, WINDUP, 1.0, 10, tol=0.005, bstep=0.02)
        if self._cube_dist_to_pinch(name) > 0.06:
            return
        self.phase = 'throw'
        for i in range(THROW_N):
            yield self._act(tb, None, 1.0 if i < THROW_R else 0.0, vel=V * THROW_DIR, dq=0.1 * np.sign(THROW_DIR))
        self.phase = 'post'
        for i in range(N_POST):
            base, q = self._robot()
            yield self._act(tb, q, 0.0)

    # ------------------------------------------------------------ API
    def reset(self, state, info):
        self.state = state
        self.fails = {}
        self.gen = self._plan()

    def get_action(self, state):
        self.state = state
        try:
            return next(self.gen)
        except StopIteration:
            self.gen = self._plan()
            return next(self.gen)
