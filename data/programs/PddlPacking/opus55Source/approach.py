"""PR2 packed approach: greedy grasp/place macros verified in an internal sim."""
"""PR2 packed approach: greedy grasp/place macros verified in an internal sim."""
import numpy as np
import pybullet as p
import scipy.linalg as _sl
from robocode.environments.pr2_tamp_env import PR2PackedEnv
import robocode.environments.ss_pybullet  # noqa (path)

Q0 = (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)

def qmul(a, b):
    return p.multiplyTransforms((0, 0, 0), a, (0, 0, 0), b)[1]

def yaw_quat(y):
    return (0.0, 0.0, np.sin(y / 2), np.cos(y / 2))

def top_quat(yaw):
    return qmul(yaw_quat(yaw), Q0)

def rotvec_err(qt, qc):
    # rotation taking qc to qt, as rotation vector (world frame)
    qi = (-qc[0], -qc[1], -qc[2], qc[3])
    d = qmul(qt, qi)
    d = np.array(d)
    if d[3] < 0:
        d = -d
    s = np.linalg.norm(d[:3])
    if s < 1e-9:
        return np.zeros(3)
    ang = 2 * np.arctan2(s, d[3])
    return d[:3] / s * ang

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Sim:
    def __init__(self, k):
        self.env = PR2PackedEnv(num_blocks=k)
        self.env.reset(seed=0)
        self.cid = self.env.client_id
        self.robot = self.env.robot
        self.joints = list(self.env.base_joints) + list(self.env.arm_joints)
        self.tool = self.env.tool_link
        lo = np.array([-5, -5, -np.pi] + list(self.env._arm_limits[0]) if hasattr(self.env, '_arm_limits') else None)
        al = self.env._arm_limits
        self.lo = np.array([-5.0, -5.0, -np.inf] + [float(x) for x in al[0]])
        self.hi = np.array([5.0, 5.0, np.inf] + [float(x) for x in al[1]])
        self.circ = np.array([True, False, False] + [False]*7)
        self.circ[2] = True
        self.circ[3 + 4] = True
        self.circ[3 + 6] = True
        cid, r = self.cid, self.robot
        self.mov = [j for j in range(p.getNumJoints(r, physicsClientId=cid)) if p.getJointInfo(r, j, physicsClientId=cid)[2] != p.JOINT_FIXED]
        self.jidx = [self.mov.index(j) for j in self.joints]
        self.lo[self.circ] = -np.inf
        self.hi[self.circ] = np.inf
        st = self.env.get_state()
        self.ff = FastFK(self)
        self.env.set_state(st)

    def get_state(self):
        return self.env.get_state()

    def set_state(self, s):
        self.env.set_state(s)

    def set_conf(self, q):
        for j, v in zip(self.joints, q):
            p.resetJointState(self.robot, j, float(v), physicsClientId=self.cid)

    def fk(self, q):
        self.set_conf(q)
        ls = p.getLinkState(self.robot, self.tool, computeForwardKinematics=1, physicsClientId=self.cid)
        return np.array(ls[4]), ls[5]

    def jac(self, q):
        self.set_conf(q)
        js = [x[0] for x in p.getJointStates(self.robot, self.mov, physicsClientId=self.cid)]
        ls = p.getLinkState(self.robot, self.tool, computeForwardKinematics=1, physicsClientId=self.cid)
        z = [0.0] * len(js)
        lt, an = p.calculateJacobian(self.robot, self.tool, (0, 0, 0), js, z, z, physicsClientId=self.cid)
        J = np.vstack([np.array(lt)[:, self.jidx], np.array(an)[:, self.jidx]])
        return np.array(ls[4]), ls[5], J

    def ik(self, pos, quat, seed, dofs, iters=100, tol_p=1e-3, tol_r=1e-2, weights=None):
        if not hasattr(self, 'ff'):
            self.ff = FastFK(self)
        q = np.array(seed, dtype=float)
        dofs = list(dofs)
        W = np.ones(len(dofs)) if weights is None else np.asarray(weights, float)
        pos = np.asarray(pos)
        Rt = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        prev_en = 1e9
        I6 = np.eye(6)
        for it in range(iters):
            F, Jf = self.ff.jac(q)
            e = np.concatenate([pos - F[:3, 3], rot_err(Rt, F[:3, :3])])
            ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if ep < tol_p and er < tol_r:
                return q
            if it % 20 == 19:
                en = ep + 0.1 * er
                if en > 0.7 * prev_en:
                    return None
                prev_en = en
            J = Jf[:, dofs]
            Jw = J / W
            A = Jw @ J.T + 0.03 ** 2 * I6
            dq = (Jw.T @ _sl.cho_solve(_sl.cho_factor(A, check_finite=False), e, check_finite=False)) / W
            m = np.max(np.abs(dq))
            if m > 0.4:
                dq *= 0.4 / m
            q[dofs] += dq
            q = np.clip(q, self.lo, self.hi)
            for d in (2, 7, 9):
                q[d] = wrap(q[d])
        return None

def quat_yaw(q):
    # yaw of a quaternion's x-axis projected... use euler
    return p.getEulerFromQuaternion(q)[2]


def obs_to_flat(state):
    names = ['robot', 'table', 'plate']
    blocks = sorted([o.name for o in state if o.name.startswith('block')], key=lambda n: int(n[5:]))
    by = {o.name: o for o in state}
    rows = [np.asarray(state[by[n]], dtype=np.float64) for n in names + blocks]
    return np.concatenate(rows), len(blocks)


def _T(pos, quat):
    M = np.eye(4)
    M[:3, :3] = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    M[:3, 3] = pos
    return M


def _rot(axis, a):
    x, y, z = axis
    c, s = np.cos(a), np.sin(a)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def mat_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S; x = (R[2, 1] - R[1, 2]) / S; y = (R[0, 2] - R[2, 0]) / S; z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S; x = 0.25 * S; y = (R[0, 1] + R[1, 0]) / S; z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S; x = (R[0, 1] + R[1, 0]) / S; y = 0.25 * S; z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S; x = (R[0, 2] + R[2, 0]) / S; y = (R[1, 2] + R[2, 1]) / S; z = 0.25 * S
    return np.array([x, y, z, w])


def rot_err(Rt, Rc):
    """rotation vector of Rt * Rc^T"""
    R = Rt @ Rc.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    if ang > np.pi - 1e-4:
        # near pi: use eigen approach
        w, V = np.linalg.eigh((R + R.T) / 2)
        ax = V[:, np.argmax(w)]
        return ax * ang
    return v * (ang / (2 * np.sin(ang)))


class FastFK:
    """Numpy FK/Jacobian for base(x,y,theta) + 7 arm joints -> tool frame."""

    def __init__(self, sim):
        cid, r = sim.cid, sim.robot
        q = np.zeros(10)
        sim.set_conf(q)
        arm = list(sim.env.arm_joints)
        def lf(link):
            ls = p.getLinkState(r, link, computeForwardKinematics=1, physicsClientId=cid)
            return _T(ls[4], ls[5])
        F_base = np.eye(4)  # base at zero conf assumed identity in xy-yaw
        # check: link of theta joint frame
        Fth = lf(2)
        self.T_base_off = Fth  # world pose of theta-link at zero base conf
        prev = Fth
        self.origins = []
        self.axes = []
        for j in arm:
            F = lf(j)
            self.origins.append(np.linalg.inv(prev) @ F)
            ax = np.array(p.getJointInfo(r, j, physicsClientId=cid)[13], float)
            self.axes.append(ax / np.linalg.norm(ax))
            prev = F
        self.T_tool = np.linalg.inv(prev) @ lf(sim.tool)
        ax = np.array(p.getJointInfo(r, 2, physicsClientId=cid)[13], float)
        self.th_axis = ax

    def fk_all(self, q):
        x, y, th = q[0], q[1], q[2]
        B = np.eye(4)
        B[:3, :3] = _rot((0, 0, 1), th)
        B[0, 3] = x; B[1, 3] = y
        F = B @ self.T_base_off
        frames = []
        for i in range(7):
            F = F @ self.origins[i]
            frames.append(F)
            F2 = F.copy()
            F2[:3, :3] = F[:3, :3] @ _rot(self.axes[i], q[3 + i])
            F = F2
        F = F @ self.T_tool
        return F, frames

    def fk(self, q):
        return self.fk_all(q)[0]

    def jac(self, q):
        F, frames = self.fk_all(q)
        pt = F[:3, 3]
        J = np.zeros((6, 10))
        J[0, 0] = 1.0
        J[1, 1] = 1.0
        J[0, 2] = -(pt[1] - q[1])
        J[1, 2] = pt[0] - q[0]
        J[5, 2] = 1.0
        A = np.empty((7, 3))
        D = np.empty((7, 3))
        for i in range(7):
            fr = frames[i]
            A[i] = fr[:3, :3] @ self.axes[i]
            D[i] = pt - fr[:3, 3]
        C = np.empty((7, 3))
        C[:, 0] = A[:, 1] * D[:, 2] - A[:, 2] * D[:, 1]
        C[:, 1] = A[:, 2] * D[:, 0] - A[:, 0] * D[:, 2]
        C[:, 2] = A[:, 0] * D[:, 1] - A[:, 1] * D[:, 0]
        J[:3, 3:] = C.T
        J[3:, 3:] = A.T
        return F, J


STEP = 0.2


def conf_delta(qa, qb):
    d = np.asarray(qb, float) - np.asarray(qa, float)
    for i in (2, 7, 9):
        d[i] = wrap(d[i])
    return d


def nsteps(qa, qb):
    m = np.max(np.abs(conf_delta(qa, qb)))
    return max(1, int(np.ceil(m / STEP - 1e-6)))




CARRY_ARM = np.array([0.67717022, -0.34313199, 1.2, -1.46688402, 1.24223232, -1.95442832, 2.22254133])
SIDE_YAWS = (0.0, np.pi, np.pi / 2, -np.pi / 2)
RING = [(-0.85, -1.15), (-0.85, 0.0), (-0.85, 1.15), (0.0, 1.15), (0.85, 1.15), (0.85, 0.0),
        (0.85, -1.15), (0.0, -1.15)]


_OCC_B64 = "eNrNWN1t2zAQJs0gzEMBZYAA8iSRB+gQGSEDBKA26Ep87BZl0QX4Fj0QYqU7lUdSki25kBQ+WKDMM7/7/86MrVtn0X9y3/aPwjf9Q3nffUrvXfeovDf9AXhXeDjRHdCMCQ9Spfe2e3QH6ltCNQgZFGIg1OBNGoRagAI/V8DpTqj/ORIyC29aAU/iAQGfe6wTe2MAQpPlKwDO0RAlPhBygZAVoBQIuQI9eXLAg04jpVHBBoU0CDm0lAmeFnggXNuSVSq4goQsCuFN/VclHsfb5QS8MuAnn8jgY4i6GqLuiW9u+eGGF1EDiBaD3SIiCjcZ9NRBJwVqCNSzDO/IJ1dN3ibWwzRwSUSvExpuWgIvjfIKTa5kv+G27DeSvbNd1+ntOya/CREm4yAmnYYw0mCIFk/bOPLbJB7ROE1snCkhd6/QTSj0To+KzS8IrQbs/yp61dnliR23KMqbOJvJDUXsBoE6Ubpb9FMNCjpUehuhBoX0WIi8f7sBYKk7aj2fGBRaKDnlv0bnCDJlMxqnzbMZDVEEQ+zuBhu7IYfHQ+ZSyDP1icXmB3TWpugrLKuf97X8u8Rgd1TsC2x0Cc1RYwV5ovRMzd+u2KhxLRzBa2J4cTaU0NSsgnDjlgyx4zo/UHfnsckpS0lBssqUpcxENicZfheZMXEQNHl998HTeWEkeG4CSlSAfsPmgdWH1BzVbkfmq+WUfYbnF8vHhvVkniEUgVn/yM77WJzjRdjdebCAziHPkPmZ2CsSrvj/UZ4QzMSiU4RqhsybnMxL9DQOMgw3Ejfftk+AgcN+SE11T8Xw1HK2TH66l56vEJLTxX4RmQ/Jldgf4t+8oF0uO0yzF+ru5R+cn5SdDvk5tpzH3ja9dEroOhQb86BkQHEh04Y55vHy80uQ+ZvFkgpLQv4Smnkg20+EZgrj1yDz5pWFkB/Gu6RnFYEZW9zU2Thz4DBlMpKbwBN5PlELYUl/PYrM6w/6E0M6peNm5RKd9ISeh7J4l88Vo1kjH1l5IE84Ox5I5oW3CSUb0bQpMj/6U9CMiYZbzWVyAjSCkrLIq3NFROZ9VkB1GBqHdL+fzP8FbzCkBw=="
_OX0, _OY0, _ORES, _ONX, _ONY, _ONT = -1.3, -1.4, 0.05, 53, 57, 16


def _load_occ():
    import base64, zlib
    bits = np.frombuffer(zlib.decompress(base64.b64decode(_OCC_B64)), np.uint8)
    return np.unpackbits(bits)[:_ONT * _ONY * _ONX].reshape(_ONT, _ONY, _ONX).astype(bool)


OCC = _load_occ()


def occ_hit(x, y, yaw):
    """Conservative table-collision test for base poses with CARRY_ARM (vectorised)."""
    x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
    yaw = np.broadcast_to(np.asarray(yaw, float), x.shape)
    fi = (x - _OX0) / _ORES; fj = (y - _OY0) / _ORES
    ft = ((yaw + np.pi) % (2 * np.pi)) / (2 * np.pi / _ONT)
    hit = np.zeros(x.shape, bool)
    inside = (fi >= 0) & (fi < _ONX - 1) & (fj >= 0) & (fj < _ONY - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, _ONX - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, _ONY - 2)
    t0 = np.floor(ft).astype(int) % _ONT
    for dt in (0, 1):
        t = (t0 + dt) % _ONT
        for di in (0, 1):
            for dj in (0, 1):
                hit |= OCC[t, j0 + dj, i0 + di]
    return hit & inside


def _mk_wp():
    pts = []
    ys = np.round(np.arange(-1.15, 1.151, 0.1), 3)
    xs = np.round(np.arange(-0.85, 0.851, 0.1), 3)
    for ax in (0.65, 0.75, 0.85):
        for sx in (-1, 1):
            for y in ys:
                pts.append((sx * ax, y))
    for by in (0.95, 1.05, 1.15):
        for sy in (-1, 1):
            for x in xs:
                pts.append((x, sy * by))
    return np.unique(np.array(pts), axis=0)


WP_XY = _mk_wp()


def _vsteps(A, B):
    d = B[None, :, :] - A[:, None, :]
    for i in (2, 7, 9):
        if i >= d.shape[2]:
            break
        d[:, :, i] = (d[:, :, i] + np.pi) % (2 * np.pi) - np.pi
    m = np.max(np.abs(d), axis=2)
    return np.maximum(1, np.ceil(m / STEP - 1e-6)).astype(int)


def seg_hit(a, b, n, include_end):
    """Grid test along the base path a->b (n steps); intermediate samples (+ end)."""
    m = n + 1 if include_end else n
    if m <= 1:
        return False
    ts = np.arange(1, m) / n
    d = b[:3] - a[:3]
    d[2] = wrap(d[2])
    pts = a[:3][None] + ts[:, None] * d[None]
    return bool(np.any(occ_hit(pts[:, 0], pts[:, 1], pts[:, 2])))


def seg_hit_many(A, B, N, relax_a=False, relax_b=False, rr=0.25):
    """Vectorised seg_hit (intermediate samples only) for rows A[i]->B[i] with N[i] steps."""
    A = np.asarray(A, float)[:, :3]; B = np.asarray(B, float)[:, :3]
    N = np.asarray(N, int)
    n = len(A)
    if n == 0:
        return np.zeros(0, bool)
    S = max(1, int(N.max()) - 1)
    k = np.arange(1, S + 1)[None, :]
    valid = k < N[:, None]
    ts = np.where(valid, k / np.maximum(N[:, None], 1), 0.0)
    d = B - A
    d[:, 2] = (d[:, 2] + np.pi) % (2 * np.pi) - np.pi
    pts = A[:, None, :] + ts[:, :, None] * d[:, None, :]
    h = occ_hit(pts[..., 0].ravel(), pts[..., 1].ravel(), pts[..., 2].ravel()).reshape(n, S)
    if relax_a:
        valid = valid & (np.hypot(pts[..., 0] - A[:, None, 0], pts[..., 1] - A[:, None, 1]) > rr)
    if relax_b:
        valid = valid & (np.hypot(pts[..., 0] - B[:, None, 0], pts[..., 1] - B[:, None, 1]) > rr)
    return np.any(h & valid, axis=1)


def side_normal(th0):
    return np.array([-np.cos(th0), -np.sin(th0)])


def base_stations(tx, ty):
    out = []
    for th0 in SIDE_YAWS:
        for dth in (0.0, -0.4, 0.4):
            th = th0 + dth
            c, s_ = np.cos(th), np.sin(th)
            fx, ly = 0.5, 0.19
            bx = tx - (c * fx - s_ * ly)
            by = ty - (s_ * fx + c * ly)
            if th0 == 0.0:
                bx = min(bx, -0.47)
            elif th0 == np.pi:
                bx = max(bx, 0.47)
            elif th0 > 0:
                by = min(by, -0.97)
            else:
                by = max(by, 0.97)
            out.append((np.array([bx, by, wrap(th)]), th0))
    return out


class Planner:
    def __init__(self, sim):
        self.sim = sim
        self.nsim = 0
        self._wpcache = {}

    # ---- low level simulation ----
    def simulate_exact(self, flat, actions):
        s = self.sim
        s.set_state(flat)
        cur = np.array(flat, float)
        term = False
        for a in actions:
            exp = cur[:10] + a[:10]
            o, _, term, _, _ = s.env.step(np.asarray(a, np.float32))
            o = np.array(o, float)
            if np.max(np.abs(conf_delta(o[:10], exp))) > 1e-3:
                return False, o, term
            cur = o
        return True, cur, term

    def _apply(self, q, a):
        t = q + np.asarray(a[:10], np.float32).astype(float)
        t = np.clip(t, self.sim.lo, self.sim.hi)
        for d in (2, 7, 9):
            t[d] = wrap(t[d])
        return t

    def simulate(self, flat, actions):
        """Fast: collision-check intermediate waypoints, exact env.step for the last action."""
        s = self.sim
        cid = s.cid
        k = s.env._num_blocks if hasattr(s.env, '_num_blocks') else len(s.env.blocks)
        s.set_state(flat)
        blocks = list(s.env.blocks)
        held = None
        for i in range(len(blocks)):
            if flat[19 + 20 + 11 * i + 7] > 0.5:
                held = i
        obst = [s.env.table, s.env.plate] + [b for i, b in enumerate(blocks) if i != held]
        G = None
        if held is not None and flat[11] > 0.5:
            G = _T(flat[12:15], flat[15:19])
        q = np.array(flat[:10], float)
        for si_, a in enumerate(actions[:-1]):
            self.nsim += 1
            q = self._apply(q, a)
            s.set_conf(q)
            for o in obst:
                cp = p.getClosestPoints(s.robot, o, 0.0, physicsClientId=cid)
                if cp:
                    self.last_fail = (si_, o, cp[0][3], np.round(q[:3], 2))
                    return False, None, False
            if G is not None:
                B = s.ff.fk(q) @ G
                p.resetBasePositionAndOrientation(blocks[held], B[:3, 3], mat_to_quat(B[:3, :3]), physicsClientId=cid)
                for o in obst:
                    if p.getClosestPoints(blocks[held], o, 0.0, physicsClientId=cid):
                        return False, None, False
        if len(actions) > 1:
            vec = np.array(flat, float)
            vec[:10] = q
            if G is not None:
                B = s.ff.fk(q) @ G
                o_ = 19 + 20 + 11 * held
                vec[o_:o_ + 3] = B[:3, 3]
                vec[o_ + 3:o_ + 7] = mat_to_quat(B[:3, :3])
        else:
            vec = flat
        return self.simulate_exact(vec, actions[-1:])

    def path_actions(self, qa, qb, n, grip_last=0.0):
        d = conf_delta(qa, qb) / n
        acts = []
        for i in range(n):
            a = np.zeros(11)
            a[:10] = d
            if i == n - 1:
                a[10] = grip_last
            acts.append(a)
        return acts

    def seq_actions(self, qc, seq, grip_last, extra=0):
        acts = []
        prev = qc
        for i, w in enumerate(seq):
            n = nsteps(prev, w) + (extra if i == len(seq) - 1 else 0)
            acts += self.path_actions(prev, w, n, grip_last if i == len(seq) - 1 else 0.0)
            prev = w
        return acts

    def seq_cost(self, qc, seq):
        c = 0
        prev = qc
        for w in seq:
            c += nsteps(prev, w)
            prev = w
        return c

    def table_collision(self, q):
        s = self.sim
        s.set_conf(q)
        with s.env.client():
            from pybullet_tools.utils import pairwise_collision
            return pairwise_collision(s.robot, s.env.table) or pairwise_collision(s.robot, s.env.plate)

    def _ik_pushout(self, pos, quat, seed, normal, use_base, wbase=1.5):
        """IK; if robot collides with table, push base outward along normal and retry arm-only."""
        s = self.sim
        dofs = ([0, 1, 2] if use_base else []) + list(range(3, 10))
        w = ([wbase] * 3 if use_base else []) + [1.0] * 7
        q = s.ik(pos, quat, seed, dofs, weights=w)
        if q is None:
            return None
        for attempt in range(6):
            if not self.table_collision(q):
                return q
            if normal is None:
                # outward from table center
                v = np.array([q[0], q[1]])
                if abs(v[0]) / 0.3 > abs(v[1]) / 0.6:
                    normal = np.array([np.sign(v[0]), 0.0])
                else:
                    normal = np.array([0.0, np.sign(v[1])])
            q2 = q.copy()
            q2[:2] += 0.05 * normal
            q = s.ik(pos, quat, q2, list(range(3, 10)))
            if q is None:
                return None
        return None

    def ik_candidates(self, pos, quat, qc, use_stations=True, best_cost=1e9):
        sols = []
        q = self._ik_pushout(pos, quat, qc, None, True)
        if q is not None:
            sols.append(q)
        if use_stations:
            for st, th0 in base_stations(pos[0], pos[1]):
                seed = np.concatenate([st, qc[3:]])
                if nsteps(qc, seed[:3].tolist() + list(qc[3:])) >= best_cost:
                    continue
                q = self._ik_pushout(pos, quat, seed, side_normal(th0), False)
                if q is None:
                    seed2 = np.concatenate([st, CARRY_ARM])
                    q = self._ik_pushout(pos, quat, seed2, side_normal(th0), False)
                if q is not None:
                    sols.append(q)
        return sols

    def path_seqs(self, qc, qt, pre=None, topn=60):
        """Candidate waypoint sequences from qc to qt (sorted by cost)."""
        seqs = [[qt]]
        if pre is not None:
            seqs.append([pre, qt])
        lift = qc.copy()
        lift[3:] = CARRY_ARM
        seqs.append([lift, qt])
        if pre is not None:
            seqs.append([lift, pre, qt])
        costed = [(self.seq_cost(qc, sq), i, sq) for i, sq in enumerate(seqs)]
        tail = [pre, qt] if pre is not None else [qt]
        first = tail[0]
        tail_cost = self.seq_cost(first, tail[1:]) if len(tail) > 1 else 0
        yc, yt = qc[2], qt[2]
        dy = wrap(yt - yc)
        yaws = [yc, wrap(yc + dy / 2), yt]
        P = WP_XY
        W = []
        for yv in yaws:
            B = np.zeros((len(P), 10))
            B[:, :2] = P
            B[:, 2] = yv
            B[:, 3:] = CARRY_ARM
            W.append(B)
        W = np.vstack(W)
        W = W[~occ_hit(W[:, 0], W[:, 1], W[:, 2])]
        c1 = _vsteps(qc[None], W)[0]
        c2 = _vsteps(W, first[None])[:, 0]
        single = c1 + c2 + tail_cost
        nW = len(W)
        bad1 = seg_hit_many(np.repeat(qc[None], nW, 0), W, c1, relax_a=True)
        bad2 = seg_hit_many(W, np.repeat(first[None], nW, 0), c2, relax_b=True)
        okW = ~(bad1 | bad2)
        sc = np.where(okW, single, 10 ** 6)
        idx = np.argsort(sc, kind='stable')[:topn]
        n0 = len(seqs)
        for j, i in enumerate(idx):
            if sc[i] >= 10 ** 6:
                break
            costed.append((int(single[i]), n0 + j, [W[i]] + tail))
        # pairs among the most promising first / second waypoints
        k1 = np.where(bad1, 10 ** 6, c1 + _vsteps(W, qt[None])[:, 0])
        k2 = np.where(bad2, 10 ** 6, c2 + _vsteps(qc[None], W)[0])
        a1 = np.argsort(k1, kind='stable')[:100]
        a2 = np.argsort(k2, kind='stable')[:100]
        a1 = a1[k1[a1] < 10 ** 6]
        a2 = a2[k2[a2] < 10 ** 6]
        if len(a1) and len(a2):
            M = _vsteps(W[a1][:, :3], W[a2][:, :3])
            pair = c1[a1][:, None] + M + c2[a2][None, :] + tail_cost
            pair[a1[:, None] == a2[None, :]] = 10 ** 6
            order = np.argsort(pair, axis=None, kind='stable')[:600]
            i1s, i2s = np.unravel_index(order, pair.shape)
            keep = pair[i1s, i2s] < 10 ** 6
            i1s, i2s = i1s[keep], i2s[keep]
            bad = seg_hit_many(W[a1[i1s]], W[a2[i2s]], M[i1s, i2s])
            n1 = n0 + len(idx)
            got = 0
            for j in np.nonzero(~bad)[0]:
                i1, i2 = i1s[j], i2s[j]
                costed.append((int(pair[i1, i2]), n1 + got, [W[a1[i1]], W[a2[i2]]] + tail))
                got += 1
                if got >= topn:
                    break
        costed.sort(key=lambda c: (c[0], c[1]))
        return costed

    def wp_ok(self, w):
        key = tuple(np.round(w, 3))
        r = self._wpcache.get(key)
        if r is None:
            r = not self.table_collision(w)
            self._wpcache[key] = r
        return r

    def best_path(self, flat, qc, qt, grip_last, check, pre=None, best_cost=1e9, max_tries=40):
        tries = 0
        for cost, _, seq in self.path_seqs(qc, qt, pre):
            if cost >= best_cost:
                break
            for extra in (0, 1):
                if cost + extra >= best_cost:
                    break
                if extra == 0 and not all(self.wp_ok(w) for w in seq[:-1]):
                    break
                acts = self.seq_actions(qc, seq, grip_last, extra)
                ok, fin, term = self.simulate(flat, acts)
                tries += 1
                if ok and check(fin):
                    return acts, fin, term
            if tries > max_tries:
                break
        return None


from scipy.optimize import linprog as _linprog
from scipy.sparse import csc_matrix as _csc
from scipy.optimize._highs._highs_wrapper import _highs_wrapper
from scipy.optimize._highs._highs_constants import (
    CONST_INF as _INF, MODEL_STATUS_OPTIMAL as _OPT, MESSAGE_LEVEL_NONE as _MLN,
    HIGHS_OBJECTIVE_SENSE_MINIMIZE as _MIN)
_LP_OPTS = {'presolve': True, 'sense': _MIN, 'solver': 'simplex', 'highs_debug_level': _MLN,
            'log_to_console': False, 'output_flag': False}


def _fast_lp(c, M, rhs, lb, ub):
    A = _csc(M)
    lhs = np.full(len(rhs), -_INF)
    r = _highs_wrapper(c, A.indptr, A.indices, A.data, lhs, rhs, lb, ub, np.empty(0, np.uint8), dict(_LP_OPTS))
    if r.get('status') != _OPT:
        return None
    return np.asarray(r['x'])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def base_region(q):
    """Half-plane keeping the base on the side of the table it is currently on."""
    x, y = q[0], q[1]
    if abs(x) / 0.46 >= abs(y) / 0.94:
        return ('x', np.sign(x) if x != 0 else -1.0)
    return ('y', np.sign(y))


class TaskGrasp:
    """Block centre c inside the gripper; tool x-axis ~down; yaw near target."""

    def __init__(self, c, yaw, yaw_tol=0.35, tilt_tol=0.25, ox=(0.012, 0.042), oy=0.003, oz=0.02):
        self.c = np.asarray(c, float)
        self.Rt = np.array(p.getMatrixFromQuaternion(top_quat(yaw))).reshape(3, 3)
        self.yaw_tol, self.tilt_tol = yaw_tol, tilt_tol
        self.ox, self.oy, self.oz = ox, oy, oz

    def rows(self, F, J):
        R, pt = F[:3, :3], F[:3, 3]
        v = self.c - pt
        o = R.T @ v
        Ao = R.T @ (-J[:3] + skew(v) @ J[3:])
        r = rot_err(self.Rt, R)
        Ar = -J[3:]
        b = np.concatenate([o, r])
        A = np.vstack([Ao, Ar])
        lo = np.array([self.ox[0], -self.oy, -self.oz, -self.tilt_tol, -self.tilt_tol, -self.yaw_tol])
        hi = np.array([self.ox[1], self.oy, self.oz, self.tilt_tol, self.tilt_tol, self.yaw_tol])
        return A, b, lo, hi


class TaskPlace:
    """Held block (grasp G) centred over (sx,sy) within tol, upright, yaw near target."""

    def __init__(self, G, sx, sy, yaw, tol_xy, yaw_tol, z=(0.79, 0.86), tilt_tol=0.01):
        self.gp = np.asarray(G[0], float)
        self.Rg = np.array(p.getMatrixFromQuaternion(G[1])).reshape(3, 3)
        self.Rt = np.array(p.getMatrixFromQuaternion(yaw_quat(yaw))).reshape(3, 3)
        self.s = np.array([sx, sy])
        self.tol, self.yaw_tol, self.z, self.tilt_tol = tol_xy, yaw_tol, z, tilt_tol

    def rows(self, F, J):
        R, pt = F[:3, :3], F[:3, 3]
        rg = R @ self.gp
        bpos = pt + rg
        Ab = J[:3] - skew(rg) @ J[3:]
        Rb = R @ self.Rg
        r = rot_err(self.Rt, Rb)
        Ar = -J[3:]
        b = np.concatenate([bpos, r])
        A = np.vstack([Ab, Ar])
        t = self.tol
        lo = np.array([self.s[0] - t[0], self.s[1] - t[1], self.z[0], -self.tilt_tol, -self.tilt_tol, -self.yaw_tol])
        hi = np.array([self.s[0] + t[0], self.s[1] + t[1], self.z[1], self.tilt_tol, self.tilt_tol, self.yaw_tol])
        return A, b, lo, hi


def slp_solve(sim, qc, q0, task, region=None, iters=12, shrink=0.8, margin=0.85, stop_steps=1e9):
    """Minimise max_i |q_i - qc_i| subject to task box constraints (linearised iteratively)."""
    ff = sim.ff
    q = np.array(q0, float)
    lo_j, hi_j = sim.lo, sim.hi
    rho = 0.4
    best = None
    for it in range(iters):
        F, J = ff.jac(q)
        A, b, lo, hi = task.rows(F, J)
        mid = (lo + hi) / 2
        half = (hi - lo) / 2
        lo_m, hi_m = mid - half * margin, mid + half * margin
        viol = np.maximum(0, np.maximum(lo - b, b - hi))
        feas = np.all(viol < 1e-9)
        delta = conf_delta(qc, q)
        tval = np.max(np.abs(delta))
        if feas and (best is None or tval < best[1] - 1e-6):
            best = (q.copy(), tval)
        m = A.shape[0]
        nv = 10 + 1 + m
        c = np.zeros(nv)
        c[10] = 1.0
        c[11:] = 100.0
        nr = 20 + 2 * m + (1 if region is not None else 0)
        M = np.zeros((nr, nv))
        I = np.eye(10)
        M[0:10, :10] = I; M[0:10, 10] = -1
        M[10:20, :10] = -I; M[10:20, 10] = -1
        rhs = np.empty(nr)
        rhs[0:10] = -delta
        rhs[10:20] = delta
        Em = np.eye(m)
        M[20:20 + m, :10] = A; M[20:20 + m, 11:] = -Em
        M[20 + m:20 + 2 * m, :10] = -A; M[20 + m:20 + 2 * m, 11:] = -Em
        rhs[20:20 + m] = hi_m - b
        rhs[20 + m:20 + 2 * m] = b - lo_m
        lbv = np.zeros(nv); ubv = np.full(nv, _INF)
        l = np.full(10, -rho); h = np.full(10, rho)
        fin = np.isfinite(lo_j)
        l[fin] = np.maximum(l[fin], (lo_j - q)[fin])
        fin = np.isfinite(hi_j)
        h[fin] = np.minimum(h[fin], (hi_j - q)[fin])
        lbv[:10] = np.minimum(l, 0.0); ubv[:10] = np.maximum(h, 0.0)
        if region is not None:
            ax, sg = region[0], region[1]
            i = 0 if ax == 'x' else 1
            lim = region[2] if len(region) > 2 else (0.47 if ax == 'x' else 0.95)
            M[nr - 1, i] = -sg
            rhs[nr - 1] = sg * q[i] - lim
        res = _fast_lp(c, M, rhs, lbv, ubv)
        if res is None:
            break
        t_lp = res[10]
        lb = int(np.ceil(t_lp / 0.2 - 1e-6))
        slack = np.sum(res[11:])
        if slack < 1e-6:
            if lb >= stop_steps and it >= 1:
                break
            if best is not None and int(np.ceil(best[1] / 0.2 - 1e-6)) <= lb:
                break
        dq = res[:10]
        q = q + dq
        q = np.clip(q, lo_j, hi_j)
        for d in (2, 7, 9):
            q[d] = wrap(q[d])
        rho *= shrink
    F, J = ff.jac(q)
    A, b, lo, hi = task.rows(F, J)
    if np.all(b >= lo - 1e-9) and np.all(b <= hi + 1e-9):
        tval = np.max(np.abs(conf_delta(qc, q)))
        if best is None or tval < best[1]:
            best = (q.copy(), tval)
    return best


def slp_safe(sim, planner, qc, q0, task, region, tries=4, stop_steps=1e9):
    ax, sg = region[0], region[1]
    i = 0 if ax == 'x' else 1
    lim = 0.47 if ax == 'x' else 0.95
    for _ in range(tries):
        r = slp_solve(sim, qc, q0, task, (ax, sg, lim), stop_steps=stop_steps)
        if r is None:
            return None
        if not planner.table_collision(r[0]):
            return r
        lim = max(lim, sg * r[0][i]) + 0.05
        q0 = r[0]
    return None
import time


def block_info(flat, k):
    out = []
    for i in range(k):
        o = 19 + 20 + 11 * i
        out.append(dict(pos=flat[o:o + 3].copy(), quat=tuple(flat[o + 3:o + 7]), held=flat[o + 7] > 0.5))
    return out


def slot_positions(k):
    """Return list of (x, y, tol_x, tol_y, yaw_tol)."""
    if k == 1:
        return [(0.0, 0.0, 0.08, 0.08, 0.8)]
    if k <= 4:
        c = 0.0675
        return [(x, y, 0.02, 0.02, 0.4) for x in (-c, c) for y in (-c, c)]
    c = 0.09
    return [(x, y, 0.004, 0.004, 0.1) for x in (-c, 0.0, c) for y in (-c, 0.0, c)]


def bidx(i):
    return 19 + 20 + 11 * i


class _Ref:
    def __init__(self, pos, quat):
        self.pos, self.quat = pos, quat

    def __getitem__(self, i):
        if i == 3:
            return self.quat
        if isinstance(i, slice):
            return self.pos[i]
        return self.pos[i]


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.sims = {}
        self.plan = []
        self.exps = []

    def _sim(self, k):
        if k not in self.sims:
            self.sims[k] = Sim(k)
        return self.sims[k]

    def reset(self, state, info):
        import time
        self.t0 = time.time()
        flat, k = obs_to_flat(state)
        self.k = k
        self.sim = self._sim(k)
        self.planner = Planner(self.sim)
        self.plan = []
        self.exps = []
        self.slots = slot_positions(k)
        self.fails = 0
        self.slp_deadline = 25.0 + 2.0 * self.k
        self.slp_budget = 2.5
        self.total_budget = 40.0
        self.la_margin = 0
        self._alts, self._alts_keys = [], []
        self.next_place = None
        self.la_n = 4
        self.hard_extra = 8.0
        self.la_budget = 1.5
        self.la_deadline = 25.0
        self.deadline = 1e18

    def elapsed(self):
        import time
        return time.time() - self.t0

    # ------------------------------------------------------------------
    def _occupied_slots(self, blocks):
        occ = set()
        for b in blocks:
            if b['held']:
                continue
            for si, (sx, sy, _, _, _) in enumerate(self.slots):
                if abs(b['pos'][0] - sx) < 0.071 and abs(b['pos'][1] - sy) < 0.071 and b['pos'][2] < 0.79:
                    occ.add(si)
        return occ

    def _block_done(self, b):
        x, y, z = b['pos']
        if b['held'] or z > 0.79:
            return False
        if not (abs(x) < 0.135 and abs(y) < 0.135):
            return False
        yaw = quat_yaw(b['quat'])
        r = abs(np.cos(yaw)) * 0.035 + abs(np.sin(yaw)) * 0.035
        return abs(x) + r <= 0.135 and abs(y) + r <= 0.135

    def grasp_targets(self, b):
        byaw = quat_yaw(b['quat'])
        for m in range(4):
            yaw = wrap(byaw + m * np.pi / 2)
            yield np.asarray(b['pos']) + np.array([0, 0, 0.03]), top_quat(yaw)

    def _targets_solutions(self, qc, tasks, use_stations, best_cost):
        """Yield candidates (cost, key, q) in nondecreasing cost order (station IK lazily)."""
        import heapq
        s, P = self.sim, self.planner
        region = base_region(qc)
        heap = []
        cnt = 0
        if not use_stations:
            for key, task, ref in tasks:
                for q in P.ik_candidates(ref.pos, ref.quat, qc, False):
                    heap.append((nsteps(qc, q), cnt, key, q, None)); cnt += 1
            if self.elapsed() < self.slp_deadline:
                t_end = time.time() + self.slp_budget
                stop = min([c[0] for c in heap], default=1e9)
                for key, task, ref in tasks:
                    if time.time() > t_end:
                        break
                    r = slp_safe(s, P, qc, qc, task, region, stop_steps=stop)
                    if r is not None:
                        heap.append((nsteps(qc, r[0]) - 0.5, cnt, key, r[0], None)); cnt += 1
        else:
            for key, task, ref in tasks:
                for st, th0 in base_stations(ref.pos[0], ref.pos[1]):
                    seed = np.concatenate([st, qc[3:]])
                    lb = nsteps(qc, seed)
                    heap.append((lb, cnt, key, None, (ref, seed, th0))); cnt += 1
        heapq.heapify(heap)
        while heap:
            c, _, key, q, th = heapq.heappop(heap)
            if q is not None:
                yield (c, key, q)
                continue
            ref, seed, th0 = th
            q = P._ik_pushout(ref.pos, ref.quat, seed, side_normal(th0), False)
            if q is None:
                seed2 = np.concatenate([seed[:3], CARRY_ARM])
                q = P._ik_pushout(ref.pos, ref.quat, seed2, side_normal(th0), False)
            if q is not None:
                heapq.heappush(heap, (max(c, nsteps(qc, q)), cnt, key, q, None)); cnt += 1

    def _pre(self, q, dz):
        s = self.sim
        F = s.ff.fk(q)
        quat = mat_to_quat(F[:3, :3])
        return s.ik(F[:3, 3] + np.array([0, 0, dz]), quat, q, list(range(3, 10)))

    def plan_grasp(self, flat, cand_blocks, use_stations, best=None):
        s, P = self.sim, self.planner
        qc = flat[:10].copy()
        blocks = block_info(flat, self.k)
        s.set_state(flat)
        best_cost = 1e9 if best is None else len(best[0])
        tasks = []
        for bi in cand_blocks:
            b = blocks[bi]
            byaw = quat_yaw(b['quat'])
            for m in range(4):
                yaw = wrap(byaw + m * np.pi / 2)
                ref = (np.asarray(b['pos']) + np.array([0, 0, 0.03]), top_quat(yaw))
                tasks.append((bi, TaskGrasp(b['pos'], yaw), (ref[0][0], ref[0][1], ref[0][2], ref[1])))
        tasks = [(k_, t_, (np.array(r[:3]), r[3])) for k_, t_, r in tasks]
        tasks = [(k_, t_, _Ref(r[0], r[1])) for k_, t_, r in tasks]
        cands = self._targets_solutions(qc, tasks, use_stations, best_cost)
        mg = self.la_margin
        seen = set(self._alts_keys)
        for c in cands:
            if c[0] >= best_cost + mg:
                break
            if time.time() > self.deadline and (best is not None or mg > 0 or time.time() > self.deadline + self.hard_extra):
                break
            bi, q = c[1], c[2]
            akey = (bi, int(round(q[0] / 0.35)), int(round(q[1] / 0.35)))
            if mg > 0 and akey in seen:
                continue
            lim = best_cost + mg
            pre = self._pre(q, 0.12)

            def check(fin, bi=bi):
                return fin[bidx(bi) + 7] > 0.5

            res = P.best_path(flat, qc, q, -1.0, check, pre, lim)
            if res is not None:
                if akey not in seen:
                    seen.add(akey)
                    self._alts_keys.append(akey)
                self._alts.append((len(res[0]), bi, res))
                if len(res[0]) < best_cost:
                    best = res
                    best_cost = len(res[0])
        return best

    def plan_place(self, flat, bi, use_stations, best=None):
        s, P = self.sim, self.planner
        qc = flat[:10].copy()
        blocks = block_info(flat, self.k)
        G = (tuple(flat[12:15]), tuple(flat[15:19]))
        inv_g = p.invertTransform(*G)
        occ = self._occupied_slots(blocks)
        free = [i for i in range(len(self.slots)) if i not in occ]
        s.set_state(flat)
        best_cost = 1e9 if best is None else len(best[0])
        tasks = []
        for si in free:
            sx, sy, tx, ty, ytol = self.slots[si]
            for m in range(4):
                tpos, tq = p.multiplyTransforms((sx, sy, 0.80), yaw_quat(m * np.pi / 2), inv_g[0], inv_g[1])
                task = TaskPlace(G, sx, sy, m * np.pi / 2, (tx, ty), ytol, z=(0.786, 0.95))
                tasks.append((si, task, _Ref(np.array(tpos), tq)))
        cands = self._targets_solutions(qc, tasks, use_stations, best_cost)
        for c in cands:
            if c[0] >= best_cost:
                break
            if time.time() > self.deadline and (best is not None or time.time() > self.deadline + self.hard_extra):
                break
            si, q = c[1], c[2]
            pre = self._pre(q, 0.1)
            sx, sy, tx, ty, _ = self.slots[si]

            def check(fin, sx=sx, sy=sy, tx=tx, ty=ty):
                if fin[11] > 0.5:
                    return False
                o = bidx(bi)
                return abs(fin[o] - sx) <= tx + 1e-3 and abs(fin[o + 1] - sy) <= ty + 1e-3 and fin[o + 2] < 0.79

            res = P.best_path(flat, qc, q, 1.0, check, pre, best_cost)
            if res is not None and len(res[0]) < best_cost:
                best = res
                best_cost = len(res[0])
        return best

    def replan(self, flat):
        k = self.k
        blocks = block_info(flat, k)
        n_todo = sum(1 for b in blocks if not self._block_done(b))
        rem = max(0.0, self.total_budget - self.elapsed())
        macro_b = max(0.3, min(3.0, rem / (n_todo + 1)))
        self.deadline = time.time() + macro_b
        self.slp_budget = 0.5 * macro_b
        held = [i for i, b in enumerate(blocks) if b['held']]
        if held:
            if self.next_place is not None and np.max(np.abs(self.next_place[0] - flat)) < 1e-4:
                res = self.next_place[1]
                self.next_place = None
                if res is not None:
                    return res
            self.next_place = None
            return self._place_full(flat, held[0])
        todo = [i for i, b in enumerate(blocks) if not self._block_done(b)]
        if not todo:
            return None
        tp = self.sim.ff.fk(flat[:10])[:3, 3] if hasattr(self.sim, 'ff') else np.zeros(3)
        todo.sort(key=lambda i: np.linalg.norm(blocks[i]['pos'][:2] - tp[:2]))
        use_la = self.elapsed() < self.la_deadline
        self.la_margin = 0
        self._alts, self._alts_keys = [], []
        res = self.plan_grasp(flat, todo[:4], False)
        if res is None or len(res[0]) > 4:
            res = self.plan_grasp(flat, todo, True, res)
        if use_la and res is not None:
            gbi = self._alts[-1][1] if self._alts else None
            for ln, bi_, r in self._alts:
                if r is res:
                    gbi = bi_
            self.deadline = time.time() + self.la_budget
            pl = self._place_full(res[1], gbi)
            if pl is None:
                return res
            best_tot, best_pair = len(res[0]) + len(pl[0]), (res, pl)
            if len(pl[0]) > 2 or len(res[0]) > 3:
                self.la_margin = min(6, best_tot - 2)
                self.deadline = time.time() + self.la_budget
                self.plan_grasp(flat, todo, True, res)
                self.la_margin = 0
                alts = sorted(self._alts, key=lambda t: t[0])
                n = 0
                for ln, bi_, r in alts:
                    if r is res or ln + 1 >= best_tot:
                        continue
                    if n >= self.la_n:
                        break
                    n += 1
                    self.deadline = time.time() + 0.5 * self.la_budget
                    pl2 = self._place_full(r[1], bi_)
                    if pl2 is not None and ln + len(pl2[0]) < best_tot:
                        best_tot, best_pair = ln + len(pl2[0]), (r, pl2)
            res = best_pair[0]
            self.next_place = (np.array(res[1]), best_pair[1])
        return res

    def _retreat(self, flat):
        """Fallback: move to a safe conf away from the table (keeps holding state)."""
        P = self.planner
        qc = flat[:10].copy()
        bx, by = qc[0], qc[1]
        cands = []
        for hx, hy, hth in ((-0.8, 0.0, 0.0), (0.8, 0.0, np.pi), (0.0, -1.15, np.pi / 2), (0.0, 1.2, -np.pi / 2),
                            (-0.85, -1.15, 0.0), (-0.85, 1.15, 0.0), (0.85, -1.15, np.pi), (0.85, 1.15, np.pi)):
            q = np.concatenate([[hx, hy, hth], CARRY_ARM])
            cands.append((np.hypot(hx - bx, hy - by), q))
        cands.sort(key=lambda c: c[0])
        held = flat[11] > 0.5
        for _, q in cands:
            gl = 0.0
            res = P.best_path(flat, qc, q, gl, lambda fin: (fin[11] > 0.5) == held, None, 1e9, 20)
            if res is not None:
                return res
        return None

    def _place_full(self, flat, bi):
        res = self.plan_place(flat, bi, False)
        if res is None or len(res[0]) > 4:
            res = self.plan_place(flat, bi, True, res)
        return res

    def get_action(self, state):
        flat, k = obs_to_flat(state)
        if self.plan and self.exps:
            if np.max(np.abs(flat - self.exps[0])) > 1e-3:
                self.plan = []
                self.mismatch = getattr(self, 'mismatch', 0) + 1
            else:
                self.exps.pop(0)
        if not self.plan:
            self.exps = []
            res = self.replan(flat)
            if res is None:
                res = self._retreat(flat)
            if res is None:
                self.fails += 1
                a = np.zeros(11, dtype=np.float32)
                rng = np.random.default_rng(self.fails)
                a[:10] = rng.uniform(-0.1, 0.1, 10)
                return a
            acts, fin, term = res
            exps = []
            self.sim.set_state(flat)
            for a in acts:
                o = self.sim.env.step(np.asarray(a, np.float32))[0]
                exps.append(np.array(o, float))
            if np.max(np.abs(exps[-1] - fin)) > 1e-3:
                self.mismatch = getattr(self, 'mismatch', 0) + 100
            self.plan = list(acts)
            self.exps = exps
        a = self.plan.pop(0)
        return np.clip(np.asarray(a, np.float32), self.action_space.low, self.action_space.high)
