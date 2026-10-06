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


