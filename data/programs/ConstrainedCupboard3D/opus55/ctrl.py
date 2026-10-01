"""Helpers for test scripts (uses env_client)."""
import numpy as np
from kin import fk, ik, ik_best
J = ['pos_arm_joint%d' % i for i in range(1, 8)]

class Sim:
    def __init__(self, env, obs):
        self.env = env; self.obs = obs
        self.R = env.observation_space.get_type('mujoco_tidybot_robot')
        self.M = env.observation_space.get_type('mujoco_movable_object')
        self.r = obs.get_objects(self.R)[0]
        self.grip = 0.0; self.rew = []
    def q(self):
        return np.array([self.obs.get(self.r, k) for k in J])
    def base(self):
        o = self.obs; r = self.r
        return np.array([o.get(r, 'pos_base_x'), o.get(r, 'pos_base_y'), o.get(r, 'pos_base_rot')])
    def rods(self):
        out = {}
        for m in self.obs.get_objects(self.M):
            out[m.name] = np.array([self.obs.get(m, k) for k in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']])
        return out
    def step(self, a):
        self.obs, rew, te, tr, info = self.env.step(a)
        self.rew.append(rew)
        if te and not getattr(self, "term", False): print("TERMINATED at", len(self.rew), rew)
        self.term = getattr(self, "term", False) or te
        return te
    def goto(self, qt=None, bt=None, grip=None, steps=60, tol=0.01):
        if grip is not None: self.grip = grip
        for t in range(steps):
            a = np.zeros(11); a[10] = self.grip
            done = True
            if qt is not None:
                e = qt - self.q(); e[[0,2,4,6]] = (e[[0,2,4,6]] + np.pi) % (2*np.pi) - np.pi
                a[3:10] = np.clip(e, -0.1, 0.1); done &= np.abs(e).max() < tol
            if bt is not None:
                b = self.base(); e = bt - b; e[2] = (e[2] + np.pi) % (2*np.pi) - np.pi
                a[0:3] = np.clip(e, -0.1, 0.1); done &= np.abs(e).max() < tol
            if done and t > 0: return t
            self.step(a)
        return steps

MX, MY, MZ = 0.1199, 0.0, 0.3949
TOOL = 0.15   # flange -> grasp point distance along ee z

def quat_yaw(qv):
    w, x, y, z = qv[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def Rdown(yaw):
    """ee z pointing down, ee x (rod axis when grasped) at world/arm yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.]])

def world_to_arm(b, pw):
    c, s = np.cos(b[2]), np.sin(b[2]); d = pw[:2] - b[:2]
    return np.array([c * d[0] + s * d[1] - MX, -s * d[0] + c * d[1] - MY, pw[2] - MZ])

def arm_to_world(b, pa):
    c, s = np.cos(b[2]), np.sin(b[2]); x, y = pa[0] + MX, pa[1] + MY
    return np.array([b[0] + c * x - s * y, b[1] + s * x + c * y, pa[2] + MZ])

def R_world_to_arm(b, Rw):
    c, s = np.cos(b[2]), np.sin(b[2]); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz.T @ Rw

def grasp_point_world(S):
    T = fk(S.q(), TOOL); return arm_to_world(S.base(), T[:3, 3])

def move_ee_world(S, pw, Rw, bt=None, grip=None, steps=80, tol=0.005):
    """Move grasp point to world pos pw with world rotation Rw; base to bt (or stay)."""
    b = S.base() if bt is None else np.asarray(bt, float)
    qt, ok = ik_best(S.q(), world_to_arm(b, pw), R_world_to_arm(b, Rw), tool=TOOL, n_random=20)
    e1 = 0.0 if ok else 1.0
    n = S.goto(qt=qt, bt=b if bt is not None else None, grip=grip, steps=steps, tol=tol)
    return e1, n

def pick_rod(S, name, stand_off=0.55):
    """Drive base next to rod and grasp it top-down. Returns True if lifted."""
    rod = S.rods()[name]; yaw = quat_yaw(rod)
    # base: place so rod is stand_off in front (base facing +x-ish toward rod)
    b = S.base(); d = rod[:2] - b[:2]; th = np.arctan2(d[1], d[0])
    bt = np.array([rod[0] - stand_off * np.cos(th), rod[1] - stand_off * np.sin(th), th])
    S.goto(bt=bt, steps=80)
    Rw = Rdown(yaw)
    Rw2 = Rdown(yaw + np.pi)
    # choose the one needing less wrist rotation: just pick whichever IK is nearer
    p_hi = np.array([rod[0], rod[1], 0.15]); p_lo = np.array([rod[0], rod[1], 0.015 + 0.005])
    move_ee_world(S, p_hi, Rw, grip=0.0, steps=80)
    move_ee_world(S, p_lo, Rw, grip=0.0, steps=60)
    S.goto(grip=1.0, steps=8)
    move_ee_world(S, np.array([rod[0], rod[1], 0.25]), Rw, grip=1.0, steps=60)
    return S.rods()[name][2] > 0.1

def quat_to_R(qv):
    w, x, y, z = qv
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def ee_world(S):
    b = S.base(); T = fk(S.q(), TOOL)
    c, s = np.cos(b[2]), np.sin(b[2]); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return arm_to_world(b, T[:3, 3]), Rz @ T[:3, :3]

def grasp_tf(S, name):
    """Rod pose in ee frame: (p_rel, R_rel)."""
    pe, Re = ee_world(S); r = S.rods()[name]
    return Re.T @ (r[:3] - pe), Re.T @ quat_to_R(r[3:7])

def ee_for_rod(G, p_rod, R_rod):
    p_rel, R_rel = G
    Re = R_rod @ R_rel.T
    return p_rod - Re @ p_rel, Re
