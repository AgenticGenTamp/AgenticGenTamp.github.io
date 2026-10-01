"""Sort cluttered blocks: pick each cube and drop it into its color bin."""
import numpy as np
from fk import ik, fk

K_BASE = 0.87
K_JOINT = 0.249
JF = [f'pos_arm_joint{i}' for i in range(1, 8)]
BF = ['pos_base_x', 'pos_base_y', 'pos_base_rot']

MOUNT_Z = 0.386       # arm base height (world z)
TH = np.pi            # base yaw (fixed)
BASE_X = 0.58         # fixed base x (table blocks the base below x ~ 0.48)
BASE_Y = 0.0
Z_TRAVEL = 0.54       # world z of fingertip while travelling
Z_GRASP = 0.412       # world z of fingertip when grasping a cube on the table

BIN_ORDER = ['bin_red', 'bin_green', 'bin_blue', 'bin_yellow']
DOWN = np.array([[1., 0, 0], [0, -1, 0], [0, 0, -1]])


def tool_R(yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ DOWN


def robot_state(s):
    o = s.get_object_from_name('robot')
    b = np.array([float(s.get(o, f)) for f in BF])
    q = np.array([float(s.get(o, f)) for f in JF])
    g = float(s.get(o, 'pos_gripper'))
    return b, q, g


def obj_pos(s, name):
    o = s.get_object_from_name(name)
    return np.array([float(s.get(o, f)) for f in 'xyz'])


def rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.z_grasp = Z_GRASP
        self.z_travel = Z_TRAVEL
        self.base_xy = np.array([BASE_X, BASE_Y])

    # ---------------- helpers ----------------
    def _cubes(self, state):
        return {n: obj_pos(state, n) for n in state.get_object_names()
                if n.startswith('cube')}

    def _bin_for(self, cname):
        i = int(cname[4:])
        return self.bins[BIN_ORDER[(i - 1) % 4]]

    def reset(self, state, info):
        self.bins = {n: obj_pos(state, n) for n in state.get_object_names()
                     if n.startswith('bin_')}
        self.order = sorted(self._cubes(state), key=lambda n: int(n[4:]))
        self.idx = 0
        self.phase = 'above'
        self.t = 0
        self.bias = np.zeros(3)
        self.target = None

    def gripper_world(self, b, q):
        p = fk(q)[:3, 3]
        xy = b[:2] + rot2(b[2]) @ p[:2]
        return np.array([xy[0], xy[1], MOUNT_Z + p[2]]), p

    def _act(self, b, q, b_tgt, q_tgt, grip, gain=0.45):
        a = np.zeros(11, dtype=np.float32)
        db = np.array(b_tgt) - b
        db[2] = (db[2] + np.pi) % (2 * np.pi) - np.pi
        ab = db / K_BASE
        if np.linalg.norm(ab[:2]) < 0.045:      # stiction: skip tiny moves
            ab[:2] = 0.0
        elif np.linalg.norm(ab[:2]) < 0.06:
            ab[:2] *= 0.06 / np.linalg.norm(ab[:2])
        a[0:3] = np.clip(ab, -0.1, 0.1)
        dq = (np.array(q_tgt) - q + np.pi) % (2 * np.pi) - np.pi
        a[3:10] = np.clip(gain * dq / K_JOINT, -0.1, 0.1)
        a[10] = grip
        return a

    def _servo(self, b, q, tw, zw, grip):
        gw, p = self.gripper_world(b, q)
        err = np.array([tw[0] - gw[0], tw[1] - gw[1], zw - gw[2]])
        Rt = rot2(b[2]).T
        pn_xy = Rt @ (np.asarray(tw) - b[:2])
        p_nom = np.array([pn_xy[0], pn_xy[1], zw - MOUNT_Z])
        d = np.concatenate([Rt @ err[:2], [err[2]]])
        self.bias = np.clip(self.bias + 0.6 * d, -0.12, 0.12)
        p_t = p_nom + self.bias
        yaw = float(np.arctan2(p_nom[1], p_nom[0]))
        qt = ik(p_t, tool_R(yaw), q)
        bt = np.array([self.base_xy[0], self.base_xy[1], TH])
        return self._act(b, q, bt, qt, grip), gw

    def get_action(self, state):
        b, q, g = robot_state(state)
        cubes = self._cubes(state)
        if self.idx >= len(self.order):
            return np.zeros(11, dtype=np.float32)
        name = self.order[self.idx]
        cpos = cubes[name]
        binp = self._bin_for(name)

        if self.phase == 'above':
            tw, zw, grip = cpos[:2], self.z_travel, 0.0
        elif self.phase == 'down':
            tw, zw, grip = self.target[:2], self.z_grasp, 0.0
        elif self.phase == 'close':
            tw, zw, grip = self.target[:2], self.z_grasp, 1.0
        elif self.phase == 'lift':
            tw, zw, grip = self.target[:2], self.z_travel, 1.0
        elif self.phase == 'tobin':
            tw, zw, grip = binp[:2], self.z_travel, 1.0
        else:
            tw, zw, grip = binp[:2], self.z_travel, 0.0

        a, gw = self._servo(b, q, tw, zw, grip)

        self.t += 1
        err_xy = float(np.linalg.norm(gw[:2] - np.asarray(tw)))
        err_z = abs(gw[2] - zw)
        if self.phase == 'down':
            done = False
        elif self.phase in ('above', 'lift', 'tobin'):
            done = (err_xy < 0.005 and err_z < 0.005) or self.t > 150
        else:
            done = self.t >= 3
        if done:
            nxt = {'above': 'down', 'down': 'close', 'close': 'lift',
                   'lift': 'tobin', 'tobin': 'release', 'release': 'above'}[self.phase]
            if self.phase == 'above':
                self.target = cpos.copy()
            if self.phase == 'release':
                self.idx += 1
            self.phase = nxt
            self.t = 0
        return a
