"""Sort cluttered blocks: pick each cube and drop it into its colour bin."""
import numpy as np
from fk import ik, fk

K_BASE = 0.87
K_JOINT = 0.249
JF = [f'pos_arm_joint{i}' for i in range(1, 8)]
BF = ['pos_base_x', 'pos_base_y', 'pos_base_rot']

MOUNT = np.array([0.132, 0.0, 0.388])   # arm base pose in mobile-base frame
MOUNT_Z = 0.388
TH = np.pi              # base yaw (kept fixed)
P_REF_X = 0.50          # nominal arm radial extension
BASE_X_MIN = 0.53       # table blocks the mobile base below x ~ 0.48
Z_TRAVEL = 0.56         # world z of fingertip while travelling
GRASP_DZ = 0.015        # fingertip frame offset above the cube centre
BIN_ORDER = ['bin_red', 'bin_green', 'bin_blue', 'bin_yellow']
DOWN = np.array([[1., 0, 0], [0, -1, 0], [0, 0, -1]])
GOPEN, GCLOSE = 0.0, 1.0


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
        self.z_travel = Z_TRAVEL
        self.grasp_dz = GRASP_DZ
        self.bin_order = list(BIN_ORDER)
        self.gopen, self.gclose = GOPEN, GCLOSE
        self.off = np.zeros(2)

    # ---------------- helpers ----------------
    def _cubes(self, state):
        return {n: obj_pos(state, n) for n in state.get_object_names()
                if n.startswith('cube')}

    def _cidx(self, cname):
        try:
            return int(''.join(ch for ch in cname if ch.isdigit()))
        except ValueError:
            return 0

    def _bin_for(self, cname):
        i = self._cidx(cname) - self.i0
        return self.bins[self.bin_order[i % len(self.bin_order)]]

    def reset(self, state, info):
        self.bins = {n: obj_pos(state, n) for n in state.get_object_names()
                     if n.startswith('bin')}
        if not self.bins:                      # fallback: any big movable object
            self.bins = {n: obj_pos(state, n) for n in state.get_object_names()
                         if not n.startswith('cube') and n not in ('robot',)}
        order = [n for n in BIN_ORDER if n in self.bins]
        for n in sorted(self.bins):
            if n not in order:
                order.append(n)
        self.bin_order = order
        cs = self._cubes(state)
        self.i0 = min([self._cidx(n) for n in cs]) if cs else 0
        self.idx = 0
        self.phase = 'app'
        self.t = 0
        self.bias = np.zeros(3)
        self.target = None
        self.tries = 0
        self.attempts = {}
        self.cur = None

    def gripper_world(self, b, q):
        p = fk(q)[:3, 3]
        xy = b[:2] + rot2(b[2]) @ (MOUNT[:2] + p[:2])
        return np.array([xy[0], xy[1], MOUNT_Z + p[2]]), p

    def base_des(self, tw):
        bx = tw[0] + MOUNT[0] + P_REF_X
        return np.array([max(bx, BASE_X_MIN), tw[1] + MOUNT[1]])

    def _servo(self, b, q, tw, zw, grip, move_base):
        gw, p = self.gripper_world(b, q)
        err = np.array([tw[0] - gw[0], tw[1] - gw[1], zw - gw[2]])
        Rt = rot2(b[2]).T
        pn_xy = Rt @ (np.asarray(tw) - b[:2]) - MOUNT[:2]
        p_nom = np.array([pn_xy[0], pn_xy[1], zw - MOUNT_Z])
        d = np.concatenate([Rt @ err[:2], [err[2]]])
        if np.linalg.norm(d) < 0.05:                    # anti-windup
            self.bias = np.clip(self.bias + 0.35 * d, -0.06, 0.06)
        p_t = p_nom + self.bias
        yaw = float(np.arctan2(p_t[1], p_t[0]))
        qt = ik(p_t, tool_R(yaw), q)

        a = np.zeros(11, dtype=np.float32)
        dq = (qt - q + np.pi) % (2 * np.pi) - np.pi
        a[3:10] = np.clip(0.5 * dq / K_JOINT, -0.1, 0.1)
        a[10] = grip
        # base
        yerr = (TH - b[2] + np.pi) % (2 * np.pi) - np.pi
        a[2] = np.clip(yerr / K_BASE, -0.1, 0.1)
        if move_base:
            db = self.base_des(tw) - b[:2]
            n = float(np.linalg.norm(db))
            if n > 0.012:
                v = db / K_BASE
                if n < 0.04:
                    v = v * (0.04 / n)
                a[0:2] = np.clip(v, -0.1, 0.1)
        return a, gw

    # ---------------- main ----------------
    def _placed(self, n, p):
        bp = self._bin_for(n)
        return float(np.linalg.norm(p - bp)) < 0.042 and p[2] > 0.412

    def _pick_next(self, cubes):
        best, bs = None, -1e9
        for n, p in cubes.items():
            if self.attempts.get(n, 0) >= 4 or p[2] < 0.35 or p[2] > 0.7:
                continue
            if abs(p[0]) > 0.45 or abs(p[1]) > 0.6:
                continue
            if self._placed(n, p):
                continue
            iso = min([float(np.linalg.norm(p[:2] - r[:2]))
                       for m, r in cubes.items() if m != n] or [0.1])
            sc = 3.0 * p[2] + min(iso, 0.06)
            if sc > bs:
                bs, best = sc, n
        return best

    def get_action(self, state):
        try:
            return self._get_action(state)
        except Exception:
            return np.zeros(11, dtype=np.float32)

    def _get_action(self, state):
        b, q, g = robot_state(state)
        cubes = self._cubes(state)
        if self.cur is None or self.cur not in cubes:
            self.cur = self._pick_next(cubes)
            self.phase, self.t, self.tries = 'app', 0, 0
            self.bias = np.zeros(3)
        if self.cur is None:
            return np.zeros(11, dtype=np.float32)
        name = self.cur
        cpos = cubes[name]
        binp = self._bin_for(name)
        ph = self.phase

        if ph == 'app':
            tw, zw, grip, mb = cpos[:2], self.z_travel, self.gopen, True
        elif ph == 'down':
            tw, zw, grip, mb = self.target[:2], self.target[2] + self.grasp_dz, self.gopen, False
        elif ph == 'close':
            tw, zw, grip, mb = self.target[:2], self.target[2] + self.grasp_dz, self.gclose, False
        elif ph == 'lift':
            tw, zw, grip, mb = self.target[:2], self.z_travel, self.gclose, False
        elif ph == 'tobin':
            tw, zw, grip, mb = binp[:2], self.z_travel, self.gclose, True
        else:
            tw, zw, grip, mb = binp[:2], self.z_travel, self.gopen, False

        a, gw = self._servo(b, q, tw, zw, grip, mb)

        self.t += 1
        exy = float(np.linalg.norm(gw[:2] - np.asarray(tw)))
        ez = abs(gw[2] - zw)
        if ph == 'app':
            done = (exy < 0.016 and ez < 0.02) or self.t > 60
        elif ph == 'down':
            done = (exy < 0.007 and ez < 0.004) or self.t > 55
        elif ph == 'close':
            done = self.t >= 3
        elif ph == 'lift':
            done = gw[2] > self.z_travel - 0.02 or self.t > 45
        elif ph == 'tobin':
            done = (exy < 0.02 and ez < 0.02) or self.t > 70
        else:
            done = self.t >= 4

        if done:
            nxt = None
            if ph == 'app':
                self.target = cpos.copy()
                nxt = 'down'
            elif ph == 'down':
                nxt = 'close'
            elif ph == 'close':
                nxt = 'lift'
            elif ph == 'lift':
                if cpos[2] > self.target[2] + 0.05:
                    nxt = 'tobin'
                else:
                    self.attempts[name] = self.attempts.get(name, 0) + 1
                    self.cur = None
                    nxt = 'app'
            elif ph == 'tobin':
                nxt = 'rel'
            else:
                self.attempts[name] = self.attempts.get(name, 0) + 1
                self.cur = None
                nxt = 'app'
            self.phase = nxt
            self.t = 0
            if self.cur is None:
                self.bias = np.zeros(3)
        return a
