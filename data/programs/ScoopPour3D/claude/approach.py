"""Approach for ScoopPour3DEnv: grasp the yellow bin, lift it, and pour the cubes
into the green bin using closed-loop feedback on the observed cube cluster."""
import numpy as np

# ---------------- Kinova Gen3 7-DoF kinematics (modified DH) ----------------
_DH = [
    (np.pi,   0.0, -0.2848, 0.0),
    (np.pi/2, 0.0, -0.0118, np.pi),
    (np.pi/2, 0.0, -0.4208, np.pi),
    (np.pi/2, 0.0, -0.0128, np.pi),
    (np.pi/2, 0.0, -0.3143, np.pi),
    (np.pi/2, 0.0,  0.0,    np.pi),
    (np.pi/2, 0.0, -0.1674, np.pi),
]
_TOOL_D = -0.0615
_QLIM = np.array([[-1e9, 1e9], [-2.41, 2.41], [-1e9, 1e9], [-2.66, 2.66],
                  [-1e9, 1e9], [-2.23, 2.23], [-1e9, 1e9]])


def _mdh(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [st*ca, ct*ca, -sa, -d*sa],
                     [st*sa, ct*sa, ca, d*ca],
                     [0.0, 0.0, 0.0, 1.0]])


def _fk_all(q, tool_len):
    T = np.eye(4)
    frames = []
    for i, (al, a, d, off) in enumerate(_DH):
        T = T @ _mdh(al, a, d, q[i] + off)
        frames.append(T.copy())
    T = T @ _mdh(np.pi, 0.0, _TOOL_D, 0.0)
    Tt = np.eye(4); Tt[2, 3] = tool_len
    frames.append((T @ Tt).copy())
    return frames


def _jac(frames, tool_len):
    pe = frames[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        Ti = frames[i]
        z = Ti[:3, 2]
        p = Ti[:3, 3]
        J[:3, i] = np.cross(z, pe - p)
        J[3:, i] = z
    return J


def _so3_err(Rc, Rd):
    R = Rc.T @ Rd
    w = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]) * 0.5
    s = np.linalg.norm(w)
    c = (np.trace(R) - 1) / 2.0
    ang = np.arctan2(s, c)
    if s < 1e-8:
        return np.zeros(3) if c > 0 else np.array([np.pi, 0.0, 0.0])
    return Rc @ (w / s * ang)


def ik(target_pos, target_R, q0, tool_len=0.12, iters=30):
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        frames = _fk_all(q, tool_len)
        T = frames[-1]
        ep = target_pos - T[:3, 3]
        er = _so3_err(T[:3, :3], target_R)
        e = np.concatenate([ep, 0.3*er])
        J = _jac(frames, tool_len)
        Jw = J.copy(); Jw[3:] *= 0.3
        lam = 0.05
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam**2*np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.2:
            dq *= 0.2 / n
        q = np.clip(q + dq, _QLIM[:, 0], _QLIM[:, 1])
        if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
            break
    err = np.linalg.norm(target_pos - _fk_all(q, tool_len)[-1][:3, 3])
    return q, err


def rot_down(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


# ---------------------------- constants ----------------------------
BASE_X = -0.16      # base x parked close to the island
Z_GRASP = 0.203     # arm-frame z for grasping the bin wall
Z_CLEAR = 0.34
YAW = 1.5708
TOOL = 0.12
WALL_DY = 0.15      # half width of bin in y
GRASP_DY = 0.145    # EE offset from bin centre when grasping the -y wall
RIM_H = 0.072       # bin rim height above its origin z
TIP_MAX = 2.45


def _quat_up(q):
    """world 'up' axis of a body given (qw,qx,qy,qz)."""
    w, x, y, z = q
    return np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)])


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.obs_space = observation_space

    def _pos(self, s, obj):
        return s.data[obj][:3].copy()

    def reset(self, state, info):
        self._ok = False
        self._robot = state.get_object_from_name('robot')
        names = state.get_object_names()
        self._green = state.get_object_from_name('bin_green_0')
        self._yellow = state.get_object_from_name('bin_yellow_0')
        self._cubes = [state.get_object_from_name(n) for n in sorted(names)
                       if n.startswith('cube_')]
        self.phase = 'BASE1'
        self.t = 0
        self.phase_t = 0
        self.qd = state.data[self._robot][3:10].copy()
        self.grip = 0.0
        self.a = 0.0
        self.fx = 0.6
        self.zarm = Z_CLEAR
        g = self._pos(state, self._green)
        y = self._pos(state, self._yellow)
        self.gx, self.gy, self.gz = float(g[0]), float(g[1]), float(g[2])
        self.wx = float(np.clip(y[0], 0.40, 0.55))
        self.z0 = float(y[2])
        self.zg = Z_GRASP + (float(y[2]) - 0.459)
        self.stage = 0
        self.phi = 0.0
        self.ydes = float(y[1])
        self.zdes = float(y[2])
        self.retries = 0
        self.piv = None
        self._ok = True
        self.attempt = 0
        self.stage_t = 0
        self.heavy = False
        self.corr = 0.0
        self.stage_t = 0

    # -------------- helpers --------------
    def _set_arm(self, q_now, pos, R, iters=None):
        key = (round(pos[0], 4), round(pos[1], 4), round(pos[2], 4), round(float(R[1, 2]), 4), round(float(R[2, 2]), 4))
        if iters is None:
            iters = 40 if getattr(self, '_last_key', None) is None else 12
        seed = self.qd if getattr(self, '_last_key', None) is not None else q_now
        qd, err = ik(np.array(pos), R, seed, TOOL, iters=iters)
        self._last_key = key
        self.qd = qd
        return err

    def _act(self, state, base_cmd=(0.0, 0.0, 0.0)):
        q = state.data[self._robot][3:10]
        a = np.zeros(11, dtype=np.float32)
        a[:3] = np.clip(base_cmd, -0.1, 0.1)
        a[3:10] = np.clip(self.qd - q, -0.1, 0.1)
        a[10] = self.grip
        return a

    def _base_cmd(self, state, bx, by, gain=1.0, mv=0.1):
        b = state.data[self._robot][:3]
        e = np.array([bx - b[0], by - b[1], -b[2]]) * gain
        return np.clip(e, -mv, mv)

    def _settled(self, state, tol=0.012):
        q = state.data[self._robot][3:10]
        return np.abs(self.qd - q).max() < tol

    # ---------------- main ----------------
    def get_action(self, state):
        try:
            if not self._ok:
                return np.zeros(11, dtype=np.float32)
            return self._get_action(state)
        except Exception:
            return np.zeros(11, dtype=np.float32)

    def _get_action(self, state):
        self.t += 1
        self.phase_t += 1
        rb = state.data[self._robot][:3]
        yb = state.data[self._yellow]
        gb = state.data[self._green]
        ybin = yb[:3]
        self.gx, self.gy, self.gz = float(gb[0]), float(gb[1]), float(gb[2])
        self.fx = float(np.clip(self.wx - rb[0], 0.30, 0.72))

        if self.phase == 'RESET':
            self.grip = 0.0
            self.a = 0.0
            self.corr = 0.0
            self._set_arm(state.data[self._robot][3:10],
                          (0.45, 0.0, 0.42), rot_down(YAW))
            if self._settled(state, 0.02) or self.phase_t > 70:
                self.phase, self.phase_t = 'BASE1', 0
            return self._act(state)

        if self.phase == 'BASE1':
            self.grip = 0.0
            bc = self._base_cmd(state, BASE_X, ybin[1] - GRASP_DY)
            if np.abs(bc).max() < 0.012 or self.phase_t > 40:
                self.phase, self.phase_t = 'PREGRASP', 0
            return self._act(state, bc)

        if self.phase == 'PREGRASP':
            self.grip = 0.0
            self._set_arm(state.data[self._robot][3:10],
                          (self.fx, 0.0, Z_CLEAR), rot_down(YAW))
            if self._settled(state) or self.phase_t > 90:
                self.phase, self.phase_t = 'ALIGN', 0
            bc = self._base_cmd(state, BASE_X, ybin[1] - GRASP_DY, gain=0.6, mv=0.05)
            return self._act(state, bc)

        if self.phase == 'ALIGN':
            self.grip = 0.0
            self._set_arm(state.data[self._robot][3:10],
                          (self.fx, 0.0, Z_CLEAR), rot_down(YAW))
            bc = self._base_cmd(state, BASE_X, ybin[1] - GRASP_DY, gain=0.8, mv=0.04)
            if np.abs(bc).max() < 0.008 or self.phase_t > 40:
                self.phase, self.phase_t = 'DOWN', 0
            return self._act(state, bc)

        if self.phase == 'DOWN':
            self.grip = 0.0
            self._set_arm(state.data[self._robot][3:10],
                          (self.fx, 0.0, self.zg), rot_down(YAW))
            if self._settled(state) or self.phase_t > 40:
                self.phase, self.phase_t = 'CLOSE', 0
            return self._act(state)

        if self.phase == 'CLOSE':
            self.grip = 1.0
            if self.phase_t > 9:
                self.zarm = self.zg
                self.ydes = float(ybin[1])
                self.zdes = float(ybin[2])
                self.stage = 1
                self.phase, self.phase_t = 'CTRL', 0
            return self._act(state)

        if self.phase == 'CTRL':
            self.grip = 1.0
            up = _quat_up(yb[3:7])
            tilt = float(np.arctan2(up[1], max(up[2], 1e-6)))
            if self.piv is None:
                piv_y = self.gy - 0.06
                piv_z = self.gz + RIM_H + 0.05
            else:
                piv_y, piv_z = self.piv
            # ---- desired trajectory ----
            if self.stage == 1:      # lift clear of green rim, then move across
                self.zdes = min(self.zdes + 0.012, piv_z)
                tilt_des = 0.0
                lifted = float(ybin[2]) > self.z0 + 0.03
                if self.phase_t > 26 and not lifted and self.retries < 3:
                    # grasp failed: back off and retry lower
                    self.retries += 1
                    self.zg = max(self.zg - 0.012, 0.17)
                    self.grip = 0.0
                    self.zdes = float(ybin[2])
                    self.phase, self.phase_t = 'PREGRASP', 0
                    return self._act(state)
                if self.zdes >= piv_z - 0.001 and float(ybin[2]) > piv_z - 0.04:
                    self.stage = 2
            elif self.stage == 2:    # move +y so far edge sits over green rim
                tgt = piv_y - WALL_DY
                self.zdes = piv_z
                self.ydes += float(np.clip(tgt - self.ydes, -0.018, 0.018))
                tilt_des = 0.0
                if abs(tgt - self.ydes) < 0.005 and abs(ybin[1] - self.ydes) < 0.03:
                    self.stage = 3
                    self.piv = (piv_y, piv_z)
            elif self.stage == 4:    # drain: raise while tilted, shake
                self.stage_t += 1
                tilt_des = self.phi + 0.22 * np.sin(self.stage_t * 0.5)
                self.zdes = piv_z + WALL_DY * np.sin(self.phi) + min(0.10, 0.004 * self.stage_t)
                self.ydes = piv_y - WALL_DY * np.cos(self.phi)
                P = np.array([state.data[c][2] for c in self._cubes])
                empty = (P > self.gz + 0.06).sum() == 0
                if self.stage_t > 40 or (empty and self.stage_t > 12):
                    self.stage = 5
                    self.stage_t = 0
            elif self.stage == 5:    # retreat: carry bin away from green bin
                self.stage_t += 1
                tilt_des = max(0.4, self.phi - 0.02 * self.stage_t)
                self.ydes -= 0.012
                self.zdes = piv_z + WALL_DY * np.sin(max(tilt_des, 0.0)) + 0.10
                if self.stage_t > 45:
                    self.phase, self.phase_t = 'SETTLE', 0
            else:                    # tip about far edge
                self.phi = min(self.phi + 0.028, TIP_MAX)
                tilt_des = self.phi
                self.ydes = piv_y - WALL_DY * np.cos(self.phi)
                self.zdes = piv_z + WALL_DY * np.sin(self.phi)
                if self.phi >= TIP_MAX - 1e-6:
                    # shake to dislodge remaining cubes
                    tilt_des = self.phi + 0.16 * np.sin(self.phase_t * 0.45)
                P = np.array([state.data[c][2] for c in self._cubes])
                empty = (P > self.gz + 0.06).sum() == 0
                if (self.phi >= TIP_MAX - 1e-6 and self.phase_t > 110) or \
                        (empty and self.phi > 1.4):
                    self.stage = 4
                    self.stage_t = 0
            # ---- feedback ----
            self.corr += float(np.clip((tilt_des - tilt) * 0.35, -0.02, 0.02))
            self.corr = float(np.clip(self.corr, -0.8, 0.8))
            a_t = -tilt_des + self.corr
            self.a += float(np.clip(a_t - self.a, -0.03, 0.03))
            self.zarm += float(np.clip((self.zdes - float(ybin[2])) * 0.5, -0.012, 0.012))
            self.zarm = float(np.clip(self.zarm, 0.18, 0.60))
            self.fx += float(np.clip((self.gx - float(ybin[0])) * 0.2, -0.005, 0.005))
            self.fx = float(np.clip(self.fx, 0.35, 0.72))
            self._set_arm(state.data[self._robot][3:10],
                          (self.fx, 0.0, self.zarm), Rx(self.a) @ rot_down(YAW))
            bx = float(np.clip((self.gx - float(ybin[0])) * 0.5, -0.04, 0.02))
            bc = (bx, float(np.clip((self.ydes - float(ybin[1])) * 0.7, -0.04, 0.04)),
                  -rb[2])
            return self._act(state, bc)

        # SETTLE: if the pour clearly failed, release and try the whole thing again
        self.grip = 1.0
        if self.phase_t > 60 and self.t < 700 and self.attempt < 2:
            P = np.array([state.data[c][:3] for c in self._cubes])
            ok = ((np.abs(P[:, 0] - self.gx) < 0.22) &
                  (np.abs(P[:, 1] - self.gy) < 0.16) &
                  (P[:, 2] > self.gz - 0.01) & (P[:, 2] < self.gz + 0.12))
            if ok.mean() < 0.5:
                self.attempt += 1
                self.retries = 0
                self.piv = None
                self.corr = 0.0
                self.stage = 0
                self.stage_t = 0
                self.phi = 0.0
                self.a = 0.0
                self.heavy = False
                self.grip = 0.0
                y = state.data[self._yellow][:3]
                self.wx = float(np.clip(y[0], 0.40, 0.55))
                self.z0 = float(y[2])
                self.zg = Z_GRASP + (float(y[2]) - 0.459)
                self.phase, self.phase_t = 'RESET', 0
        return self._act(state)
