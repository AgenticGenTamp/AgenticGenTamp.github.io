"""BalanceBeam3D-o3: pick three cubes and place them on the seesaw.

Control model (calibrated empirically):
  action[0:2] : base x/y world-frame delta, achieved = 0.87 * a
  action[2]   : base yaw delta, achieved = 0.994 * a
  action[3:10]: arm joint deltas, achieved = 0.25 * a  (target = measured + delta)
  action[10]  : absolute gripper (0 open, 1 closed)
Kinematics: Kinova Gen3 7-DoF DH chain, arm mounted at (0.13, 0, 0.45) in the
mobile-base frame; tool point (fingertips) 0.20 m past the wrist flange.
"""
import numpy as np

# ---------------------------------------------------------------- kinematics
_DH = [
    (0.0,     -0.2848, np.pi/2),
    (np.pi,   -0.0118, np.pi/2),
    (np.pi,   -0.4208, np.pi/2),
    (np.pi,   -0.0128, np.pi/2),
    (np.pi,   -0.3143, np.pi/2),
    (np.pi,    0.0,    np.pi/2),
    (np.pi,   -0.1674, np.pi),
]
_QLO = np.array([-1e9, -2.20, -1e9, -2.55, -1e9, -2.05, -1e9])
_QHI = np.array([ 1e9,  1.80,  1e9,  2.55,  1e9,  2.05,  1e9])
_T0 = np.array([[1.,0,0,0],[0,-1.,0,0],[0,0,-1.,0],[0,0,0,1.]])

MOUNT = np.array([0.13, 0.0, 0.45])
TOOL = 0.20
ARM_GAIN = 0.25
BASE_GAIN = 0.87
YAW_GAIN = 0.994


def _dh_T(th, d, al):
    ct, st = np.cos(th), np.sin(th)
    ca, sa = np.cos(al), np.sin(al)
    return np.array([[ct, -st*ca,  st*sa, 0.0],
                     [st,  ct*ca, -ct*sa, 0.0],
                     [0.0,    sa,     ca,   d],
                     [0.0,   0.0,    0.0, 1.0]])


def _fk_chain(q, tool=0.0):
    T = _T0.copy(); out = []
    for i in range(7):
        tho, d, al = _DH[i]
        T = T @ _dh_T(q[i] + tho, d, al)
        out.append(T)
    if tool:
        Tt = np.eye(4); Tt[2, 3] = tool
        out.append(T @ Tt)
    return out


def fk(q, tool=0.0):
    return _fk_chain(q, tool)[-1]


def jacobian(q, tool=0.0):
    Ts = _fk_chain(q, tool)
    pe = Ts[-1][:3, 3]
    J = np.zeros((6, 7))
    Tprev = _T0.copy()
    for i in range(7):
        z = Tprev[:3, 2]; p = Tprev[:3, 3]
        J[:3, i] = np.cross(z, pe - p)
        J[3:, i] = z
        Tprev = Ts[i]
    return J


def ik_step(q, p_des, z_des, tool=0.0, yaw_des=None, rot_w=0.5, damp=0.06, null_w=0.02):
    T = fk(q, tool)
    ep = p_des - T[:3, 3]
    zc = T[:3, 2]
    er = np.cross(zc, z_des)
    s = np.linalg.norm(er); c = float(np.dot(zc, z_des))
    ang = np.arctan2(s, c)
    er = er / s * ang if s > 1e-9 else (np.zeros(3) if c > 0 else np.array([np.pi, 0, 0]))
    er = er * rot_w
    if yaw_des is not None:
        er = er + 0.25 * rot_w * np.cross(T[:3, 0], yaw_des)
    e = np.concatenate([ep, er])
    J = jacobian(q, tool)
    Jinv = J.T @ np.linalg.inv(J @ J.T + (damp ** 2) * np.eye(6))
    dq = Jinv @ e
    dn = np.zeros(7)
    for i in (1, 3, 5):
        dn[i] = -(q[i] - 0.5 * (_QLO[i] + _QHI[i]))
    dq = dq + (np.eye(7) - Jinv @ J) @ (dn * null_w)
    for i in (1, 3, 5):
        if q[i] + dq[i] < _QLO[i]: dq[i] = _QLO[i] - q[i]
        if q[i] + dq[i] > _QHI[i]: dq[i] = _QHI[i] - q[i]
    return dq


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def w2a(p, base):
    return Rz(base[2]).T @ (p - np.array([base[0], base[1], 0.0])) - MOUNT


def ee_world(obs):
    b = obs[16:19]
    return np.array([b[0], b[1], 0.0]) + Rz(b[2]) @ (MOUNT + fk(obs[19:26], TOOL)[:3, 3])


def yaw_of(quat):
    return 2.0 * np.arctan2(quat[3], quat[0])


def quat_mat(q):
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w),   s*(x*z+y*w)],
        [s*(x*y+z*w),   1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w),   s*(y*z+x*w),   1-s*(x*x+y*y)],
    ])


def beam_frame(obs):
    """Return (centre_xy, long-axis unit vec, cross-axis unit vec) of the plank."""
    R = quat_mat(obs[41:45])
    u = R[:2, 0]
    nu = np.linalg.norm(u)
    u = u / nu if nu > 1e-6 else np.array([0.0, 1.0])
    v = np.array([-u[1], u[0]])
    return obs[38:40].copy(), u, v


def plank_point(obs, dy):
    """World point on the plank's top face at offset `dy` along the beam,
    together with the plank's surface normal (handles a tipped beam)."""
    R = quat_mat(obs[41:45])
    p = obs[38:41] + R @ np.array([dy, 0.0, SURF_LOCAL])
    return p, R[:, 2]


def beam_local(obs, p):
    R = quat_mat(obs[41:45])
    return R.T @ (np.asarray(p, float) - obs[38:41])


# ---------------------------------------------------------------- approach
REACH = 0.45          # base stand-off from the manipulation point
CARRY_Z = 0.20        # tool height while transporting
SURF = 0.047          # top surface of the seesaw plank (level beam)
SURF_LOCAL = 0.0472   # plank top face offset from the seesaw origin, body frame
GRIP_DZ = 0.032       # block centre height above the tool point when grasped
PLACE_DZ = 0.005      # extra clearance at release
LEAD = 0.02           # max set-point lead over the planned tool position
MIN_TOOL_Z = 0.009    # never take a loaded tool closer than this to the plank
BLOCKS = [0, 54, 70]  # obs offsets of large_block, small_block_1, small_block_2
PRE_Z = 0.115         # pre-grasp tool height
DY = 0.055            # placement offset along the beam from its centre
HOLD = 55             # steps spent holding the last cube over the plank
GRASP_LAT = -0.012    # lateral aim shift (tool x) that best centres the fingers


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # -- low level -------------------------------------------------------
    def _arm_action(self, ff=None):
        err = self.q_des - self.obs[19:26]
        self.qi = np.clip(self.qi + 0.04 * err, -0.3, 0.3)
        if ff is None:
            ff = np.zeros(7)
        return np.clip(4.0 * (ff + 0.5 * err + self.qi), -0.1, 0.1)

    def _base_action(self, base, vmax):
        a = np.zeros(3)
        if base is None:
            return a
        a[0] = np.clip((base[0] - self.obs[16]) / BASE_GAIN, -vmax, vmax)
        a[1] = np.clip((base[1] - self.obs[17]) / BASE_GAIN, -vmax, vmax)
        a[2] = np.clip((base[2] - self.obs[18]) / YAW_GAIN, -0.1, 0.1)
        return a

    def _go(self, goal, base=None, rel=False, fyaw=None, fmod=np.pi/2, speed=0.03,
            tol=0.004, maxsteps=200, settle=4, bvmax=0.1, lead=LEAD, lat=0.0):
        goal = np.asarray(goal, float)
        done = 0
        for _ in range(maxsteps):
            b = self.obs[16:19]
            g = np.array([b[0] + goal[0], b[1] + goal[1], goal[2]]) if rel else goal
            yd = None
            if fyaw is not None:
                Rw = Rz(b[2]) @ fk(self.q_des, TOOL)[:3, :3]
                cur = np.arctan2(Rw[1, 0], Rw[0, 0])
                fy = fyaw + np.round((cur - fyaw) / fmod) * fmod
                yd = Rz(b[2]).T @ np.array([np.cos(fy), np.sin(fy), 0.0])
                if lat:
                    g = g + lat * np.array([np.cos(cur), np.sin(cur), 0.0])
            d = g - self.setp
            n = float(np.linalg.norm(d))
            # rubber band: never let the set-point run away from the planned tool
            b3 = np.array([b[0], b[1], 0.0])
            ee_plan = b3 + Rz(b[2]) @ (MOUNT + fk(self.q_des, TOOL)[:3, 3])
            self.setp = self.setp + d / n * speed if n > speed else g.copy()
            lag = self.setp - ee_plan
            ln = float(np.linalg.norm(lag))
            if ln > lead:
                self.setp = ee_plan + lag / ln * lead
            dq = ik_step(self.q_des, w2a(self.setp, b), np.array([0., 0., -1.]),
                         tool=TOOL, yaw_des=yd)
            sd = np.clip(dq, -0.024, 0.024)
            self.q_des = self.q_des + sd
            a = np.zeros(11)
            a[3:10] = self._arm_action(sd)
            a[10] = self.grip
            a[0:3] = self._base_action(base, bvmax)
            yield a
            bok = base is None or (abs(base[0] - self.obs[16]) < 2e-3 and
                                   abs(base[1] - self.obs[17]) < 2e-3)
            if n < 1e-6 and bok and np.linalg.norm(g - ee_world(self.obs)) < tol:
                done += 1
                if done >= settle:
                    return

    def _set_grip(self, g, n=14):
        self.grip = g
        for _ in range(n):
            a = np.zeros(11)
            a[3:10] = self._arm_action()
            a[10] = g
            yield a

    # -- plan ------------------------------------------------------------
    def _grasp_yaw(self, bi):
        """Pick the finger axis (cube yaw or +90 deg) that best clears neighbours."""
        p = self.obs[bi:bi + 2]
        cyaw = yaw_of(self.obs[bi + 3:bi + 7])
        best, best_pen = cyaw, None
        for cand in (cyaw, cyaw + np.pi / 2):
            ax = np.array([np.cos(cand), np.sin(cand)])
            pe = np.array([-ax[1], ax[0]])
            pen = 0.0
            for oj in BLOCKS:
                if oj == bi:
                    continue
                d = self.obs[oj:oj + 2] - p
                if np.linalg.norm(d) > 0.16:
                    continue
                along, perp = abs(float(d @ ax)), abs(float(d @ pe))
                if perp < 0.05:
                    pen += max(0.0, 0.085 - along)
            if best_pen is None or pen < best_pen - 1e-9:
                best, best_pen = cand, pen
        return best

    def _pick(self, bi):
        """Drive to cube `bi` and grasp it.  Returns with the cube at CARRY_Z."""
        blk = self.obs[bi:bi + 3].copy()
        yaw = self._grasp_yaw(bi)
        base = [blk[0] - REACH, blk[1], 0.0]
        self.grip = 0.0                       # always approach with an open hand
        for a in self._go([blk[0], blk[1], PRE_Z], base=base, fyaw=yaw, fmod=np.pi,
                          lead=0.10, speed=0.06, tol=0.008, maxsteps=140, settle=2):
            yield a
        blk = self.obs[bi:bi + 3].copy()
        for a in self._go([blk[0], blk[1], PRE_Z], fyaw=yaw, fmod=np.pi, lead=0.03,
                          speed=0.03, tol=0.007, maxsteps=45, settle=2, lat=GRASP_LAT):
            yield a
        zg = max(0.004, float(blk[2]) - float(self.obs[bi + 13]) / 2.0 + 0.004)
        for a in self._go([blk[0], blk[1], zg], fyaw=yaw, fmod=np.pi, lead=0.015,
                          speed=0.025, tol=0.005, maxsteps=55, settle=3, lat=GRASP_LAT):
            yield a
        for a in self._set_grip(1.0, n=11):
            yield a
        for a in self._go([blk[0], blk[1], CARRY_Z], fyaw=yaw, fmod=np.pi, lead=0.03,
                          speed=0.03, tol=0.01, maxsteps=70, settle=1):
            yield a

    def _carry_over(self, bi, dy):
        """Move the held cube above beam offset `dy`; returns (eep, pyaw, ok)."""
        Rw = Rz(self.obs[18]) @ fk(self.obs[19:26], TOOL)[:3, :3]
        self.off_tool = Rw.T @ (self.obs[bi:bi + 3] - ee_world(self.obs))
        c, u, v = beam_frame(self.obs)
        tgt = c + u * dy
        self.pyaw = float(np.arctan2(v[1], v[0]))
        for a in self._go([REACH, 0.0, CARRY_Z], base=[tgt[0] - REACH, tgt[1], 0.0],
                          rel=True, fyaw=self.pyaw, fmod=np.pi, lead=0.10, speed=0.06,
                          tol=0.02, maxsteps=90, settle=1, bvmax=0.04):
            yield a
        eep = self._release_point(bi, dy)
        for a in self._go([eep[0], eep[1], CARRY_Z], fyaw=self.pyaw, fmod=np.pi,
                          lead=0.03, speed=0.05, tol=0.008, maxsteps=45, settle=2):
            yield a

    def _release_point(self, bi, dy):
        half = float(self.obs[bi + 13]) / 2.0
        sp, nrm = plank_point(self.obs, dy)
        Rw = Rz(self.obs[18]) @ fk(self.q_des, TOOL)[:3, :3]
        off_w = Rw @ self.off_tool
        eep = sp + nrm * (half + PLACE_DZ) - off_w
        eep[2] = max(eep[2], sp[2] + MIN_TOOL_Z)
        return eep

    def _place(self, bi, dy):
        eep = self._release_point(bi, dy)
        for a in self._go(eep, fyaw=self.pyaw, fmod=np.pi, lead=0.015, speed=0.025,
                          tol=0.006, maxsteps=55, settle=3):
            yield a
        for a in self._set_grip(0.0, n=8):
            yield a
        for a in self._go([eep[0], eep[1], CARRY_Z], lead=0.03, speed=0.04,
                          tol=0.02, maxsteps=50, settle=1):
            yield a

    def _over_beam(self, bi, margin=0.0):
        """True if cube `bi` sits inside the plank footprint (height ignored)."""
        l = beam_local(self.obs, self.obs[bi:bi + 3])
        return abs(l[0]) < 0.17 - margin and abs(l[1]) < 0.030 - margin

    def _resting(self, bi):
        l = beam_local(self.obs, self.obs[bi:bi + 3])
        return self._over_beam(bi) and l[2] > SURF_LOCAL - 0.005

    def _order(self):
        """Cubes on top of a pile first, then the most isolated ones."""
        def iso(b):
            return min([float(np.linalg.norm(self.obs[o:o + 2] - self.obs[b:b + 2]))
                        for o in BLOCKS if o != b] or [1.0])
        bs = sorted(BLOCKS, key=lambda b: (-round(float(self.obs[b + 2]), 3), -iso(b)))
        out = []
        for k, b in enumerate(bs):
            if k == 2:
                out.append((b, 0.0))
            else:
                w = (float(self.obs[b + 13]) / 0.024) ** 3      # relative mass
                out.append((b, (-DY if k == 0 else DY) / w))
        return out

    def _plan(self):
        order = self._order()
        for k, (bi, dy) in enumerate(order):
            last = (k == len(order) - 1)
            for attempt in range(3):
                if self._resting(bi) and not last:
                    break
                if self.n > self.budget - (150 if last else 200):
                    break
                for a in self._pick(bi):
                    yield a
                if self.obs[bi + 2] < 0.12:          # grasp failed - try again
                    continue
                for a in self._carry_over(bi, dy):
                    yield a
                if last:
                    # merely holding the cube above the plank satisfies the goal
                    for _ in range(HOLD):
                        yield self._hold_action()
                for a in self._place(bi, dy):
                    yield a
                if self._resting(bi):
                    break
        # repair: anything that ended up off the plank gets another try
        while self.n < self.budget - 230:
            bad = [b for b in BLOCKS if not self._over_beam(b)]
            if not bad:
                break
            bi = bad[0]
            dy = dict(order).get(bi, 0.0)
            for a in self._pick(bi):
                yield a
            if self.obs[bi + 2] < 0.12:
                continue
            for a in self._carry_over(bi, dy):
                yield a
            for a in self._place(bi, dy):
                yield a
        while True:
            yield self._hold_action()

    def _hold_action(self):
        a = np.zeros(11)
        a[3:10] = self._arm_action()
        a[10] = self.grip
        return a

    # -- interface -------------------------------------------------------
    def reset(self, state, info=None):
        self.obs = np.asarray(state, dtype=float)
        self.q_des = self.obs[19:26].copy()
        self.qi = np.zeros(7)
        self.setp = ee_world(self.obs)
        self.grip = 0.0
        self.n = 0
        self.budget = int(getattr(self, "max_steps", 1000))
        self.off_tool = np.zeros(3)
        self.pyaw = 0.0
        self.gen = self._plan()

    def get_action(self, state):
        self.obs = np.asarray(state, dtype=float)
        self.n += 1
        try:
            a = next(self.gen)
        except StopIteration:
            a = self._hold_action()
        except Exception:                      # never crash the episode
            self.gen = self._plan()
            a = np.zeros(11)
            a[10] = self.grip
        if a is None or np.any(~np.isfinite(np.asarray(a, dtype=float))):
            a = np.zeros(11)
            a[10] = self.grip
        out = np.asarray(a, dtype=np.float32).copy()
        out[:10] = np.clip(out[:10], -0.1, 0.1)
        out[10] = np.clip(out[10], 0.0, 1.0)
        return out
