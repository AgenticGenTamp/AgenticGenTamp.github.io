"""Push-based policy for Rearrange3D: put boxed drink and can next to the bowl."""
import time
import numpy as np
import kinova_fk as K

Z_OFF = 0.44          # world_z of model TCP = fk_z + Z_OFF
GRIP_DX = 0.04        # real gripper x = model TCP x + GRIP_DX
PUSH_Z = 0.515        # model TCP world z while pushing
CLEAR_Z = 0.62        # safe travel height
BASE_X = -0.20
NEAR = 0.125          # desired final centre distance to bowl

def R_down(yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]]) @ np.array([[1,0,0],[0,-1,0],[0,0,-1.]])

def _rot_log(R):
    c = (np.trace(R)-1.0)*0.5
    c = max(-1.0, min(1.0, c))
    th = np.arccos(c)
    if th < 1e-6:
        return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])*0.5
    return (th/(2*np.sin(th)))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])


def fast_ik(p, Rd, q0, iters=40, lam=0.08):
    """Damped least squares IK in the arm base frame. Returns (q, pos_err)."""
    q = np.array(q0, dtype=float)
    best = None; bestе = 1e9
    for _ in range(iters):
        T = K.fk(q)
        ep = np.asarray(p, float) - T[:3, 3]
        er = _rot_log(Rd @ T[:3, :3].T)
        e = np.concatenate([ep, 0.5*er])
        n = np.linalg.norm(ep)
        if n < bestе:
            bestе = n; best = q.copy()
        if n < 0.0015 and np.linalg.norm(er) < 0.02:
            return q, n
        J = K.jacobian(q)
        JJt = J @ J.T + (lam**2)*np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)
        nd = np.linalg.norm(dq)
        if nd > 0.4:
            dq *= 0.4/nd
        q = q + dq
    return best, bestе


def world_to_rel(p, base):
    bx, by, th = base[0], base[1], base[2]
    v = np.array([p[0]-bx, p[1]-by]); c, s = np.cos(th), np.sin(th)
    return np.array([c*v[0]+s*v[1], -s*v[0]+c*v[1], p[2]-Z_OFF])

def rel_to_world(r, base):
    bx, by, th = base[0], base[1], base[2]; c, s = np.cos(th), np.sin(th)
    return np.array([bx+c*r[0]-s*r[1], by+s*r[0]+c*r[1], r[2]+Z_OFF])


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.observation_space = observation_space

    # ---------- helpers ----------
    def obj(self, s, i):
        return np.asarray(s, dtype=float)[i:i+3]

    def base(self, s):
        return np.asarray(s, dtype=float)[93:96]

    def q(self, s):
        return np.asarray(s, dtype=float)[96:103]

    def tcp(self, s):
        s = np.asarray(s, dtype=float)
        return rel_to_world(K.fk(s[96:103])[:3, 3], s[93:96])

    def reset(self, state, info=None):
        s = np.asarray(state, dtype=float)
        self.t = 0
        self.qhold = None
        self.bias = np.zeros(3)
        self.phase = 0
        self.phase_t = 0
        self.gdx = GRIP_DX
        self.t0 = time.time()
        self.cur = None
        self.task_i = 0
        self.done = False
        self.push_dir = 0.0
        self.qtarget = None
        return None

    # phase list per object:
    # 0 rotate base to 0 / move base y to staging
    # 1 move base x
    # 2 arm to above-approach point
    # 3 arm down to push height
    # 4 push with base
    # 5 lift arm, retreat base
    def _act(self, base_cmd=(0.0, 0.0, 0.0), q_target=None, s=None, kp=3.0):
        a = np.zeros(11, dtype=np.float32)
        a[0:3] = np.clip(base_cmd, -0.1, 0.1)
        if q_target is not None:
            a[3:10] = np.clip(kp*(q_target - self.q(s)), -0.1, 0.1)
        a[10] = 0.0
        return a

    def _goal_dy(self, o, bowl):
        dx = o[0]-bowl[0]
        r2 = NEAR*NEAR - dx*dx
        return np.sqrt(r2) if r2 > 0.0004 else 0.0

    def _ik(self, s, p, yaw=0.0):
        b = self.base(s)
        cur = self.tcp(s)
        err = np.asarray(p, float) - cur
        self.bias = np.clip(self.bias + 0.8*err, -0.25, 0.25)
        rel = world_to_rel(np.asarray(p, float) + self.bias, b)
        qt, err = fast_ik(rel, R_down(yaw), self.q(s))
        if err > 0.02:
            rel = world_to_rel(np.asarray(p, float), b)
            qt2, err2 = fast_ik(rel, R_down(yaw), self.q(s))
            if err2 < err:
                qt = qt2
                self.bias = np.zeros(3)
        return qt

    def _tuck_point(self, s):
        b = self.base(s)
        return np.array([b[0]+0.34, b[1], 0.80])

    def get_action(self, state):
        s = np.asarray(state, dtype=float)
        self.t += 1
        self.phase_t += 1
        if self.t > 900 or (time.time()-self.t0) > 48.0:
            return np.zeros(11, dtype=np.float32)
        if getattr(self, 'finished', False):
            return np.zeros(11, dtype=np.float32)
        if self.cur is None or (self.phase == 0 and self.phase_t == 1):
            bowl0 = s[0:3]
            dd = np.linalg.norm(s[16:18]-bowl0[:2])
            dc = np.linalg.norm(s[32:34]-bowl0[:2])
            if max(dd, dc) <= NEAR + 0.03 and self.cur is not None:
                self.finished = True
                return np.zeros(11, dtype=np.float32)
            if self.cur is None:
                self.cur = 32 if dc >= dd else 16
            else:
                other = 16 if self.cur == 32 else 32
                do = dd if other == 16 else dc
                dcur = dc if self.cur == 32 else dd
                if do > NEAR + 0.03 and do >= dcur - 0.02:
                    self.cur = other
            self.done_ct = getattr(self, 'done_ct', 0)
        idx = self.cur
        o = s[idx:idx+3]
        bowl = s[0:3]
        b = self.base(s)
        if self.phase <= 1:
            self.sign = 1.0 if o[1] < bowl[1] else -1.0
            self.obj_x = o[0]
            self.steer = 0.0
        sign = self.sign
        stage_y = o[1] - sign*0.13
        bx = BASE_X if self.obj_x < 0.46 else -0.15

        if self.phase == 0:      # tuck arm back/up before driving base
            p = self._tuck_point(s)
            if self.phase_t % 5 == 1 or self.qtarget is None:
                qt = self._ik(s, p)
                if qt is not None:
                    self.qtarget = qt
            if self.tcp(s)[2] > 0.74 or self.phase_t > 45:
                self.phase, self.phase_t, self.bias = 1, 0, np.zeros(3)
                self.qtarget = None
                self.qhold = self.q(s).copy()
            return self._act((0, 0, 0), self.qtarget, s)

        if self.phase == 1:      # base: yaw, y, x
            eth = -((b[2]+np.pi) % (2*np.pi) - np.pi)
            ey = stage_y - b[1]
            ex = bx - b[0]
            if (abs(ey) < 0.012 and abs(eth) < 0.02 and abs(ex) < 0.02) or self.phase_t > 70:
                self.phase, self.phase_t, self.bias = 2, 0, np.zeros(3)
                self.qtarget = None
            cx = np.clip(1.2*ex, -0.04, 0.04) if abs(ey) < 0.05 else 0.0
            return self._act((cx, np.clip(1.2*ey, -0.1, 0.1), np.clip(1.5*eth, -0.1, 0.1)),
                             self.qhold, s)

        if self.phase == 2:      # arm above approach point
            p = np.array([self.obj_x+self.steer-self.gdx, b[1], CLEAR_Z])
            if self.phase_t % 5 == 1 or self.qtarget is None:
                qt = self._ik(s, p)
                if qt is not None:
                    self.qtarget = qt
            if (np.linalg.norm(self.tcp(s)-p) < 0.025 and self.phase_t > 10) or self.phase_t > 70:
                self.phase, self.phase_t, self.bias = 3, 0, np.zeros(3)
            return self._act((0, 0, 0), self.qtarget, s)

        if self.phase == 3:      # arm down to push height
            p = np.array([self.obj_x+self.steer-self.gdx, b[1], PUSH_Z])
            if self.phase_t % 5 == 1 or self.qtarget is None:
                qt = self._ik(s, p)
                if qt is not None:
                    self.qtarget = qt
            if (np.linalg.norm(self.tcp(s)-p) < 0.02 and self.phase_t > 6) or self.phase_t > 40:
                self.phase, self.phase_t = 4, 0
                self.qhold = self.q(s).copy()
                self.bowl0 = bowl.copy()
                self.push0 = o.copy()
            return self._act((0, 0, 0), self.qtarget, s)

        if self.phase == 4:      # push with base
            dy_goal = self._goal_dy(o, bowl)
            cur_dy = abs(o[1]-bowl[1])
            d = np.linalg.norm(o[:2]-bowl[:2])
            bowl_moved = np.linalg.norm(bowl[:2]-self.bowl0[:2]) > 0.02
            if cur_dy <= dy_goal + 0.004 or d < NEAR-0.02 or bowl_moved or \
               o[2] < 0.40 or self.phase_t > 85:
                drift = o[0] - self.push0[0]
                if abs(self.steer) < 0.005:
                    self.gdx = float(np.clip(self.gdx - 0.7*np.clip(drift, -0.06, 0.06), -0.02, 0.12))
                self.phase, self.phase_t, self.bias = 0, 0, np.zeros(3)
                self.qtarget = None
                return self._act((0, 0, 0), self.qhold, s)
            v = 0.014 if cur_dy - dy_goal > 0.06 else 0.008
            return self._act((0.0, sign*v, 0.0), self.qhold, s)

        return np.zeros(11, dtype=np.float32)
