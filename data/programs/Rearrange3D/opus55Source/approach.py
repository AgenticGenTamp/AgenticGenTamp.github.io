"""Pick-and-place approach for Rearrange3D: put boxed drink and can next to bowl."""
import numpy as np
import mujoco

ENV_ID = "kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0"

BOWL, DRINK, CAN = 0, 16, 32
BASE_I, ARM_I, GRIP_I = 93, 96, 103

_MODEL_CACHE = {}


def _get_model():
    if "m" not in _MODEL_CACHE:
        from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
        env = KinderGeom3DEnv(ENV_ID)
        env.reset(seed=0)
        re = env._kinder_env._object_centric_env._robot_env
        m = re.sim.model.mj_model
        d = mujoco.MjData(m)
        d.qpos[:] = re.sim.data.mj_data.qpos
        _MODEL_CACHE["m"] = m
        _MODEL_CACHE["d"] = d
        _MODEL_CACHE["env"] = env
    return _MODEL_CACHE["m"], _MODEL_CACHE["d"]


def _quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.m, self.d = _get_model()
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
        self.arm_q = [m.jnt_qposadr[jid(f"robot_joint_{i}")] for i in range(1, 8)]
        self.arm_v = [m.jnt_dofadr[jid(f"robot_joint_{i}")] for i in range(1, 8)]
        self.base_q = [m.jnt_qposadr[jid(n)] for n in
                       ("robot_joint_x", "robot_joint_y", "robot_joint_th")]
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))

    # ------------------------------------------------------------------ IK
    def _fk(self, base, q):
        d = self.d
        d.qpos[self.base_q] = base
        d.qpos[self.arm_q] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, base, pos, yaw, q0, iters=300):
        """Pinch site at pos, z-axis down, x-axis (finger axis) at yaw."""
        xd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        zd = np.array([0.0, 0.0, -1.0])
        yd = np.cross(zd, xd)
        q = np.array(q0, dtype=float).copy()
        for _ in range(iters):
            p, R = self._fk(base, q)
            ep = pos - p
            er = 0.5 * (np.cross(R[:, 0], xd) + np.cross(R[:, 1], yd) + np.cross(R[:, 2], zd))
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.arm_v], self.jr[:, self.arm_v]])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            n = np.abs(dq).max()
            if n > 0.3:
                dq *= 0.3 / n
            q += dq
            # joint limits for bounded joints (2,4,6)
            q[1] = np.clip(q[1], -2.2, 2.2)
            q[3] = np.clip(q[3], -2.5, 2.5)
            q[5] = np.clip(q[5], -2.05, 2.05)
        return q

    # ------------------------------------------------------------ helpers
    def _pinch(self, q=None, base=None):
        if q is None:
            q = self.obs[ARM_I:ARM_I + 7]
        if base is None:
            base = self.obs[BASE_I:BASE_I + 3]
        p, R = self._fk(base, q)
        return p, np.arctan2(R[1, 0], R[0, 0])

    def _bowl(self):
        return self.obs[BOWL:BOWL + 3]

    def _obj(self, o):
        return self.obs[o:o + 3].copy()

    def _half_h(self, o):
        return 0.5 * self.obs[o + 15]

    def _in_region(self, o):
        b = self._bowl()
        p = self._obj(o)
        return (abs(p[0] - b[0]) < 0.12 and abs(p[1] - b[1]) < 0.12
                and b[2] - 0.005 < p[2] < b[2] + 0.045)

    def _choose_spot(self, o):
        b = self._bowl()
        other = CAN if o == DRINK else DRINK
        op = self._obj(other)
        other_placed = self._in_region(other)
        if other_placed or other in self.spots:
            oth = op[:2] if other_placed else self.spots[other]
        else:
            oth = op[:2]
        pref = np.deg2rad(135.0) if o == DRINK else np.deg2rad(-135.0)
        best, bc = None, 1e9
        for r in (0.115, 0.11, 0.12):
            for ang in np.deg2rad(np.arange(0, 360, 10)):
                d = np.array([np.cos(ang), np.sin(ang)]) * r
                if max(abs(d[0]), abs(d[1])) > 0.115:
                    continue
                s = b[:2] + d
                cost = abs(_wrap(ang - pref))
                if s[0] < 0.20:
                    cost += 10 + (0.20 - s[0]) * 100
                if s[0] > 0.80 or abs(s[1]) > 0.9:
                    cost += 10
                dd = np.linalg.norm(s - oth)
                if dd < 0.10:
                    cost += 5 + (0.10 - dd) * 100
                cost += 0.5 * r
                if cost < bc:
                    bc, best = cost, s
        return best

    # ------------------------------------------------------------ planning
    def reset(self, state, info):
        self.obs = np.asarray(state, dtype=float)
        self.attempts = {BOWL: 0, DRINK: 0, CAN: 0}
        self.spots = {}
        self.phase = "idle"
        self.q_cmd = None
        self.q_target = None
        self.base_target = None
        self.grip = 0.0
        self.seg = None
        self._plan()
        self._start_next()

    def _rest_z(self, o):
        return 0.46 + self._half_h(o)

    def _stuck(self, o):
        return abs(self.obs[o + 2] - self._rest_z(o)) > 0.008

    def _plan(self):
        self.tasks = [DRINK, CAN]
        self.bowl_target = None
        band = [o for o in (DRINK, CAN) if self._stuck(o)
                and 0.472 < self.obs[o + 2] < 0.524 and not self._in_region(o)]
        if band:
            s = band[0]
            others = [o for o in (DRINK, CAN) if o != s]
            self.bowl_target = self._choose_bowl_spot(s, others)
            self.tasks = [BOWL] + others

    def _choose_bowl_spot(self, s, others):
        sp = self._obj(s)[:2]
        best, bc = None, 1e9
        for r in (0.115, 0.12):
            for ang in np.deg2rad(np.arange(0, 360, 10)):
                d = np.array([np.cos(ang), np.sin(ang)]) * r
                if max(abs(d[0]), abs(d[1])) > 0.118:
                    continue
                b = sp + d
                cost = 0.3 * np.linalg.norm(b - np.array([0.33, 0.0]))
                if b[0] < 0.22:
                    cost += 10 + (0.22 - b[0]) * 100
                if b[0] > 0.72:
                    cost += 10
                if abs(b[1]) > 0.30:
                    cost += 10 + (abs(b[1]) - 0.30) * 100
                for o in others:
                    dd = np.linalg.norm(b - self._obj(o)[:2])
                    if dd < 0.14:
                        cost += 5 + (0.14 - dd) * 100
                if cost < bc:
                    bc, best = cost, b
        return best

    def _start_next(self):
        self.cur_obj = None
        for o in self.tasks:
            p = self._obj(o)
            if o == BOWL:
                if self.attempts[o] >= 2 or np.linalg.norm(p[:2] - self.bowl_target) < 0.03:
                    continue
            elif (self._in_region(o) or self.attempts[o] >= 4 or p[2] < 0.43
                  or (self._stuck(o) and self.attempts[o] >= 1)):
                continue
            self.cur_obj = o
            break
        if self.cur_obj is None:
            self.phase = "idle"
            return
        o = self.cur_obj
        self.attempts[o] += 1
        p = self._obj(o)
        if o == BOWL:
            spot = self.bowl_target
        else:
            spot = self._choose_spot(o)
        self.spots[o] = spot
        by = np.clip(0.5 * (p[1] + spot[1]), -0.6, 0.6)
        self.base_target = np.array([-0.17, by, 0.0])
        if o == BOWL:
            dvec = np.array([self.base_target[0], self.base_target[1]]) - p[:2]
            dvec /= np.linalg.norm(dvec)
            self.grasp_xy_off = 0.058 * dvec
            yaw = np.arctan2(dvec[1], dvec[0])
            _, cur_yaw = self._pinch()
            k = np.round((cur_yaw - yaw) / np.pi)
            self.grasp_yaw = yaw + k * np.pi
        else:
            self.grasp_xy_off = np.zeros(2)
            self.grasp_yaw = self._grasp_yaw(o)
        self._set_phase("pre")

    def _grasp_yaw(self, o):
        _, cur_yaw = self._pinch()
        if o == DRINK:
            oy = _quat_yaw(self.obs[o + 3:o + 7])
            base_yaw = oy + np.pi / 2  # close along object's short (local y) axis
        else:
            base_yaw = cur_yaw
        k = np.round((cur_yaw - base_yaw) / np.pi)
        return base_yaw + k * np.pi

    def _start_seg(self, p1, yaw1, speed):
        q = self.obs[ARM_I:ARM_I + 7].copy()
        p0, yaw0 = self._pinch(q, self.base_target)
        if self.q_cmd is None:
            self.q_cmd = q
        dyaw = _wrap(yaw1 - yaw0)
        dist = np.linalg.norm(np.asarray(p1) - p0)
        n = max(1, int(np.ceil(max(dist / speed, abs(dyaw) / 0.08))))
        self.seg = dict(p0=p0, p1=np.asarray(p1, float), y0=yaw0, dy=dyaw, n=n, k=0)
        self.q_target = self.ik(self.base_target, self.seg["p1"], yaw0 + dyaw, self.q_cmd)

    def _seg_cmd(self):
        s = self.seg
        if s["k"] < s["n"]:
            s["k"] += 1
            f = s["k"] / s["n"]
            pt = s["p0"] + f * (s["p1"] - s["p0"])
            self.q_cmd = self.ik(self.base_target, pt, s["y0"] + f * s["dy"], self.q_cmd, iters=50)
        else:
            self.q_cmd = self.q_target
        return self.q_cmd

    def _seg_done(self, tol, stall=False):
        if self.seg["k"] < self.seg["n"]:
            return False
        err = np.abs(self.q_target - self.obs[ARM_I:ARM_I + 7]).max()
        vel = np.abs(self.obs[107:114]).max()
        if stall and self.t > self.seg["n"] + 6 and vel < 0.02 and err < 0.04:
            return True
        return err < tol and vel < 0.1

    def _set_phase(self, ph):
        self.phase = ph
        self.t = 0
        o = self.cur_obj
        p = self._obj(o)
        top = p[2] + self._half_h(o)
        if ph == "pre":
            self.grip = 0.0
            g = p[:2] + self.grasp_xy_off
            self._start_seg(np.array([g[0], g[1], top + 0.07]), self.grasp_yaw, 0.025)
        elif ph == "descend":
            self.grip = 0.0
            g = p[:2] + self.grasp_xy_off
            gz = top - 0.012 if o == BOWL else max(top - 0.04, 0.488)
            self._start_seg(np.array([g[0], g[1], gz]), self.grasp_yaw, 0.01)
        elif ph == "close":
            self.grip = 1.0
        elif ph == "lift":
            self.grip = 1.0
            pin, _ = self._pinch()
            self._start_seg(np.array([pin[0], pin[1], 0.72]), self.grasp_yaw, 0.015)
        elif ph == "move":
            spot = self.spots[o]
            pin, _ = self._pinch()
            off = pin[:2] - p[:2]
            self._start_seg(np.array([spot[0] + off[0], spot[1] + off[1], 0.72]),
                            self.grasp_yaw, 0.025)
        elif ph == "lower":
            spot = self.spots[o]
            pin, _ = self._pinch()
            off = pin - p
            z_rest = 0.46 + self._half_h(o) + 0.008
            self._start_seg(np.array([spot[0] + off[0], spot[1] + off[1], z_rest + off[2]]),
                            self.grasp_yaw, 0.01)
        elif ph == "open":
            self.grip = 0.0
        elif ph == "retract":
            self.grip = 0.0
            pin, yaw = self._pinch()
            self._start_seg(pin + [0, 0, 0.10], yaw, 0.015)

    def _holding(self):
        o = self.cur_obj
        p = self._obj(o)
        pin, _ = self._pinch()
        lim = 0.09 if o == BOWL else 0.05
        return np.linalg.norm(p[:2] - pin[:2]) < lim and p[2] > 0.46 + self._half_h(o) + 0.03

    def get_action(self, state):
        self.obs = np.asarray(state, dtype=float)
        if self.phase == "idle":
            self._start_next()
            if self.phase == "idle":
                a = np.zeros(11)
                return a.astype(np.float32)
        self.t += 1
        ph = self.phase
        if ph == "pre":
            if (self._seg_done(0.01) and self._base_err() < 0.01) or self.t > 150:
                self._set_phase("descend")
        elif ph == "descend":
            if self._seg_done(0.008, stall=True) or self.t > 80:
                self._set_phase("close")
        elif ph == "close":
            if self.t >= 6:
                self._set_phase("lift")
        elif ph == "lift":
            if self._seg_done(0.05) or self.t > 60:
                if self._holding():
                    self._set_phase("move")
                else:
                    self._set_phase("retract")
        elif ph == "move":
            if not self._holding() and self.t > 3:
                self._set_phase("retract")
            elif self._seg_done(0.01) or self.t > 150:
                self._set_phase("lower")
        elif ph == "lower":
            if self._seg_done(0.008, stall=True) or self.t > 80:
                self._set_phase("open")
        elif ph == "open":
            if self.t >= 5:
                self._set_phase("retract")
        elif ph == "retract":
            if self._seg_done(0.05) or self.t > 40:
                self._start_next()
        return self._action()

    def _base_err(self):
        b = self.obs[BASE_I:BASE_I + 3]
        e = self.base_target - b
        return np.abs(e[:2]).max()

    def _action(self):
        a = np.zeros(11)
        b = self.obs[BASE_I:BASE_I + 3]
        if self.base_target is not None:
            e = self.base_target - b
            e[2] = _wrap(e[2])
            a[:3] = np.clip(e, -0.1, 0.1)
        if self.seg is not None:
            qc = self._seg_cmd()
            a[3:10] = np.clip(qc - self.obs[ARM_I:ARM_I + 7], -0.1, 0.1)
        a[10] = self.grip
        return a.astype(np.float32)
