"""ScoopPour3D approach: grasp the source bin's far wall, lift it over the
target bin and invert it (pour everything at once)."""
import numpy as np
import mujoco

ARM_LO = np.array([-np.inf, -2.24, -np.inf, -2.57, -np.inf, -2.09, -np.inf])
ARM_HI = -ARM_LO


KP = np.array([200.0, 200, 200, 200, 100, 100, 100])


class Kin:
    def __init__(self):
        from kinder.envs.dynamic3d.task_families import ScoopPour3DEnv
        env = ScoopPour3DEnv(num_objects=10, scene_bg=False)
        env.reset(seed=0)
        re = env._object_centric_env._robot_env if hasattr(env, "_object_centric_env") else env._robot_env
        self.m = re.sim.model.mj_model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = re.sim.data.mj_data.qpos
        m = self.m
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.base_q = [m.jnt_qposadr[jid(n)] for n in ["robot_joint_x", "robot_joint_y", "robot_joint_th"]]
        self.arm_q = [m.jnt_qposadr[jid("robot_joint_%d" % i)] for i in range(1, 8)]
        self.arm_v = [m.jnt_dofadr[jid("robot_joint_%d" % i)] for i in range(1, 8)]
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        try:
            env.close()
        except Exception:
            pass

    def grav_delta(self, base, arm, point, mass):
        """Joint offsets (rad) giving PD torque to hold `mass` at world `point` (rigidly attached to hand)."""
        self.fk(base, arm)
        mujoco.mj_comPos(self.m, self.d)
        jp = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jp, None, np.asarray(point, dtype=float), int(self.m.site_bodyid[self.sid]))
        tau = jp[:, self.arm_v].T @ np.array([0.0, 0.0, 9.81 * mass])
        return tau / KP

    def fk(self, base, arm):
        d = self.d
        d.qpos[self.base_q] = base
        d.qpos[self.arm_q] = arm
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, base, arm0, pos, R=None, iters=60, rot_w=0.3, tol=1e-4, kns=0.1):
        q = np.array(arm0, dtype=float)
        e = np.zeros(3)
        for it in range(iters):
            p, Rc = self.fk(base, q)
            ep = pos - p
            if R is not None:
                er = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
                e = np.concatenate([ep, rot_w * er])
            else:
                e = ep
            if np.linalg.norm(e) < tol and it > 0:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            Jp = self.jacp[:, self.arm_v]
            J = np.vstack([Jp, rot_w * self.jacr[:, self.arm_v]]) if R is not None else Jp
            Jpinv = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(J.shape[0]))
            dq = Jpinv @ e
            if kns > 0:
                g = np.zeros(7)
                g[[1, 3, 5]] = -q[[1, 3, 5]]
                dq += (np.eye(7) - Jpinv @ J) @ (kns * g)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, ARM_LO, ARM_HI)
        return q, float(np.linalg.norm(e))


def down_R(yaw):
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0, 0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def axis_R(axis, a):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def quat_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


class GeneratedApproach:
    GX_PREF = -0.03
    BX = -0.16          # base x while working (island blocks x > -0.06)
    KI = 0.15
    FF_GAIN = 1.0
    BY_OFF = -0.05
    BASE_LIM_LOADED = 0.04
    NSHAKE = 2
    POUR_SHIFT = 0.0
    PHI_END = np.radians(165)
    ZC = 0.72           # height of bin centre while pouring
    GRIP_DZ = 0.056     # grip height above bin base (wall top is 0.07)
    WALL_OFF = 0.14     # wall centre distance from bin centre

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.kin = Kin()

    # ---------------- state helpers
    def _robot(self, s):
        r = s.get_object_from_name("robot")
        base = np.array([s.get(r, f) for f in ["pos_base_x", "pos_base_y", "pos_base_rot"]])
        arm = np.array([s.get(r, "pos_arm_joint%d" % i) for i in range(1, 8)])
        return base, arm

    def _pose(self, s, o):
        return np.array([s.get(o, f) for f in ["x", "y", "z", "qw", "qx", "qy", "qz"]])

    def reset(self, state, info):
        s = state
        bins, cubes = [], []
        for o in s.get_objects(s.get_object_from_name("robot").type.__class__) if False else []:
            pass
        for name in s.get_object_names():
            o = s.get_object_from_name(name)
            if name.startswith("bin"):
                bins.append(o)
            elif name.startswith("cube"):
                cubes.append(o)
        self.cubes = cubes
        cen = np.mean([self._pose(s, c)[:3] for c in cubes], axis=0) if cubes else np.zeros(3)
        bins.sort(key=lambda b: np.linalg.norm(self._pose(s, b)[:2] - cen[:2]))
        green = [b for b in bins if "green" in b.name]
        self.src = bins[0]
        self.tgt = green[0] if green and green[0] is not self.src else bins[-1]
        ps, pt = self._pose(s, self.src), self._pose(s, self.tgt)
        d = pt[:2] - ps[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        self.dir = np.array([d[0], d[1], 0.0])
        self.dyaw = np.arctan2(d[1], d[0])
        # pour rotation axis: rotating by negative angle tips the +dir side down
        self.axis = np.array([d[1], -d[0], 0.0])  # for d=+y -> +x
        self.R0 = down_R(self.dyaw)
        self.src_pos0 = ps[:3]
        self.tgt_pos0 = pt[:3]
        Rt = quat_R(*pt[3:])
        self.tgt_yaw = np.arctan2(Rt[1, 0], Rt[0, 0])
        self.phase = "approach"
        self.t = 0
        self.pt = 0
        self.phi = 0.0
        self.wps = None
        self.I = np.zeros(7)
        self.regrasps = 0
        self.payload = None

    # ---------------- control
    def _base_for(self, grip):
        # keep base at fixed x, y a little behind the grip (toward -dir)
        return np.array([self.BX, grip[1] - self.BY_OFF * np.sign(self.dir[1] if abs(self.dir[1]) > 0.1 else 1.0), 0.0])

    def _cmd(self, s, bt, pos, R, g):
        base, arm = self._robot(s)
        q, err = self.kin.ik(base, arm, pos, R)
        a = np.zeros(11, dtype=np.float32)
        db = bt - base
        db[2] = (db[2] + np.pi) % (2 * np.pi) - np.pi
        bl = self.BASE_LIM_LOADED if g > 0.5 else 0.1
        a[:3] = np.clip(db, -bl, bl)
        err = q - arm
        self.I = np.clip(self.I + self.KI * err, -0.15, 0.15)
        ff = np.zeros(7)
        if g > 0.5 and self.payload is not None:
            ff = self.kin.grav_delta(base, arm, self.payload[0], self.payload[1])
        a[3:10] = np.clip(err + self.I + ff, -0.1, 0.1)
        a[10] = g
        p, _ = self.kin.fk(base, arm)
        return a, np.linalg.norm(p - pos), np.abs(db).max()

    def _plan(self, s):
        """Waypoints: (kind, pos, R, grip, pos_tol, rot_tol, max_steps).
        kind 'ee': pos/R are the end-effector target.
        kind 'bin': pos/R are the desired source-bin centre/orientation."""
        ps = self._pose(s, self.src)
        Rb = quat_R(*ps[3:])
        # slide the grasp along the wall to avoid pinching cubes near it
        nl = -(Rb.T @ self.dir); nl[2] = 0; nl /= np.linalg.norm(nl) + 1e-9
        grip_local = np.array([0.0, 0.0, self.GRIP_DZ]) + self.WALL_OFF * nl
        tilted = Rb[2, 2] < 0.95
        Rg = Rb @ down_R(np.arctan2(-nl[1], -nl[0])) if tilted else self.R0
        up = Rb[:, 2] if tilted else np.array([0, 0, 1.0])
        tl = np.array([-nl[1], nl[0], 0.0])
        cl = [Rb.T @ (self._pose(s, c)[:3] - ps[:3]) for c in self.cubes]
        near = [c for c in cl if c @ nl > self.WALL_OFF - 0.06]
        best, bs = 0.0, -1e9
        com = sum(c @ tl for c in cl) * 0.1 / (0.1 + 0.1 * len(cl))  # bin mass ~ one cube
        pref = float(np.clip(0.5 * com + self.GX_PREF, -0.1, 0.06))
        for off in np.linspace(-0.12, 0.12, 25):
            dmin = min([abs(c @ tl - off) for c in near], default=1.0)
            sc = min(dmin, 0.05) - 0.05 * abs(off - pref)
            if sc > bs:
                best, bs = off, sc
        grip_local = grip_local + best * tl
        gw = ps[:3] + Rb @ grip_local
        wps = []
        add = lambda *w: wps.append(w)
        add("ee", gw + 0.06 * up, Rg, 0.0, 0.03, 0.1, 70)
        add("ee", gw, Rg, 0.0, 0.011, 0.1, 40)
        add("ee", gw, Rg, 1.0, -1, -1, 4)
        Ryaw_s = axis_R(np.array([0, 0, 1.0]), np.arctan2(Rb[1, 0], Rb[0, 0]))
        pt = self.tgt_pos0
        Ryaw_t = axis_R(np.array([0, 0, 1.0]), self.tgt_yaw)
        c0 = np.array([ps[0], ps[1], self.ZC])
        c1 = np.array([pt[0], pt[1], self.ZC]) - self.POUR_SHIFT * self.dir
        phi_cur = 0.0
        if Rb[2, 2] < 0.95 and np.linalg.norm(ps[:2] - c1[:2]) < 0.15:
            # bin already tilted over the target (dropped mid-pour): resume pouring
            grid = np.radians(np.arange(0, 181, 3))
            dots = [axis_R(self.axis, -p)[:, 2] @ Rb[:, 2] for p in grid]
            phi_cur = float(grid[int(np.argmax(dots))])
            add("bin", c1, axis_R(self.axis, -phi_cur) @ Ryaw_t, 1.0, 0.04, 0.2, 30)
        else:
            for z in [ps[2] + 0.035 + 0.1, self.ZC]:
                add("bin", np.array([ps[0], ps[1], z]), Ryaw_s, 1.0, 0.03, 0.15, 30)
            for f in np.linspace(0.1, 1.0, 10):
                add("bin", c0 + f * (c1 - c0), Ryaw_t, 1.0, 0.03, 0.1, 15)
        phis = np.concatenate([np.radians(np.arange(6, 36, 6)), np.radians(np.arange(37, 66, 1.5)),
                               np.linspace(np.radians(70), self.PHI_END, 10)])
        for phi in phis[phis > phi_cur + 0.02]:
            slow = np.radians(36) < phi < np.radians(66)
            early = phi < np.radians(36)
            add("bin", c1, axis_R(self.axis, -phi) @ Ryaw_t, 1.0, 0.05 if early else 0.03, 0.05 if slow else (0.12 if early else 0.08), 12 if early else 25)
        for k in range(self.NSHAKE):
            for dp in [0.35, -0.12]:
                add("bin", c1 + np.array([0, 0, 0.03 if dp > 0 else 0.0]), axis_R(self.axis, -self.PHI_END + dp) @ Ryaw_t, 1.0, 0.03, 0.1, 8)
        self.wps = wps
        self.G0 = None
        self.wi = 0
        self.G = None

    def get_action(self, state):
        s = state
        self.t += 1
        self.pt += 1
        if self.wps is None:
            self._plan(s)
        wi = min(self.wi, len(self.wps) - 1)
        kind, pos, R, g, ptol, rtol, mx = self.wps[wi]
        base, arm = self._robot(s)
        pe, Re = self.kin.fk(base, arm)
        ps = self._pose(s, self.src)
        Rb = quat_R(*ps[3:])
        cb = ps[:3] + Rb @ np.array([0, 0, 0.035])
        if g > 0.5:
            # grasp transform (ee in bin-centre frame), low-pass filtered
            Gr, Gp = Rb.T @ Re, Rb.T @ (pe - cb)
            if self.G is None:
                self.G = (Gr, Gp)
            else:
                M = 0.7 * self.G[0] + 0.3 * Gr
                U, _, Vt = np.linalg.svd(M)
                self.G = (U @ Vt, 0.7 * self.G[1] + 0.3 * Gp)
        if g > 0.5 and self.wi >= 3 and self.G is not None:
            if self.G0 is None:
                self.G0 = self.G[1].copy()
            elif self.wi >= 4 and self.regrasps < 3 and np.linalg.norm(Gp - self.G0) > 0.06:
                # grasp lost: open and re-plan a fresh grasp of the source bin
                self.regrasps += 1
                self.wps = None
                self.I[:] = 0
                a = np.zeros(11, dtype=np.float32)
                return a
        self.payload = None
        if g > 0.5 and self.wi >= 3:
            ins, pts = 0, [cb * 0.1]
            for c in self.cubes:
                p = self._pose(s, c)[:3]
                l = Rb.T @ (p - ps[:3])
                if abs(l[0]) < 0.23 and abs(l[1]) < 0.16 and -0.01 < l[2] < 0.09:
                    ins += 1
                    pts.append(p * 0.1)
            m = 0.1 + 0.1 * ins
            self.payload = (np.sum(pts, axis=0) / m, m * self.FF_GAIN)
        if kind == "bin":
            Gr, Gp = self.G
            eR, ep = R @ Gr, pos + R @ Gp
            e = np.linalg.norm(cb - pos)
            re = np.arccos(np.clip((np.trace(Rb.T @ R) - 1) / 2, -1, 1))
        else:
            eR, ep = R, pos
            e = np.linalg.norm(pe - pos)
            re = np.arccos(np.clip((np.trace(Re.T @ R) - 1) / 2, -1, 1))
        ep = np.clip(ep, [0.0, -0.75, 0.47], [0.9, 0.75, 1.15])
        bt = self._base_for(ep)
        a, _, eb = self._cmd(s, bt, ep, eR, g)
        self.last_err = (e, re)
        if self.wi < len(self.wps) and ((e < ptol and re < rtol and eb < 0.02) or self.pt >= mx):
            self.wi += 1
            self.pt = 0
        self.phase = "wp%d" % self.wi
        return a
