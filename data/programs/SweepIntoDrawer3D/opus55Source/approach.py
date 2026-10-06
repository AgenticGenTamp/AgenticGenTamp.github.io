"""Approach for kinder/SweepIntoDrawer3D: open island drawer s1c1, then sweep
cubes off the counter edge into it with the wiper tool."""
import numpy as np
import mujoco

# ---------------------------------------------------------------- kinematics
ARM_QADR = 55
ARM_DADR = 49
BASE_QADR = 52
LIM = np.array([[-1e9, 1e9], [-2.2, 2.2], [-1e9, 1e9], [-2.5, 2.5],
                [-1e9, 1e9], [-2.05, 2.05], [-1e9, 1e9]])
Q_HOME = np.array([0, 0.26, np.pi, -2.27, 0, 0.96, np.pi / 2])
Q_RETRACT = np.array([0, -0.349, 3.1416, -2.548, 0, -0.873, 1.5708])

_MODEL_CACHE = {}


def _get_model():
    if 'm' not in _MODEL_CACHE:
        from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
        env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
        env.reset(seed=0)
        re = env._kinder_env._object_centric_env._robot_env
        _MODEL_CACHE['env'] = env
        _MODEL_CACHE['m'] = re.sim.model.mj_model
    return _MODEL_CACHE['m']


def rot_err(R, Rt):
    return 0.5 * (np.cross(R[:, 0], Rt[:, 0]) + np.cross(R[:, 1], Rt[:, 1])
                  + np.cross(R[:, 2], Rt[:, 2]))


def R_front(psi):
    """Approach toward -x pitched down by psi; fingers close in x-z plane."""
    z = np.array([-np.cos(psi), 0, -np.sin(psi)])
    y = np.array([0, 1.0, 0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def R_down(tilt=0.0, fingers_along_y=True):
    """Gripper approach pointing down, tilted so that the gripper top leans
    toward +x by `tilt` rad. Fingers close along world y (or x)."""
    z = np.array([-np.sin(tilt), 0, -np.cos(tilt)])
    if fingers_along_y:
        x = np.array([0, 1.0, 0])
    else:
        x = np.array([np.cos(tilt), 0, -np.sin(tilt)])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def R_grasp(tilt, phi):
    """Approach mostly downward; fingers close along horizontal direction
    phi. The approach is tilted (by up to `tilt`) only about the closing
    axis, so both fingertips stay at the same height."""
    x = np.array([np.cos(phi), np.sin(phi), 0.0])
    u = np.array([-x[1], x[0], 0.0])
    if u[0] < 0:
        u = -u
    t = tilt * u[0]
    z = -np.sin(t) * u + np.array([0.0, 0.0, -np.cos(t)])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def R_hook(a):
    """Approach pointing down and toward -x (a rad from vertical); fingers
    close along world y so the thin finger dimension lies along x."""
    z = np.array([-np.sin(a), 0.0, -np.cos(a)])
    x = np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


class Kin:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                     'robot_pinch_site')
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def fk(self, base, q):
        d = self.d
        d.qpos[BASE_QADR:BASE_QADR + 3] = base
        d.qpos[ARM_QADR:ARM_QADR + 7] = q
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, base, q0, pt, Rt, iters=60, wrot=0.5, tol=2e-4):
        q = np.array(q0, dtype=float)
        ep = er = np.zeros(3)
        for _ in range(iters):
            p, R = self.fk(base, q)
            ep = pt - p
            er = rot_err(R, Rt)
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, ARM_DADR:ARM_DADR + 7],
                           wrot * self.jr[:, ARM_DADR:ARM_DADR + 7]])
            e = np.concatenate([ep, wrot * er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            dq = np.clip(dq, -0.3, 0.3)
            q = np.clip(q + dq, LIM[:, 0], LIM[:, 1])
        ok = np.linalg.norm(ep) < 3e-3 and np.linalg.norm(er) < 2e-2
        return q, ok

    def ik_multi(self, base, q_seed, pt, Rt, n_random=20):
        best = None
        rng = np.random.default_rng(0)
        seeds = [q_seed, Q_HOME, Q_RETRACT] + [
            rng.uniform(-np.pi, np.pi, 7) * np.array([1, .7, 1, .8, 1, .65, 1])
            for _ in range(n_random)]
        for k, s in enumerate(seeds):
            q, ok = self.ik(base, s, pt, Rt, iters=60 if k == 0 else 100)
            if ok:
                for j in (0, 2, 4, 6):
                    q[j] = q_seed[j] + ((q[j] - q_seed[j] + np.pi) % (2 * np.pi) - np.pi)
                cost = np.sum((q - q_seed) ** 2)
                if best is None or cost < best[1]:
                    best = (q, cost)
                if k == 0:
                    break
                if k >= 2 and best is not None and k >= 6:
                    break
        return None if best is None else best[0]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- approach
class GeneratedApproach:
    DRAWER_OPEN = 0.12
    BX_MIN = 1.38
    R_GRASP = 0.62
    TILT = 0.2
    G_OFF = 0.10
    BB_TARGET = 0.4615
    GTILT = 0.2
    Z_PRE = 0.55
    Z_GRASP = 0.497
    PRE_GRIP = 0.3
    HOOK_A = 0.5
    Z_SAFE = 0.58
    HOOK_BASE = (1.42, -0.33)
    BACK_D = 0.10
    ROT_Z = 0.485
    N_EVAL = 8
    YAW_TOL = 0.15
    FIRST_FROM_CUR = True
    BACK_N = 15
    RISE_N = 8
    BACK_VTOL = 10.0
    HI_TOL = 0.03
    HI_VTOL = 0.1
    HOOK_X = 0.912
    HOOK_Z = 0.322
    APPR_SPEED = 0.01
    HGRIP_WAIT = 6
    PULL_STEP = 0.03
    R_PICK = 0.70
    CARRY_STEP = 0.1
    X_DROP = 0.935
    GRIP_WAIT = 1
    REL_WAIT = 1
    STROKE_STEP = 0.008

    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.kin = Kin(_get_model())
        self.obs = None
        self.gen = None

    # ---- observation helpers
    def base(self):
        return self.obs[125:128].copy()

    def q(self):
        return self.obs[128:135].copy()

    def qvel(self):
        return self.obs[139:146].copy()

    def cubes(self):
        return np.array([self.obs[16 * i:16 * i + 3] for i in range(5)])

    def wiper(self):
        return self.obs[147:150].copy()

    def drawer(self):
        return float(self.obs[107])

    def ee(self):
        return self.kin.fk(self.base(), self.q())

    # ---- interface
    def reset(self, state, info=None):
        self.obs = np.asarray(state, dtype=float)
        self.gen = self.plan()
        self.drawer_done = False
        self.misses = []
        self.t = 0
        self.grip = 0.0
        self.log = []

    def get_action(self, state):
        self.obs = np.asarray(state, dtype=float)
        self.t += 1
        try:
            a = next(self.gen)
        except StopIteration:
            a = self.hold()
        return np.asarray(a, dtype=np.float32)

    # ---- low-level action makers
    def hold(self):
        a = np.zeros(11)
        a[10] = self.grip
        return a

    def act(self, base_t=None, q_t=None, bstep=0.1, qstep=0.1):
        a = np.zeros(11)
        if base_t is not None:
            b = self.base()
            if getattr(self, 'drawer_done', False):
                base_t = np.array(base_t, dtype=float)
                base_t[0] = max(base_t[0], self.bx_min())
            e = np.array([base_t[0] - b[0], base_t[1] - b[1], wrap(base_t[2] - b[2])])
            a[0:3] = np.clip(e, -bstep, bstep)
        if q_t is not None:
            a[3:10] = np.clip(q_t - self.q(), -qstep, qstep)
        a[10] = self.grip
        return a

    # ---- motion primitives (generators)
    def wait(self, n, base_t=None, q_t=None):
        for _ in range(n):
            yield self.act(base_t, q_t)

    def joint_move(self, q_t, base_t=None, max_steps=150, tol=0.02, vtol=0.1):
        for _ in range(max_steps):
            q = self.q()
            berr = 0.0
            if base_t is not None:
                b = self.base()
                berr = max(abs(base_t[0] - b[0]), abs(base_t[1] - b[1]),
                           abs(wrap(base_t[2] - b[2])))
            if np.max(np.abs(q_t - q)) < tol and berr < 0.005 \
                    and np.max(np.abs(self.qvel())) < vtol:
                return
            yield self.act(base_t, q_t)

    def pose_move(self, pos, R, base_t, max_steps=150, tol=0.02):
        b = base_t if base_t is not None else self.base()
        q_t = self.kin.ik_multi(np.asarray(b), self.q(), np.asarray(pos), R)
        if q_t is None:
            self.log.append(('ik_fail', pos))
            return
        yield from self.joint_move(q_t, base_t, max_steps, tol)

    def cart_move(self, pos, R, speed=0.01, max_steps=200, tol=0.004, base_t=None, vtol=0.1):
        """Straight-line move of the pinch site with fixed base."""
        pos = np.asarray(pos, dtype=float)
        b = self.base() if base_t is None else np.asarray(base_t)
        ref, _ = self.kin.fk(b, self.q())
        q_ref = self.q()
        for _ in range(max_steps):
            dp = pos - ref
            n = np.linalg.norm(dp)
            if n > speed:
                dp = dp * speed / n
            ref = ref + dp
            q_new, ok = self.kin.ik(b, q_ref, ref, R, iters=30)
            if ok:
                q_ref = q_new
            p, _ = self.kin.fk(self.base(), self.q())
            if n < 1e-6 and np.linalg.norm(p - pos) < tol \
                    and np.max(np.abs(self.qvel())) < vtol:
                return
            yield self.act(base_t, q_ref)

    # ---- task plan
    def plan(self):
        yield from self.open_drawer()
        self.drawer_done = True
        self.log.append(('t_open', self.t))
        # transit to first pick
        q_canon = None
        for seed in (self.q(), Q_HOME):
            q_canon = self.kin.ik_multi(np.array([1.5, 0.0, np.pi]), seed, np.array([0.8, 0.0, self.Z_PRE]),
                                        R_grasp(self.GTILT, np.pi / 2))
            if q_canon is not None:
                break
        if q_canon is None:
            q_canon = Q_HOME
        if self.FIRST_FROM_CUR:
            nxt = self.prep_pick(self.remaining(), q_from=self.q())
        else:
            nxt = self.prep_pick(self.remaining(), q_from=q_canon)
        if nxt is not None:
            # keep base x until the hand is above counter height
            c = self.cubes()[nxt[0]]
            b0 = self.base().copy()
            for _ in range(40):
                p = self.ee()[0]
                if p[2] > 0.50:
                    break
                yield self.act(np.array([b0[0], c[1] + 0.0, np.pi]), nxt[3])
        self.nxt = nxt
        self.log.append(('t_tr', self.t))
        for _ in range(12):
            rem = self.remaining()
            if len(rem) == 0:
                break
            self.log.append(('t', self.t))
            nxt = self.nxt if (self.nxt is not None and self.nxt[0] in rem) else self.prep_pick(rem)
            self.nxt = None
            if nxt is None:
                break
            yield from self.pick_place(*nxt)
        while True:
            yield self.hold()

    def phi_cur(self, q=None):
        R = self.kin.fk(np.array([1.5, 0.0, np.pi]), self.q() if q is None else q)[1]
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def prep_pick(self, rem, q_from=None):
        """Choose next cube + finger direction; returns (i, phi, r, q_hi, q_lo)."""
        if not rem:
            return None
        q_from = self.q() if q_from is None else q_from
        cands = self.choose_pick(rem, self.phi_cur(q_from))
        cands.sort(key=lambda x: x[0] + max(x[3], x[4]))
        best = None
        for (cs, i, phi, yt, pt) in cands[:self.N_EVAL]:
            if best is not None and cs + yt >= best[0]:
                continue
            c = self.cubes()[i]
            r = max(self.R_PICK, self.bx_min() - c[0])
            q = self.pick_cfg(r, self.Z_PRE, phi, q_from)
            if q is None:
                continue
            at = np.max(np.abs(q - q_from)) / 0.025
            tot = cs + max(at, yt)
            if best is None or tot < best[0]:
                best = (tot, i, phi, r, q)
        if best is None:
            self.log.append(('ik_fail_pre',))
            return None
        _, i, phi, r, q_hi = best
        q_lo = self.pick_cfg(r, self.Z_GRASP - 0.015, phi, q_hi)
        if q_lo is None:
            q_lo = q_hi
        return (i, phi, r, q_hi, q_lo)

    def cube_yaw(self, i):
        qw, qx, qy, qz = self.obs[16 * i + 3:16 * i + 7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def choose_pick(self, rem, phi_c):
        c = self.cubes()
        b = self.base()
        cands = []
        for i in rem:
            yaw = self.cube_yaw(i)
            d = (yaw - np.pi / 2 + np.pi / 4) % (np.pi / 2) - np.pi / 4
            for phi in (np.pi / 2 + d, d):
                # gripper is symmetric under pi: pick equivalent closest to current
                phi = phi_c + ((phi - phi_c + np.pi / 2) % np.pi - np.pi / 2)
                h = np.array([np.cos(phi), np.sin(phi)])
                hp = np.array([-h[1], h[0]])
                cost = 0.0
                for j in rem:
                    if j == i:
                        continue
                    cost += self.finger_cost(c[j, :2] - c[i, :2], self.cube_yaw(j) - phi, h, hp)
                for (mi, mphi) in self.misses:
                    if mi == i and abs(wrap(2 * (mphi - phi))) < 0.4:
                        cost += 300.0
                # time-like costs (steps)
                cost += (0.85 - c[i, 0]) * 20
                yt = abs(c[i, 1] - b[1]) / 0.05
                for ph in (phi, phi + np.pi):
                    ph = ph + np.clip(wrap(phi_c - ph), -self.YAW_TOL, self.YAW_TOL)
                    pt = abs(wrap(ph - phi_c)) / 0.025
                    cands.append((cost, i, ph, yt, pt))
        return cands

    @staticmethod
    def finger_cost(dv, yrel, h, hp):
        ax, ay = dv @ h, dv @ hp
        if np.hypot(ax, ay) > 0.1:
            return 0.0
        ex = 0.01 * (abs(np.cos(yrel)) + abs(np.sin(yrel)))
        xlo, xhi = abs(ax) - ex, abs(ax) + ex
        ylo, yhi = ay - ex, ay + ex
        # pads + followers region (one side, |x| in [0.019, 0.068], |y| <= 0.013)
        px = min(xhi, 0.068) - max(xlo, 0.019)
        py = min(yhi, 0.013) - max(ylo, -0.013)
        if px > 0 and py > 0:
            return 400.0 + 10000.0 * min(px, py)
        gap = max(-px, -py)
        cost = 0.0
        if gap < 0.004:
            cost += 40.0 * (0.004 - gap) / 0.004 + 10.0
        if np.hypot(ax, ay) < 0.025:
            cost += 20.0
        return cost

    def pick_cfg(self, r, z, phi, seed):
        b = np.array([1.5, 0.0, np.pi])
        return self.kin.ik_multi(b, seed, np.array([1.5 - r, 0.0, z]), R_grasp(self.GTILT, phi))

    def pick_place(self, i, phi, r, q_hi, q_lo):
        c = self.cubes()[i]
        self.grip = self.PRE_GRIP
        base_t = np.array([c[0] + r, c[1], np.pi])
        dq0 = np.round(q_hi - self.q(), 2)
        yield from self.joint_move(q_hi, base_t, max_steps=120, tol=self.HI_TOL, vtol=self.HI_VTOL)
        self.log.append(('hi', self.t, tuple(dq0)))
        c = self.cubes()[i]
        base_t = np.array([c[0] + r, c[1], np.pi])
        q_g = q_lo
        k = -1
        for k in range(40):
            p = self.ee()[0]
            c = self.cubes()[i]
            e = c[:2] - p[:2]
            base_t[:2] += np.clip(0.6 * e, -0.01, 0.01)
            if p[2] < self.Z_GRASP + 0.002 and np.linalg.norm(e) < 0.003:
                q_g = self.q()
                break
            yield self.act(base_t, q_lo)
        p = self.ee()[0]
        c = self.cubes()[i]
        dbg = (k, round(float(p[2]), 4), round(float(np.linalg.norm(c[:2] - p[:2])), 4),
               round(float(wrap(4 * (self.cube_yaw(i) - phi)) / 4), 3),
               round(float(wrap(4 * (self.phi_cur(self.q()) - phi)) / 4), 3),
               [tuple(np.round([np.cos(phi) * (cc[0] - c[0]) + np.sin(phi) * (cc[1] - c[1]),
                                -np.sin(phi) * (cc[0] - c[0]) + np.cos(phi) * (cc[1] - c[1])], 3))
                for jj, cc in enumerate(self.cubes()) if jj != i and cc[2] > 0.44
                and np.linalg.norm(cc[:2] - c[:2]) < 0.12])
        self.log.append(('lo', self.t))
        self.grip = 1.0
        yield from self.wait(self.GRIP_WAIT, base_t, q_g)
        # plan next pick now so the wrist can pre-rotate during lift + carry
        rem = [j for j in self.remaining() if j != i]
        nxt = self.prep_pick(rem, q_from=q_hi)
        q_carry = q_hi
        if nxt is not None:
            q_carry = q_hi.copy()
            q_carry[6] = nxt[3][6]
        # lift
        for _ in range(25):
            cz = self.cubes()[i][2]
            if cz > 0.50:
                break
            yield self.act(base_t, q_carry if cz > self.ROT_Z else q_hi)
        c = self.cubes()[i]
        if c[2] < 0.49:   # grasp failed; release and let the planner retry
            self.grip = 0.0
            yield from self.wait(3, base_t, q_hi)
            self.log.append(('miss', i, dbg, round(float(c[2]), 3)))
            self.misses.append((i, phi))
            return
        self.log.append(('up', self.t))
        # track whichever cube is actually in the hand (stacked cubes)
        ee = self.ee()[0]
        cs = self.cubes()
        held = [j for j in range(5) if cs[j, 2] > 0.49 and np.linalg.norm(cs[j] - ee) < 0.06]
        if held and i not in held:
            i = min(held, key=lambda j: np.linalg.norm(cs[j] - ee))
            self.log.append(('held', i))
        c = self.cubes()[i]
        ydrop = float(np.clip(c[1], -0.2, 0.2))
        for _ in range(40):
            c = self.cubes()[i]
            if c[2] < 0.46:   # dropped (maybe already in the drawer)
                break
            if np.linalg.norm(c - self.ee()[0]) > 0.08:   # lost it
                break
            b = self.base()
            bt = b + np.array([self.X_DROP - c[0], ydrop - c[1], 0.0])
            bt[2] = np.pi
            if abs(c[0] - self.X_DROP) < 0.012 and abs(c[1] - ydrop) < 0.02:
                break
            yield self.act(bt, q_carry, bstep=self.CARRY_STEP)
        self.grip = 0.0
        yield from self.wait(self.REL_WAIT, self.base(), q_carry)
        self.nxt = nxt
        self.log.append(('pp', i, self.t))

    def bx_min(self):
        return max(self.BX_MIN, 0.935 + self.drawer() + 0.29 + 0.025)

    def remaining(self):
        c = self.cubes()
        return [i for i in range(5) if c[i, 2] > 0.44]

    def base_move_hold(self, base_t, q_hold, max_steps=120, bstep=0.04):
        """Move base (holonomic) while holding arm joints."""
        for _ in range(max_steps):
            b = self.base()
            err = max(abs(base_t[0] - b[0]), abs(base_t[1] - b[1]),
                      abs(wrap(base_t[2] - b[2])))
            if err < 0.004 and np.max(np.abs(q_hold - self.q())) < 0.02:
                return
            yield self.act(base_t, q_hold, bstep=bstep)

    def rel_config(self, r, z, lat=0.0, seed=None):
        """Arm config placing pinch site at base-relative forward reach r,
        lateral offset lat, world height z (base yaw pi)."""
        b = np.array([1.5, 0.0, np.pi])
        p = np.array([1.5 - r, -lat, z])
        q0 = self.q() if seed is None else seed
        return self.kin.ik_multi(b, q0, p, R_down(self.TILT))

    def grasp_wiper(self):
        w = self.wiper()
        gx = w[0] + self.G_OFF
        r = self.R_GRASP
        bx = gx + r
        if bx < self.BX_MIN:
            bx = self.BX_MIN
            r = bx - gx
        base_t = np.array([bx, w[1], np.pi])
        q_up = self.rel_config(r, 0.56)
        self.grip = 0.0
        yield from self.joint_move(q_up, base_t, max_steps=200)
        R = R_down(self.TILT)
        g = np.array([gx, w[1], 0.500])
        yield from self.cart_move(g, R, speed=0.006, base_t=base_t, max_steps=60)
        self.grip = 1.0
        yield from self.wait(10, base_t, self.q())
        p, _ = self.ee()
        self.w_off = self.wiper() - p   # wiper body minus pinch site
        self.wq_grasp = self.obs[150:154].copy()
        yield from self.cart_move(p + [0, 0, 0.05], R, speed=0.01, base_t=base_t, max_steps=40)

    def blade_bottom(self):
        w = self.obs[147:150]
        qw, qx, qy, qz = self.obs[150:154]
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        zs = [w[2] + R[2] @ np.array([x, y, 0.0]) for x in (-0.115, -0.085) for y in (-0.1, 0.1)]
        return float(min(zs))

    def rel_q(self, r, z, q0):
        b = np.array([1.5, 0.0, np.pi])
        q, ok = self.kin.ik(b, q0, np.array([1.5 - r, 0.0, z]), R_down(self.TILT), iters=30)
        return q if ok else None

    def sweep_pass(self, rem):
        c = self.cubes()[rem]
        R = R_down(self.TILT)
        ys = np.sort(c[:, 1])
        lo = ys[0]
        sel = [i for i, ci in zip(rem, c) if ci[1] <= lo + 0.165]
        cs = self.cubes()[sel]
        yc = 0.5 * (cs[:, 1].min() + cs[:, 1].max())
        yc = float(np.clip(yc, -0.2, 0.2))
        blade_x0 = min(cs[:, 0].min() - 0.045, 0.80)
        # current blade bottom relative to the site
        p, _ = self.ee()
        bb_off = self.blade_bottom() - p[2]
        z_dn = self.BB_TARGET - bb_off
        z_up = z_dn + 0.05
        site_x0 = blade_x0 + 0.10 - self.w_off[0]
        site_y = yc - self.w_off[1]
        bx = max(self.BX_MIN, site_x0 + self.R_GRASP)
        r = bx - site_x0
        base_t = np.array([bx, site_y, np.pi])
        q_up = self.rel_config(r, z_up)
        if q_up is None:
            self.log.append(('ik_fail_sweep', r))
            return
        yield from self.joint_move(q_up, base_t, max_steps=150)
        yield from self.cart_move([site_x0, site_y, z_dn], R, speed=0.006, base_t=base_t, max_steps=40)
        # closed-loop lowering until blade touches counter
        z_cmd = z_dn
        q_cmd = self.rel_config(r, z_cmd, seed=q_up)
        if q_cmd is None:
            q_cmd = self.q()
        for _ in range(20):
            e = self.blade_bottom() - self.BB_TARGET
            if abs(e) < 0.002:
                break
            z_cmd -= 0.7 * e
            qn = self.rel_q(r, z_cmd, q_cmd)
            if qn is not None:
                q_cmd = qn
            yield self.act(base_t, q_cmd)
        # stroke: drive base forward, keep regulating blade height
        stroke = 0.905 - blade_x0
        bt = base_t + np.array([stroke, 0, 0])
        for _ in range(90):
            b = self.base()
            if abs(bt[0] - b[0]) < 0.004:
                break
            e = self.blade_bottom() - self.BB_TARGET
            z_cmd -= 0.3 * e
            qn = self.rel_q(r, z_cmd, q_cmd)
            if qn is not None:
                q_cmd = qn
            yield self.act(bt, q_cmd, bstep=self.STROKE_STEP)
        yield from self.wait(3, self.base(), q_cmd)
        self.log.append(('pass', round(yc, 3), round(blade_x0, 3), 'bb', round(self.blade_bottom(), 3),
                         'rem', len(self.remaining())))
        yield from self.cart_move(self.ee()[0] + [0, 0, 0.06], R, speed=0.01, max_steps=40)

    def open_drawer(self):
        """Hook closed fingers into the gap between drawer face and handle
        bar, then pull by backing the base off."""
        a = self.HOOK_A
        bx, by = self.HOOK_BASE
        base_t = np.array([bx, by, np.pi])
        R = R_hook(a)
        dv = R[:, 2]
        tip = np.array([self.HOOK_X, 0.0, self.HOOK_Z])
        site = tip - 0.02 * dv
        self.grip = 1.0
        yield from self.pose_move(site + np.array([0, 0, 0.08]), R, base_t, max_steps=200, tol=0.04)
        self.log.append(('od_pre', self.t))
        yield from self.cart_move(site, R, speed=0.012, base_t=base_t, max_steps=14, tol=0.006)
        self.log.append(('od_in', self.t))
        target = bx + self.DRAWER_OPEN + 0.03
        qhold = self.q()
        for _ in range(80):
            b = self.base()
            if self.drawer() >= self.DRAWER_OPEN:
                break
            bt = np.array([min(b[0] + self.PULL_STEP, target), by, np.pi])
            yield self.act(bt, qhold, bstep=self.PULL_STEP)
        self.log.append(('od_pull', self.t))
        # back out along the finger axis, then rise
        p = self.ee()[0]
        yield from self.cart_move(p - self.BACK_D * dv, R, speed=0.025, base_t=self.base().copy(),
                                  max_steps=self.BACK_N, tol=0.015, vtol=self.BACK_VTOL)
        if self.RISE_N > 0:
            p = self.ee()[0]
            yield from self.cart_move([p[0], p[1], self.Z_SAFE], R, speed=0.025, base_t=self.base().copy(),
                                      max_steps=self.RISE_N, tol=0.02, vtol=self.BACK_VTOL)
