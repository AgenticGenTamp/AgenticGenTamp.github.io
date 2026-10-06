"""Pick-and-place approach for kinder/BalanceBeam3D-o3-v0.

Uses a private copy of the MuJoCo model for forward kinematics and a damped
least-squares IK on the 7 arm joints. A generator-based state machine picks
each block and places it on the seesaw beam (large block at the pivot, small
blocks symmetric on either side).
"""

import math

import mujoco
import numpy as np

# observation indices
LB, ROB, SS, SB1, SB2 = 0, 16, 38, 54, 70
BASE_Q = slice(16, 19)
ARM_Q = slice(19, 26)
GRIP_Q = 26


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def down_rot(psi):
    """Rotation of pinch site: z down, finger axis (x) at yaw psi."""
    x = np.array([math.cos(psi), math.sin(psi), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


class Kin:
    def __init__(self):
        from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv

        env = KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
        ke = env._kinder_env
        oc = ke._object_centric_env
        oc.reset(seed=0) if hasattr(oc, "reset") else None
        re = oc._robot_env
        self.m = re.sim.model.mj_model
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.base_qadr = [m.jnt_qposadr[jid(n)] for n in
                          ("robot_joint_x", "robot_joint_y", "robot_joint_th")]
        arm_j = [jid(f"robot_joint_{i}") for i in range(1, 8)]
        self.arm_qadr = [m.jnt_qposadr[j] for j in arm_j]
        self.arm_dadr = [m.jnt_dofadr[j] for j in arm_j]
        self.lim = [(m.jnt_range[j].copy() if m.jnt_limited[j] else None) for j in arm_j]
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def _set(self, base, q):
        d = self.d
        for a, v in zip(self.base_qadr, base):
            d.qpos[a] = v
        for a, v in zip(self.arm_qadr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(self.m, d)

    def fk(self, base, q):
        self._set(base, q)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, base, pos, R, q0, iters=300, wrot=0.5, bias=None):
        q = np.array(q0, dtype=float)
        best = (1e9, q.copy())
        for _ in range(iters):
            self._set(base, q)
            p = self.d.site_xpos[self.site]
            Rc = self.d.site_xmat[self.site].reshape(3, 3)
            ep = pos - p
            er = 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
            err = np.linalg.norm(ep) + 0.1 * np.linalg.norm(er)
            if err < best[0]:
                best = (err, q.copy())
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, self.arm_dadr], wrot * self.jacr[:, self.arm_dadr]])
            e = np.concatenate([ep, wrot * er])
            lam = 1e-3
            JJ = J @ J.T + lam * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            if bias is not None:
                N = np.eye(7) - J.T @ np.linalg.solve(JJ, J)
                dq += N @ (0.1 * np.array([wrap(x) for x in (bias - q)]))
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = q + dq
            for i, l in enumerate(self.lim):
                if l is not None:
                    q[i] = np.clip(q[i], l[0] + 1e-3, l[1] - 1e-3)
        self._set(base, best[1])
        p = self.d.site_xpos[self.site]
        Rc = self.d.site_xmat[self.site].reshape(3, 3)
        perr = np.linalg.norm(pos - p)
        rerr = np.linalg.norm(0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3)))
        return best[1], perr, rerr


HOME = np.array([0.0, -0.349, np.pi, -2.548, 0.0, -0.873, np.pi / 2])
HOME = np.array([0.0, -0.349, np.pi, -2.548, 0.0, -0.873, np.pi / 2])

# base footprint in base frame (x fwd, y left), approximate (from mesh bbox)
FP_X = (-0.16, 0.165)
FP_Y = (-0.262, 0.262)
REACH = 0.5           # horizontal forward reach of the gripper in base frame
ZC_MARGIN = 0.02
Z_CARRY = 0.13        # pinch-site height while carrying a block
Z_EMPTY = 0.12        # pinch-site height while moving empty
BASE_V = 0.085        # base translation per step (per axis)
BASE_W = 0.1          # base rotation per step


def rot2(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.kin = Kin()
        self.s = None
        self.gen = None
        self._post = {}
        # pre-compute the main postures
        q = HOME.copy()
        for z in (Z_CARRY, Z_EMPTY, 0.07, 0.015):
            q = self.posture(z, q)

    # ------------------------------------------------------------------
    def posture(self, z, seed=None):
        """Arm joints placing pinch site at (REACH, 0, z) in base frame,
        pointing down, fingers along base y."""
        key = round(z, 3)
        if key in self._post:
            return self._post[key][0]
        if seed is None:
            # nearest cached posture as seed
            if self._post:
                k0 = min(self._post, key=lambda k: abs(k - key))
                seed = self._post[k0][0]
            else:
                seed = HOME
        best = None
        for sd in (seed, HOME):
            q, pe, re_ = self.kin.ik(np.zeros(3), np.array([REACH, 0.0, key]),
                                     down_rot(np.pi / 2), sd, iters=400, bias=sd)
            if best is None or pe + 0.05 * re_ < best[1]:
                best = (q, pe + 0.05 * re_)
            if pe < 1e-3:
                break
        q = best[0]
        p, _ = self.kin.fk(np.zeros(3), q)
        self._post[key] = (q, p[:2].copy())
        return q

    def site_off(self, z):
        self.posture(z)
        return self._post[round(z, 3)][1]

    def base_for(self, target_xy, yaw, z):
        """Base pose that puts the pinch site (posture z) over target_xy."""
        off = self.site_off(z)
        xy = np.asarray(target_xy[:2]) - rot2(yaw) @ off
        return np.array([xy[0], xy[1], yaw])

    # ------------------------------------------------------------------
    def reset(self, state, info):
        self.s = np.asarray(state, dtype=float)
        self.gen = self._plan()
        self.last_grip = 0.0
        self.zc = Z_CARRY

    def get_action(self, state):
        self.s = np.asarray(state, dtype=float)
        try:
            a = next(self.gen)
        except StopIteration:
            a = self._hold_action()
        a = np.asarray(a, dtype=np.float32)
        return np.clip(a, self.action_space.low, self.action_space.high)

    def _hold_action(self):
        a = np.zeros(11)
        a[10] = self.last_grip
        return a

    def base(self):
        return self.s[BASE_Q].copy()

    def arm(self):
        return self.s[ARM_Q].copy()

    def blk(self, idx):
        o = {0: LB, 1: SB1, 2: SB2}[idx]
        return self.s[o:o + 3].copy(), self.s[o + 3:o + 7].copy()

    # ------------------------------------------------------------------
    # collision checking for the base footprint
    def _obstacles(self, exclude=()):
        """List of (center_xy, R, half_extents) boxes for floor blocks and seesaw."""
        obs = []
        for i in range(3):
            if i in exclude:
                continue
            p, _ = self.blk(i)
            if p[2] < 0.03:  # on the floor
                obs.append((p[:2], 0.0, np.array([0.03, 0.03])))
        ss = self.s[SS:SS + 3]
        R = quat_to_mat(self.s[SS + 3:SS + 7])
        th = math.atan2(R[1, 0], R[0, 0])
        obs.append((ss[:2], th, np.array([0.21, 0.06])))
        return obs

    @staticmethod
    def _box_overlap(c1, th1, h1, c2, th2, h2):
        """Separating axis test between two 2D oriented boxes."""
        R1, R2 = rot2(th1), rot2(th2)
        d = c2 - c1
        for ax in (R1[:, 0], R1[:, 1], R2[:, 0], R2[:, 1]):
            r1 = h1[0] * abs(R1[:, 0] @ ax) + h1[1] * abs(R1[:, 1] @ ax)
            r2 = h2[0] * abs(R2[:, 0] @ ax) + h2[1] * abs(R2[:, 1] @ ax)
            if abs(d @ ax) > r1 + r2:
                return False
        return True

    def _fp_collides(self, pose, obstacles, margin=0.02):
        cx = 0.5 * (FP_X[0] + FP_X[1])
        cy = 0.5 * (FP_Y[0] + FP_Y[1])
        h = np.array([0.5 * (FP_X[1] - FP_X[0]) + margin, 0.5 * (FP_Y[1] - FP_Y[0]) + margin])
        c = pose[:2] + rot2(pose[2]) @ np.array([cx, cy])
        for oc, oth, oh in obstacles:
            if self._box_overlap(c, pose[2], h, np.asarray(oc), oth, oh):
                return True
        return False

    def _sim_leg(self, start, goal):
        """Simulate base motion (per-axis clipped) from start to goal."""
        p = np.array(start, dtype=float)
        pts = [p.copy()]
        for _ in range(200):
            d = np.array(goal) - p
            d[2] = wrap(d[2])
            if np.max(np.abs(d[:2])) < 1e-3 and abs(d[2]) < 1e-3:
                break
            step = np.array([np.clip(d[0], -BASE_V, BASE_V), np.clip(d[1], -BASE_V, BASE_V),
                             np.clip(d[2], -BASE_W, BASE_W)])
            p = p + step
            pts.append(p.copy())
        return pts

    @staticmethod
    def _leg_est(p, q):
        d = np.abs(np.array(q[:2]) - np.array(p[:2]))
        dth = abs(wrap(q[2] - p[2]))
        return max(d[0] / BASE_V, d[1] / BASE_V, dth / BASE_W) + 2.0

    def _path_free(self, legs, obstacles, margin):
        p = legs[0]
        steps = 0
        for wp in legs[1:]:
            pts = self._sim_leg(p, wp)
            steps += len(pts) + 2
            for q in pts[1:]:
                if self._fp_collides(q, obstacles, margin):
                    return None
            p = wp
        return steps

    def plan_route(self, goal, exclude=(), margin=0.02, start=None, return_cost=False):
        """Return list of base waypoints from current base pose to goal."""
        start = self.base() if start is None else np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        obstacles = self._obstacles(exclude)
        cands = [(self._leg_est(start, goal), [goal])]
        for x in np.linspace(-0.4, 2.2, 14):
            for y in np.linspace(-1.0, 1.0, 11):
                for th in {round(start[2], 4), round(goal[2], 4)}:
                    w = np.array([x, y, th])
                    c = self._leg_est(start, w) + self._leg_est(w, goal)
                    cands.append((c, [w, goal]))
        for L in (0.62, -0.62, 0.8, -0.8):
            for th in (start[2], goal[2]):
                w1 = np.array([start[0], L, th])
                w2 = np.array([goal[0], L, goal[2]])
                c = self._leg_est(start, w1) + self._leg_est(w1, w2) + self._leg_est(w2, goal)
                cands.append((c, [w1, w2, goal]))
        cands.sort(key=lambda c: c[0])
        best = None
        for est, wps in cands[:400]:
            if best is not None and est >= best[0]:
                break
            steps = self._path_free([start] + wps, obstacles, margin)
            if steps is not None and (best is None or steps < best[0]):
                best = (steps, wps)
        if best is None:
            # fall back to least-colliding lane route
            best = (1e4, cands[0][1])
            for est, wps in cands:
                if len(wps) == 3:
                    best = (1e4, wps)
                    break
        if return_cost:
            return best[1], best[0]
        return best[1]

    # ------------------------------------------------------------------
    def move(self, base_t, q_t, grip, tol_b=0.006, tol_yaw=0.01, tol_p=0.005, tol_q=None,
             max_steps=120, gain=1.5, ki=0.3, kib=0.3, q_near=None, near_pt=None,
             near_d=0.75):
        self.last_grip = grip
        q_final = np.array(q_t, dtype=float)
        base_t = np.array(base_t, dtype=float)
        q_t = np.array(q_t, dtype=float)
        p_t, _ = self.kin.fk(np.zeros(3), q_final)
        integ = np.zeros(7)
        integ_b = np.zeros(3)
        for k in range(max_steps):
            b = self.base()
            q = self.arm()
            db = base_t - b
            db[2] = wrap(db[2])
            if q_near is not None and np.linalg.norm(b[:2] - near_pt) < near_d:
                q_t = np.array(q_near, dtype=float)
            else:
                q_t = q_final
            dq = np.array([wrap(x) for x in (q_t - q)])
            if tol_q is not None:
                arm_ok = np.max(np.abs(dq)) < tol_q
            else:
                p, _ = self.kin.fk(np.zeros(3), q)
                arm_ok = np.linalg.norm(p - p_t) < tol_p and np.max(np.abs(dq)) < 0.08
            if np.max(np.abs(db[:2])) < tol_b and abs(db[2]) < tol_yaw and arm_ok:
                return
            small = np.abs(dq) < 0.06
            integ = np.where(small, integ + dq, 0.0)
            smallb = np.abs(db) < 0.02
            integ_b = np.where(smallb, integ_b + db, 0.0)
            a = np.zeros(11)
            a[0:3] = np.clip(db + kib * integ_b, -0.1, 0.1)
            a[3:10] = np.clip(gain * dq + ki * integ, -0.1, 0.1)
            a[10] = grip
            yield a

    def wait(self, n, grip):
        self.last_grip = grip
        for _ in range(n):
            a = np.zeros(11)
            a[10] = grip
            yield a

    def go(self, goal, q_t, grip, exclude=(), q_transit=None, tol_p=0.02, **kw):
        """Route the base to goal (collision-aware) while moving arm.
        The arm stays at carry height while the base is near the seesaw."""
        wps = self.plan_route(goal, exclude)
        qq = q_t if q_transit is None else q_transit
        qc = self.posture(self.zc)
        near = self.s[SS:SS + 2].copy()
        if np.linalg.norm(goal[:2] - near) < 0.75:
            qn = None  # final target is at the seesaw itself
        else:
            qn = qc
        for wp in wps[:-1]:
            yield from self.move(wp, qq, grip, tol_b=0.05, tol_yaw=0.1, tol_q=10.0, max_steps=100,
                                 q_near=qc, near_pt=near)
        yield from self.move(goal, q_t, grip, tol_p=tol_p, max_steps=150,
                             q_near=qn, near_pt=near, **kw)

    def grasp_candidates(self, idx, bpos, byaw, z_grasp):
        obstacles = self._obstacles(exclude=(idx,))
        others = []
        for i in range(3):
            if i == idx:
                continue
            p, _ = self.blk(i)
            if p[2] < 0.05:
                others.append(p[:2])
        cands = []
        for k in range(4):
            th = wrap(byaw + k * np.pi / 2)
            cand = self.base_for(bpos, th, z_grasp)
            coll = self._fp_collides(cand, obstacles, 0.02)
            # finger clearance: fingers along base y (u), thin along base x (v)
            Rt = rot2(th)
            clear = 0.1
            for o in others:
                v, u = Rt.T @ (o - bpos[:2])
                if abs(u) < 0.09:
                    clear = min(clear, abs(v))
            fpen = max(0.0, 0.045 - clear)
            cost = 100.0 * coll + 100.0 * fpen + 0.01 * abs(th)
            if not coll and fpen <= 0.0:
                _, rc = self.plan_route(cand, exclude=(idx,), return_cost=True)
                cost += 0.01 * rc
            cands.append((cost, cand))
        cands.sort(key=lambda c: c[0])
        return [c[1] for c in cands]

    def pick(self, idx, q_empty, q_carry, attempt=0):
        bpos, bq = self.blk(idx)
        Rb = quat_to_mat(bq)
        byaw = math.atan2(Rb[1, 0], Rb[0, 0])
        hb = 0.018 if idx == 0 else 0.012
        z_grasp = max(bpos[2] + 0.0105, bpos[2] - hb + 0.033)
        q_grasp = self.posture(z_grasp)
        cands = self.grasp_candidates(idx, bpos, byaw, z_grasp)
        gbase = cands[attempt % len(cands)]
        yield from self.go(gbase, q_empty, 0.0, exclude=(idx,), tol_p=0.01)
        bpos, _ = self.blk(idx)
        gbase = self.base_for(bpos, gbase[2], z_grasp)
        yield from self.move(gbase, q_grasp, 0.0, tol_p=0.004, max_steps=60)
        yield from self.wait(3, 1.0)
        yield from self.move(self.base(), q_carry, 1.0, tol_p=0.07, max_steps=40)
        if self.held_block() is not None:
            return True
        # failed: open and back up to empty height
        yield from self.move(self.base(), q_empty, 0.0, tol_p=0.03, max_steps=20)
        return False

    # ------------------------------------------------------------------
    def _plan(self):
        kin = self.kin
        ss_pos = self.s[SS:SS + 3].copy()
        ss_R = quat_to_mat(self.s[SS + 3:SS + 7])
        bdir = ss_R[:2, 0].copy()
        bdir /= np.linalg.norm(bdir)
        beam_top = ss_pos[2] + 0.0475
        half = {0: 0.018, 1: 0.012, 2: 0.012}

        # robot approaches the beam along its axis from side s
        s = 1.0 if (ss_pos[:2] @ bdir) >= 0 else -1.0
        place_yaw = math.atan2(-s * bdir[1], -s * bdir[0])
        # (block index, offset along bdir)  far slot first
        plan = [(1, -s * 0.07), (0, 0.0), (2, s * 0.07)]

        q_empty = self.posture(Z_EMPTY)
        q_carry = self.posture(Z_CARRY)
        slots = {"far": -s * 0.07, "center": 0.0, "near": s * 0.07}
        filled = set()
        placed = set()
        for n in range(12):
            onb = [i for i in range(3) if self.in_region(i)]
            # carry height: carried block must clear blocks already on the beam
            if onb:
                top = max(self.blk(i)[0][2] + half[i] for i in onb)
                self.zc = max(Z_CARRY, top + 0.045 + 0.03)
            else:
                self.zc = Z_CARRY
            q_carry = self.posture(self.zc)
            if len(onb) == 3:
                break
            placed = set(onb)
            filled = set()
            for i in onb:
                o = (self.blk(i)[0][:2] - ss_pos[:2]) @ bdir
                filled.add(min(slots, key=lambda k: abs(slots[k] - o)))
            remaining = [i for i in range(3) if i not in placed]
            stacked = [i for i in remaining if self.blk(i)[0][2] > 0.03]
            if stacked:
                idx = max(stacked, key=lambda i: self.blk(i)[0][2])
            else:
                def pcost(i):
                    bp, _ = self.blk(i)
                    _, c = self.plan_route(self.base_for(bp, 0.0, 0.033), exclude=(i,),
                                           return_cost=True)
                    return c
                idx = min(remaining, key=pcost)
            ok = False
            for attempt in range(4):
                ok = yield from self.pick(idx, q_empty, q_carry, attempt)
                if ok:
                    break
            held = self.held_block()
            if held is None:
                continue
            idx = held
            last = len(placed) == 2
            sp, _ = kin.fk(self.base(), self.arm())
            bnow, _ = self.blk(idx)
            off_w = bnow - sp  # block relative to site (world)
            cur_yaw = self.base()[2]
            if onb:
                top = max(self.blk(i)[0][2] + half[i] for i in onb)
                self.zc = max(Z_CARRY, top + ZC_MARGIN + half[idx] - off_w[2])
                q_carry = self.posture(self.zc)

            if last:
                # hover the held block anywhere inside the goal region
                opts = []
                for o in (-0.1, -0.05, 0.0, 0.05, 0.1):
                    for k in range(8):
                        yaw = wrap(k * np.pi / 4)
                        off_xy = rot2(yaw - cur_yaw) @ off_w[:2]
                        site_xy = ss_pos[:2] + bdir * o - off_xy
                        pb = self.base_for(site_xy, yaw, self.zc)
                        opts.append(pb)
                obstacles = self._obstacles(exclude=(idx,))
                opts = [p for p in opts if not self._fp_collides(p, obstacles, 0.03)]
                opts.sort(key=lambda p: self._leg_est(self.base(), p))
                best = None
                for p in opts[:8]:
                    _, c = self.plan_route(p, exclude=(idx,), return_cost=True)
                    if best is None or c < best[0]:
                        best = (c, p)
                if best is not None:
                    yield from self.go(best[1], q_carry, 1.0, exclude=(idx,), tol_p=0.02)
                    yield from self.move(best[1], q_carry, 1.0, max_steps=20)
                    if self.in_region(idx):
                        yield from self.wait(5, 1.0)
                        continue

            # choose slot + approach end by route cost
            free = [k for k in slots if k not in filled]
            if idx == 0 and "center" in free:
                cand_slots = ["center"]
            else:
                cand_slots = [k for k in free if k != "center"] or free
            opts = []
            obstacles = self._obstacles(exclude=(idx,))
            for slot in cand_slots:
                for e in (1.0, -1.0):
                    yaw = math.atan2(-e * bdir[1], -e * bdir[0])
                    tgt = ss_pos[:2] + bdir * slots[slot]
                    tz = beam_top + half[idx] + 0.004
                    off_xy = rot2(yaw - cur_yaw) @ off_w[:2]
                    site_xy = tgt - off_xy
                    z_place = tz - off_w[2]
                    pb = self.base_for(site_xy, yaw, z_place)
                    if self._fp_collides(pb, obstacles, 0.02):
                        continue
                    _, c = self.plan_route(self.base_for(site_xy, yaw, self.zc),
                                           exclude=(idx,), return_cost=True)
                    # prefer placing on the far side of already placed blocks
                    o_rel = slots[slot] * e
                    c += 0.0 if o_rel <= 0 else 3.0
                    opts.append((c, slot, yaw, site_xy, z_place, pb))
            if not opts:
                slot = cand_slots[0]
                yaw = place_yaw
                tgt = ss_pos[:2] + bdir * slots[slot]
                off_xy = rot2(yaw - cur_yaw) @ off_w[:2]
                site_xy = tgt - off_xy
                z_place = beam_top + half[idx] + 0.004 - off_w[2]
                pb = self.base_for(site_xy, yaw, z_place)
                opts = [(0, slot, yaw, site_xy, z_place, pb)]
            _, slot, yaw, site_xy, z_place, pbase = min(opts, key=lambda o: o[0])
            q_place = self.posture(z_place)
            pbase_c = self.base_for(site_xy, yaw, self.zc)
            yield from self.go(pbase_c, q_carry, 1.0, exclude=(idx,), tol_p=0.02)
            yield from self.move(pbase, q_place, 1.0, tol_p=0.004, max_steps=60)
            yield from self.wait(1, 0.0)
            # lift clear while backing straight out along the beam
            q_up = self.posture(z_place + 0.05)
            yield from self.move(pbase, q_up, 0.0, tol_p=0.03, max_steps=30)
            back = pbase.copy()
            back[:2] -= 0.15 * np.array([math.cos(yaw), math.sin(yaw)])
            yield from self.move(back, q_up, 0.0, tol_b=0.04, tol_yaw=0.1, tol_q=10.0, max_steps=30)
        yield from self.wait(1000, 0.0)

    def in_region(self, i):
        p, _ = self.blk(i)
        ss = self.s[SS:SS + 3]
        R = quat_to_mat(self.s[SS + 3:SS + 7])
        loc = R.T @ (p - ss)
        return abs(loc[0]) < 0.18 and abs(loc[1]) < 0.04 and 0.035 < loc[2] < 0.2

    def held_block(self):
        sp, _ = self.kin.fk(self.base(), self.arm())
        best = None
        for i in range(3):
            p, _ = self.blk(i)
            d = np.linalg.norm(p - sp)
            if p[2] > 0.05 and d < 0.06 and (best is None or d < best[0]):
                best = (d, i)
        return None if best is None else best[1]
