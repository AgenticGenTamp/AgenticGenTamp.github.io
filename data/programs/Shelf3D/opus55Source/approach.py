"""Shelf3D (variable count) approach: pick cubes from the floor, place them on shelf.

Strategy: keep the arm in two (nearly) fixed configurations in the robot base frame
(tilted floor grasp close to the base, near-horizontal reach for shelf placement) that
differ mainly in two joints, and use the fast holonomic base for all horizontal
positioning.  Closed loop on the observed cube positions.
"""
import os
import numpy as np
import mujoco

_XML = "/robocode/third-party/kindergarden/src/kinder/envs/dynamic3d/models/stanford_tidybot/tidybot.xml"
if not os.path.exists(_XML):
    try:
        import kinder.envs.dynamic3d as _kd
        _XML = os.path.join(os.path.dirname(_kd.__file__), "models", "stanford_tidybot", "tidybot.xml")
    except Exception:  # pragma: no cover
        pass

CUBE_OFF = 0.015  # cube center along gripper approach axis from the pinch site
CONT = np.array([1, 0, 1, 0, 1, 0, 1], bool)  # continuous joints

# Offline-found good seeds (base frame).
Q_PICK_SEED = np.array([0.003, 1.705, 3.1426, -1.6836, 0.0097, 0.047, 1.5641])
Q_PLACE_SEED = np.array([-0.001, 1.095, 3.148, -1.3163, 0.005, 0.6404, 1.5616])
TH_PICK = -0.2
TH_PLACE = TH_PICK + np.pi / 2
R_PICK = 0.40   # horizontal reach (cube center) at pick
R_PLACE = 1.00  # horizontal reach (cube center) at place
Z_GRASP = 0.022
Z_HOVER = 0.075

SHELF_TOP_OFF = 0.5461  # shelf-2 surface height above cupboard origin
CUP_FRONT_OFF = -0.127  # front edge x relative to cupboard x


def rot_pitch(th, sign=1.0):
    z = np.array([np.sin(th), 0.0, -np.cos(th)])
    x = np.array([0.0, sign, 0.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class _Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(_XML)
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.jac_p = np.zeros((3, self.m.nv))
        self.jac_r = np.zeros((3, self.m.nv))

    def fk(self, q):
        d = self.d
        d.qpos[:] = 0
        d.qpos[3:10] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def cube_center(self, q):
        p, R = self.fk(q)
        return p + CUBE_OFF * R[:, 2], R

    def ik(self, pos_c, Rt, seed, iters=300, null_target=None):
        q = np.array(seed, dtype=float)
        for _ in range(iters):
            p, R = self.cube_center(q)
            ep = pos_c - p
            er = 0.5 * (np.cross(R[:, 0], Rt[:, 0]) + np.cross(R[:, 1], Rt[:, 1])
                        + np.cross(R[:, 2], Rt[:, 2]))
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 5e-4:
                return q, True
            mujoco.mj_jacSite(self.m, self.d, self.jac_p, self.jac_r, self.sid)
            Jr = self.jac_r[:, 3:10]
            off = CUBE_OFF * R[:, 2]
            Jp = self.jac_p[:, 3:10] + np.cross(Jr.T, off[None, :]).T
            J = np.vstack([Jp, Jr])
            JJt = J @ J.T + 1e-3 * np.eye(6)
            err = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(JJt, err)
            if null_target is not None:
                N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
                dq += N @ (0.1 * (null_target - q))
            n = np.abs(dq).max()
            if n > 0.3:
                dq *= 0.3 / n
            q = q + dq
        p, _ = self.cube_center(q)
        return q, bool(np.linalg.norm(pos_c - p) < 2e-3)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.kin = _Kin()
        self._cfg_cache = {}

    # ------------------------------------------------------------------ setup
    def _configs(self, cup_z):
        key = round(cup_z, 3)
        if key in self._cfg_cache:
            return self._cfg_cache[key]
        k = self.kin
        qa, _ = k.ik(np.array([R_PICK, 0, Z_GRASP]), rot_pitch(TH_PICK), Q_PICK_SEED,
                     null_target=Q_PICK_SEED)
        qh, _ = k.ik(np.array([R_PICK, 0, Z_HOVER]), rot_pitch(TH_PICK), qa, null_target=qa)
        zc = cup_z + SHELF_TOP_OFF + 0.02 + 0.010
        qb, _ = k.ik(np.array([R_PLACE, 0, zc]), rot_pitch(TH_PLACE), Q_PLACE_SEED,
                     null_target=Q_PLACE_SEED)
        cfg = dict(qa=qa, qh=qh, qb=qb,
                   pick_off=k.cube_center(qa)[0][:2],
                   place_off=k.cube_center(qb)[0][:2])
        self._cfg_cache[key] = cfg
        return cfg

    def _pick_cfg(self, zc):
        """Grasp/hover configs for a cube whose center is at height zc."""
        zc = max(Z_GRASP, zc + 0.002)
        key = int(round((zc - Z_GRASP) / 0.004))
        if key <= 0:
            return self.cfg["qa"], self.cfg["qh"], self.cfg["pick_off"]
        if key in self._pick_cache:
            return self._pick_cache[key]
        z = Z_GRASP + key * 0.004
        k = self.kin
        qa, _ = k.ik(np.array([R_PICK, 0, z]), rot_pitch(TH_PICK), self.cfg["qa"], null_target=self.cfg["qa"])
        qh, _ = k.ik(np.array([R_PICK, 0, z + Z_HOVER - Z_GRASP]), rot_pitch(TH_PICK), qa, null_target=qa)
        res = (qa, qh, k.cube_center(qa)[0][:2])
        self._pick_cache[key] = res
        return res

    def _arm_pts(self, q):
        self.kin.fk(q)
        d = self.kin.d
        pts = []
        for b in range(1, self.kin.m.nbody):
            pts.append(d.xpos[b].copy())
        # add geom centers too
        pts.extend(d.geom_xpos[g].copy() for g in range(self.kin.m.ngeom) if self.kin.m.geom_bodyid[g] > 1)
        return np.array(pts)

    def reset(self, state, info):
        self.robot = state.get_object_from_name("robot")
        cups = [state.get_object_from_name(n) for n in state.get_object_names() if n.startswith("cupboard")]
        self.cup = cups[0]
        self.cx = state.get(self.cup, "x")
        self.cy = state.get(self.cup, "y")
        self.cz = state.get(self.cup, "z")
        self.cfg = self._configs(self.cz)
        self.z_carry = self.cz + SHELF_TOP_OFF + 0.03
        self._pick_cache = {}
        self.pick = (self.cfg["qa"], self.cfg["qh"], self.cfg["pick_off"])
        pa = self._arm_pts(self.cfg["qa"])
        ph = self._arm_pts(self.cfg["qh"])
        self.pick_pts = np.vstack([pa, ph])
        names = sorted([n for n in state.get_object_names() if n.startswith("cube")],
                       key=lambda s: (len(s), s))
        self.cubes = [state.get_object_from_name(n) for n in names]
        self.n = len(self.cubes)
        self._make_slots()
        self.phase = "select"
        self.target = None
        self.slot = None
        self.timer = 0
        self.integ = np.zeros(7)
        self.steps = 0
        self.pick_yaw = 0.0
        self.base_goal = None
        self.failed_slots = set()
        self.fail_count = {}

    @staticmethod
    def _type(state, name):
        for t in state.type_features.keys() if hasattr(state, "type_features") else []:
            if t.name == name:
                return t
        return None

    def _make_slots(self):
        n = max(self.n, 1)
        if n <= 4:
            nrows = 1
        elif n <= 8:
            nrows = 2
        else:
            nrows = 3
        cap = int(np.ceil(n / nrows))
        if nrows == 1:
            xs = [-0.05]
        elif nrows == 2:
            xs = [0.02, -0.07]
        else:
            xs = [0.085, 0.0, -0.08]
        W = min(0.20, 0.07 * (cap - 1))
        ys = [0.0] if cap == 1 else list(np.linspace(-W, W, cap))
        slots = []
        for rx in xs:  # back row first
            for yy in ys:
                slots.append(np.array([self.cx + rx, self.cy + yy]))
        self.slots = slots
        self.slot_used = [False] * len(slots)

    # ------------------------------------------------------------- helpers
    def _robot(self, state):
        r = self.robot
        bx = state.get(r, "pos_base_x")
        by = state.get(r, "pos_base_y")
        bt = state.get(r, "pos_base_rot")
        q = np.array([state.get(r, f"pos_arm_joint{i}") for i in range(1, 8)])
        return np.array([bx, by, bt]), q

    def _cube(self, state, c):
        return np.array([state.get(c, "x"), state.get(c, "y"), state.get(c, "z")])

    def _cube_yaw(self, state, c):
        w, x, y, z = (state.get(c, f) for f in ("qw", "qx", "qy", "qz"))
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        # pick the most horizontal body axis
        i = int(np.argmin(np.abs(R[2, :])))
        return float(np.arctan2(R[1, i], R[0, i]))

    def _in_goal(self, p):
        return (self.cx - 0.12 <= p[0] <= self.cx + 0.12 and abs(p[1] - self.cy) <= 0.30
                and self.cz + 0.546 <= p[2] <= self.cz + 0.576)

    def _arm_action(self, q, qt):
        e = qt - q
        e[CONT] = wrap(e[CONT])
        # small integral term to remove steady-state error
        near = np.abs(e) < 0.05
        self.integ = np.where(near, np.clip(self.integ + 0.3 * e, -0.05, 0.05), 0.0)
        return np.clip(e + self.integ, -0.1, 0.1), np.abs(e).max()

    def _base_action(self, b, bt):
        e = np.array([bt[0] - b[0], bt[1] - b[1], wrap(bt[2] - b[2])])
        return np.clip(e, -0.1, 0.1), e

    def _pick_options(self, state, c, b, others):
        """Return list of (cost, base_pose) for picking cube c."""
        p = self._cube(state, c)
        yaw0 = self._cube_yaw(state, c)
        off = self.cfg["pick_off"]
        out = []
        for k in range(4):
            yaw = wrap(yaw0 + k * np.pi / 2)
            cs, sn = np.cos(yaw), np.sin(yaw)
            bx = p[0] - (cs * off[0] - sn * off[1])
            by = p[1] - (sn * off[0] + cs * off[1])
            cost = 0.0
            for o in others:
                if o[2] > 0.3:
                    continue
                dx, dy = o[0] - bx, o[1] - by
                lx = cs * dx + sn * dy
                ly = -sn * dx + cs * dy
                if abs(lx) < 0.36 and abs(ly) < 0.34:
                    cost += 10.0  # base would bump into it
                # relative to the target cube (gripper frame)
                gx, gy = o[0] - p[0], o[1] - p[1]
                glx = cs * gx + sn * gy
                gly = -sn * gx + cs * gy
                if abs(glx) < 0.055 and abs(gly) < 0.105:
                    cost += 30.0  # fingers would hit it
                elif abs(glx) < 0.08 and abs(gly) < 0.08:
                    cost += 5.0
            # base footprint / arm vs cupboard
            Rz = np.array([[cs, -sn], [sn, cs]])
            corners = np.array([[sx * 0.30, sy * 0.28] for sx in (-1, 0, 1) for sy in (-1, 0, 1)]) @ Rz.T + [bx, by]
            pts = self.pick_pts[:, :2] @ Rz.T + [bx, by]
            pts = pts[self.pick_pts[:, 2] < self.cz + 0.84]
            if self._in_cup(corners, 0.03).any():
                cost += 40.0
            if self._in_cup(pts, 0.05).any():
                cost += 40.0
            cost += np.hypot(bx - b[0], by - b[1]) + 0.3 * abs(wrap(yaw - b[2]))
            out.append((cost, np.array([bx, by, yaw])))
        return out

    def _in_cup(self, xy, margin):
        return ((xy[:, 0] > self.cx + CUP_FRONT_OFF - margin) & (xy[:, 0] < self.cx + 0.13 + margin)
                & (np.abs(xy[:, 1] - self.cy) < 0.305 + margin))

    def _select(self, state):
        b, _ = self._robot(state)
        allp = {c.name: self._cube(state, c) for c in self.cubes}
        best = None
        for c in self.cubes:
            p = allp[c.name]
            if self._in_goal(p) or p[2] > 0.3:
                continue
            others = [allp[o.name] for o in self.cubes if o is not c]
            for cost, pose in self._pick_options(state, c, b, others):
                cost += self.fail_count.get(c.name, 0) * 6.0
                if best is None or cost < best[0]:
                    best = (cost, c, pose)
        if best is None:
            return None, None
        return best[1], best[2]

    def _choose_slot(self, state):
        placed = [self._cube(state, c) for c in self.cubes]
        for i, s in enumerate(self.slots):
            if i in self.failed_slots:
                continue
            occ = any(np.hypot(p[0] - s[0], p[1] - s[1]) < 0.05 and p[2] > 0.3 for p in placed)
            if not occ:
                return i
        # fallback: any slot
        return 0

    # --------------------------------------------------------------- policy
    def get_action(self, state):
        self.steps += 1
        b, q = self._robot(state)
        cfg = self.cfg
        grip = 0.0
        arm_t = q.copy()
        base_t = b.copy()

        if self.phase == "select":
            self.target, self.base_goal = self._select(state)
            if self.target is None:
                return np.zeros(11, dtype=np.float32)
            self.pick = self._pick_cfg(self._cube(state, self.target)[2])
            self.phase = "go_pick"
            self.timer = 0

        if self.phase == "go_pick":
            # refresh goal with current cube position (cube may have moved)
            p = self._cube(state, self.target)
            yaw = self.base_goal[2]
            off = self.pick[2]
            cs, sn = np.cos(yaw), np.sin(yaw)
            self.base_goal[:2] = [p[0] - (cs * off[0] - sn * off[1]), p[1] - (sn * off[0] + cs * off[1])]
            base_t = self.base_goal
            arm_t = self.pick[1]
            grip = 0.0
            ba, be = self._base_action(b, base_t)
            aa, ae = self._arm_action(q, arm_t)
            self.timer += 1
            if np.hypot(be[0], be[1]) < 0.012 and abs(be[2]) < 0.03 and ae < 0.12:
                self.phase = "descend"
                self.timer = 0
            elif self.timer > 110:
                self.fail_count[self.target.name] = self.fail_count.get(self.target.name, 0) + 1
                self.phase = "select"
            return self._act(ba, aa, grip)

        if self.phase == "descend":
            p = self._cube(state, self.target)
            yaw = self.base_goal[2]
            off = self.pick[2]
            cs, sn = np.cos(yaw), np.sin(yaw)
            self.base_goal[:2] = [p[0] - (cs * off[0] - sn * off[1]), p[1] - (sn * off[0] + cs * off[1])]
            ba, be = self._base_action(b, self.base_goal)
            aa, ae = self._arm_action(q, self.pick[0])
            self.timer += 1
            if (ae < 0.015 and np.hypot(be[0], be[1]) < 0.006) or self.timer > 40:
                self.phase = "grasp"
                self.timer = 0
            return self._act(ba, aa, 0.0)

        if self.phase == "grasp":
            aa, ae = self._arm_action(q, self.pick[0])
            self.timer += 1
            if self.timer >= 5:
                self.phase = "transfer"
                self.timer = 0
                self.slot = self._choose_slot(state)
            return self._act(np.zeros(3), aa, 1.0)

        if self.phase == "transfer":
            p = self._cube(state, self.target)
            self.timer += 1
            if self.timer > 15 and p[2] < 0.05:
                # dropped the cube
                self.phase = "select"
                return self._act(np.zeros(3), np.zeros(7), 0.0)
            s = self.slots[self.slot]
            off = cfg["place_off"]
            stage = np.array([s[0] - off[0] - 0.25, s[1] - off[1], 0.0])
            ba, be = self._base_action(b, stage)
            aa, ae = self._arm_action(q, cfg["qb"])
            zok = p[2] > self.z_carry - 0.008
            if self.timer > 90 and not zok:
                self.fail_count[self.target.name] = self.fail_count.get(self.target.name, 0) + 1
                self.phase = "putdown"
                self.timer = 0
            if ae < 0.03 and zok and np.hypot(be[0], be[1]) < 0.03 and abs(be[2]) < 0.03:
                self.phase = "insert"
                self.timer = 0
            return self._act(ba, aa, 1.0)

        if self.phase == "insert":
            p = self._cube(state, self.target)
            s = self.slots[self.slot]
            # closed loop on the cube's measured position
            off = cfg["place_off"]
            nominal_x = s[0] - off[0]
            gx = min(b[0] + (s[0] - p[0]), nominal_x + 0.04)
            gy = b[1] + (s[1] - p[1])
            if p[2] < self.z_carry - 0.008 and p[0] < self.cx + CUP_FRONT_OFF + 0.03:
                gx = b[0]  # wait until the cube is high enough to clear the shelf lip
            goal = np.array([gx, gy, 0.0])
            ba, be = self._base_action(b, goal)
            aa, ae = self._arm_action(q, cfg["qb"])
            self.timer += 1
            if (np.hypot(be[0], be[1]) < 0.008 and ae < 0.02) or self.timer > 40:
                self.phase = "release"
                self.timer = 0
            return self._act(ba, aa, 1.0)

        if self.phase == "putdown":
            aa, ae = self._arm_action(q, cfg["qh"])
            self.timer += 1
            grip = 1.0
            if ae < 0.03 or self.timer > 60:
                grip = 0.0
                if ae < 0.03:
                    self.timer = max(self.timer, 61)
                if self.timer > 66:
                    self.phase = "select"
            return self._act(np.zeros(3), aa, grip)

        if self.phase == "release":
            aa, ae = self._arm_action(q, cfg["qb"])
            self.timer += 1
            if self.timer >= 3:
                self.phase = "retreat"
                self.timer = 0
                self.retreat_goal = np.array([b[0] - 0.22, b[1], 0.0])
            return self._act(np.zeros(3), aa, 0.0)

        if self.phase == "retreat":
            ba, be = self._base_action(b, self.retreat_goal)
            aa, ae = self._arm_action(q, cfg["qb"])
            self.timer += 1
            if np.hypot(be[0], be[1]) < 0.03 or self.timer > 10:
                self.phase = "select"
            return self._act(ba, aa, 0.0)

        return np.zeros(11, dtype=np.float32)

    def _act(self, ba, aa, grip):
        a = np.zeros(11, dtype=np.float32)
        a[0:3] = ba
        a[3:10] = aa
        a[10] = grip
        return np.clip(a, self.action_space.low, self.action_space.high).astype(np.float32)
