"""Scripted approach for ConstrainedCupboard3D (variable rod count).

Strategy: for every rod, approach it along its long axis, grasp it top-down near
its rear end, drive the base in front of the target cupboard compartment
(facing +x), slide the rod in with the base, lower it onto the shelf, release.
The arm uses a small set of canonical configurations (computed with IK on a
standalone MuJoCo TidyBot model); fine positioning is done with the base.
"""

import json
import os

import mujoco
import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
REACH = 0.65          # horizontal distance base center -> pinch site (base x)
GRIP_OFF = 0.10       # distance from grasp point to rod center along rod
FLOOR_PINCH_Z = 0.035
PINCH_ABOVE_ROD = 0.02  # pinch z - rod center z when grasped
CUP_FRONT_DX = -0.15   # cupboard front face relative to cupboard x
ROD_HALF = 0.15
ROD_THICK = 0.03
SHELF_T = 0.02
ARM_LEAD = 0.5  # command beyond the target: speeds up the slow arm PD
INSERT_DZ = 0.08  # insert above final height, lower inside the cupboard
TILT = 0.6            # gripper pitch (approach axis tilted toward +x)
PINCH_MAX_DX = -0.035  # max pinch x relative to cupboard front

CONT_JOINTS = [0, 2, 4, 6]  # continuous arm joints (indices in 7)
JLIM = {1: 2.24, 3: 2.57, 5: 2.09}
NOMINAL_Q = np.array([0.0, 1.86, np.pi, -1.66, 0.0, 0.98, 1.55])
HOME_Q = np.deg2rad([0, -20, 180, -146, 0, -50, 90])


def _pkg_dir(pkg):
    if getattr(pkg, "__file__", None):
        return os.path.dirname(pkg.__file__)
    return list(pkg.__path__)[0]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _quat_to_mat(qw, qx, qy, qz):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.array([qw, qx, qy, qz], dtype=float))
    return m.reshape(3, 3)


# --------------------------------------------------------------------------
# Kinematics on a standalone TidyBot model
# --------------------------------------------------------------------------
class _Kin:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        import kinder.envs.dynamic3d as pkg

        path = os.path.join(
            _pkg_dir(pkg), "models", "stanford_tidybot", "tidybot.xml"
        )
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.cache = {}

    def fk(self, q, base=(0.0, 0.0, 0.0)):
        self.d.qpos[:] = 0
        self.d.qpos[0:3] = base
        self.d.qpos[3:10] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, p_des, R_des, q_seed, iters=300, min_it=60):
        q = np.array(q_seed, dtype=float).copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        err = None
        for it in range(iters):
            p, R = self.fk(q)
            mujoco.mj_comPos(self.m, self.d)
            ep = p_des - p
            # orientation error (axis-angle of R_des R^T)
            Re = R_des @ R.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, Re.flatten())
            if quat[0] < 0:
                quat = -quat
            ang = 2 * np.arccos(np.clip(quat[0], -1, 1))
            s = np.sqrt(max(1 - quat[0] ** 2, 1e-12))
            er = quat[1:] / s * ang if ang > 1e-9 else np.zeros(3)
            e = np.concatenate([ep, 0.5 * er])
            err = (np.linalg.norm(ep), np.linalg.norm(er))
            if err[0] < 1e-4 and err[1] < 1e-3 and it >= min_it:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, 3:10], 0.5 * jr[:, 3:10]])
            lam = 1e-3
            Jp = _spd_solve(J @ J.T + lam * np.eye(6), J).T
            dq = Jp @ e
            # null-space posture bias (j1 -> 0, j3 -> pi, j5 -> 0)
            dpost = np.zeros(7)
            dpost[0] = _wrap(0.0 - q[0])
            dpost[2] = _wrap(np.pi - q[2])
            dpost[4] = _wrap(0.0 - q[4])
            dq += (np.eye(7) - Jp @ J) @ (0.2 * dpost)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q += dq
            for j, lim in JLIM.items():
                q[j] = np.clip(q[j], -lim + 0.02, lim - 0.02)
        return q, err

    def canon(self, z, tilt=None, gamma=0.0, reach=REACH):
        """Arm config with pinch at base-frame (reach, 0, z), approach axis
        tilted by `tilt` from vertical, finger axis horizontal; with gamma=0 a
        held rod lies along base x, gamma rotates the gripper about vertical."""
        if tilt is None:
            tilt = TILT
        z = round(z / 0.004) * 0.004
        tilt = round(tilt / 0.02) * 0.02
        gamma = round(gamma / 0.04) * 0.04
        key = (round(z, 3), round(tilt, 2), round(gamma, 2), round(reach, 3))
        if key in self.cache:
            return self.cache[key]
        seed = NOMINAL_Q.copy()
        seed[6] += gamma
        min_it = 60
        best = None
        for k, qv in self.cache.items():
            if k[2] == key[2] and k[3] == key[3]:
                d = abs(k[0] - key[0]) + abs(k[1] - key[1])
                if d < 0.1 and (best is None or d < best[0]):
                    best = (d, qv)
        if best is not None:
            seed, min_it = best[1].copy(), 8
        cg, sg = np.cos(gamma), np.sin(gamma)
        Rz = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
        q, err = self.ik(np.array([reach, 0.0, z]), Rz @ _R_tilt(tilt), seed,
                         min_it=min_it)
        self.cache[key] = q
        return q


def _spd_solve(A, B):
    """Solve A X = B for small SPD A with MuJoCo's (single-threaded) Cholesky;
    avoids LAPACK thread contention on busy machines."""
    L = np.ascontiguousarray(A, dtype=float).copy()
    mujoco.mju_cholFactor(L, 0.0)
    X = np.empty(B.shape)
    x = np.empty(A.shape[0])
    for j in range(B.shape[1]):
        mujoco.mju_cholSolve(x, L, np.ascontiguousarray(B[:, j], dtype=float))
        X[:, j] = x
    return X


def _R_tilt(tilt):
    xs = np.array([0.0, 1.0, 0.0])
    zs = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    return np.column_stack([xs, np.cross(zs, xs), zs])


# --------------------------------------------------------------------------
# Task specification (goal assignment)
# --------------------------------------------------------------------------
def _load_goals(n):
    """Return list of (cuboid_name, cupboard_name, shelf_idx, shelf_heights)."""
    try:
        import kinder.envs.dynamic3d as pkg

        path = os.path.join(
            _pkg_dir(pkg), "tasks", "ConstrainedCupboard3D",
            f"ConstrainedCupboard3D-o{n}.json",
        )
        with open(path) as f:
            cfg = json.load(f)
        cups = cfg["fixtures"]["cupboard"]
        out = []
        for _, obj, region in cfg["goal_state"]:
            parts = region.split("_")
            cup = f"{parts[0]}_{parts[1]}"
            shelf = int(parts[3])
            out.append((obj, cup, shelf, list(cups[cup]["shelf_heights"])))
        return out
    except Exception:  # pragma: no cover - fallback table
        if n <= 2:
            return [(f"cuboid_{i}", "cupboard_1", 0, [0.4, 0.1]) for i in range(n)]
        out = []
        for i in range(n):
            if i < 3:
                out.append((f"cuboid_{i}", f"cupboard_{i}", 1, [0.1, 0.4]))
            else:
                out.append((f"cuboid_{i}", f"cupboard_{i}", 0, [0.4, 0.1]))
        return out


def _shelf_surface_z(heights, shelf):
    z = SHELF_T
    for h in heights[:shelf]:
        z += h + SHELF_T
    return z


# --------------------------------------------------------------------------
# Approach
# --------------------------------------------------------------------------
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.kin = _Kin.get()
        # warm up canonical configs
        for z in (0.6, 0.3, 0.2, FLOOR_PINCH_Z + 0.12, FLOOR_PINCH_Z):
            self.kin.canon(z)

    # ---------------- state helpers ----------------
    def _read(self, state):
        rob = state.get_object_from_name("robot")
        base = np.array([state.get(rob, "pos_base_x"), state.get(rob, "pos_base_y"),
                         state.get(rob, "pos_base_rot")])
        q = np.array([state.get(rob, f"pos_arm_joint{i}") for i in range(1, 8)])
        return base, q

    def _rod(self, state, name):
        o = state.get_object_from_name(name)
        c = np.array([state.get(o, "x"), state.get(o, "y"), state.get(o, "z")])
        R = _quat_to_mat(state.get(o, "qw"), state.get(o, "qx"), state.get(o, "qy"),
                         state.get(o, "qz"))
        bb = np.array([state.get(o, "bb_x"), state.get(o, "bb_y"), state.get(o, "bb_z")])
        ax = int(np.argmax(bb))
        u = R[:, ax]
        return c, u, bb[ax] / 2

    def _cup(self, state, name):
        o = state.get_object_from_name(name)
        return np.array([state.get(o, "x"), state.get(o, "y"), state.get(o, "z")])

    # ---------------- episode ----------------
    def reset(self, state, info):
        names = state.get_object_names()
        rods = sorted([n for n in names if n.startswith("cuboid_")],
                      key=lambda s: int(s.split("_")[1]))
        self.n = len(rods)
        goals = _load_goals(self.n)
        self.goal = {}
        for obj, cup, shelf, heights in goals:
            self.goal[obj] = (cup, shelf, heights)
        # default for rods without explicit goal: nothing
        self.rods = [r for r in rods if r in self.goal]
        self.attempts = {}
        self.plan = None
        self.staged = False
        self.gamma = 0.0
        cups = [n for n in names if n.startswith("cupboard")]
        self.front_x = min([self._cup(state, n)[0] for n in cups] or [2.0]) + CUP_FRONT_DX
        self.cur = None
        self.phase = "select"
        self.t = 0
        self.grip = 0.0
        base, q = self._read(state)
        self.base_t = base.copy()
        self.q_t = q.copy()

    def _region(self, state, rod):
        cup, shelf, heights = self.goal[rod]
        cp = self._cup(state, cup)
        zs = _shelf_surface_z(heights, shelf) + cp[2]
        lo = np.array([cp[0] - 0.15, cp[1] - 0.05, zs])
        hi = np.array([cp[0] + 0.15, cp[1] + 0.05, zs + heights[shelf]])
        return lo, hi

    def _in_region(self, state, rod, margin=0.0):
        lo, hi = self._region(state, rod)
        c = self._rod(state, rod)[0]
        return bool(np.all(c >= lo + margin) and np.all(c <= hi - margin))

    def _target_zone(self, state, rod):
        cup, shelf, heights = self.goal[rod]
        cp = self._cup(state, cup)
        zs = _shelf_surface_z(heights, shelf) + cp[2]
        top = zs
        lo, hi = self._region(state, rod)
        for r in self.rods:
            if r == rod or self.goal.get(r, (None, None))[:2] != (cup, shelf):
                continue
            c = self._rod(state, r)[0]
            if np.all(c >= lo - 0.02) and np.all(c <= hi + 0.02):
                top = max(top, c[2] + ROD_THICK / 2)
        rod_z = top + ROD_THICK / 2
        return cp, rod_z

    def _action(self, state):
        base, q = self._read(state)
        a = np.zeros(11)
        db = self.base_t - base
        db[2] = _wrap(db[2])
        n = np.linalg.norm(db[:2])
        if n > self.base_step:
            db[:2] *= self.base_step / n
        db[2] = np.clip(db[2], -self.base_yaw_step, self.base_yaw_step)
        a[0:3] = db
        dq = self.q_t - q
        for j in CONT_JOINTS:
            dq[j] = _wrap(dq[j])
        dq = np.clip(dq * (1.0 + ARM_LEAD), -0.6, 0.6)
        for j, lim in JLIM.items():
            dq[j] = np.clip(q[j] + dq[j], -lim + 0.01, lim - 0.01) - q[j]
        a[3:10] = dq
        a[10] = self.grip
        return a.astype(np.float32)

    def _arm_err(self, state):
        _, q = self._read(state)
        dq = self.q_t - q
        for j in CONT_JOINTS:
            dq[j] = _wrap(dq[j])
        return np.max(np.abs(dq))

    def _base_err(self, state):
        base, _ = self._read(state)
        d = self.base_t - base
        return np.linalg.norm(d[:2]), abs(_wrap(d[2]))

    def _set_phase(self, p):
        self.phase = p
        self.t = 0

    def get_action(self, state):
        self.base_step = 0.1
        self.base_yaw_step = 0.15
        try:
            self._step(state)
        except Exception:  # never crash: fall back to re-selecting
            self.phase, self.t, self.grip = "select", 0, 0.0
            try:
                self.base_t, self.q_t = self._read(state)[0], self.kin.canon(0.3)
            except Exception:
                pass
        self.t += 1
        try:
            return self._action(state)
        except Exception:
            return np.zeros(11, dtype=np.float32)

    # ---------------- state machine ----------------
    def _base_cost(self, state, b, yaw, rod):
        """Collision cost of a base pose (x, y, yaw)."""
        cy, sy = np.cos(yaw), np.sin(yaw)
        # base footprint corners (+ arm-plate) in world
        cost = 0.0
        hx0, hx1, hy = -0.16, 0.16, 0.26
        corners = []
        for lx in (hx0, hx1):
            for ly in (-hy, hy):
                corners.append((b[0] + cy * lx - sy * ly, b[1] + sy * lx + cy * ly))
        maxx = max(cx for cx, _ in corners)
        if maxx > self.front_x - 0.04:
            cost += 100.0 + 100 * (maxx - self.front_x)
        # other rods under the footprint (inflated)
        for r in self.rods:
            if r == rod:
                continue
            c, u, half = self._rod(state, r)
            if c[2] > 0.1:
                continue
            for t in np.linspace(-half, half, 7):
                pnt = c[:2] + t * u[:2] - b
                lx = cy * pnt[0] + sy * pnt[1]
                ly = -sy * pnt[0] + cy * pnt[1]
                if hx0 - 0.04 < lx < hx1 + 0.04 + 0.1 and abs(ly) < hy + 0.04:
                    cost += 3.0
        return cost

    def _grasp_plan(self, state, rod):
        """Choose (base pose, wrist yaw gamma) for grasping the rod."""
        c, u, half = self._rod(state, rod)
        u = np.array([u[0], u[1], 0.0])
        u /= max(np.linalg.norm(u), 1e-9)
        base, _ = self._read(state)
        best = None
        for s in (1.0, -1.0):
            uu = s * u
            off = GRIP_OFF
            if uu[0] > 0.3:
                need = (c[0] - (self.front_x - 0.045)) / uu[0]
                off = float(np.clip(need, GRIP_OFF, 0.14))
            g = c - off * uu  # grasp point (rear part), rod extends along +uu
            rod_yaw = np.arctan2(uu[1], uu[0])
            near = 0.0  # penalise grasp points next to other rods (double pick)
            for r in self.rods:
                if r == rod:
                    continue
                c2, u2, h2 = self._rod(state, r)
                d = g - c2
                t_ = np.clip(np.dot(d, u2), -h2, h2)
                dist = np.linalg.norm((d - t_ * u2)[:2])
                if dist < 0.07 and abs(c2[2] - c[2]) < 0.05:
                    near += 2.0 * (0.07 - dist) / 0.07 + 0.5
            for gamma in np.deg2rad([0, 30, -30, 60, -60, 90, -90]):
                yaw = _wrap(rod_yaw - gamma)
                b = g[:2] - REACH * np.array([np.cos(yaw), np.sin(yaw)])
                cost = (np.linalg.norm(b - base[:2]) + 0.2 * abs(_wrap(yaw - base[2]))
                        + 0.4 * abs(gamma) + self._base_cost(state, b, yaw, rod)
                        + near)
                if best is None or cost < best[0]:
                    best = (cost, np.array([b[0], b[1], yaw]), float(gamma))
        return best[1], best[2]

    def _held(self, state, rod):
        base, q = self._read(state)
        p, _ = self.kin.fk(q, base)
        c, _, _ = self._rod(state, rod)
        return np.linalg.norm(c - p) < 0.2

    def _arm_for_rod(self, state, rod, z_des):
        """Arm config that keeps the held rod horizontal with its center at
        height z_des (compensating for pitch slip in the grasp)."""
        base, q = self._read(state)
        p, R = self.kin.fk(q, base)
        c, u, _ = self._rod(state, rod)
        if np.dot(u, c - p) < 0:
            u = -u
        v = R.T @ u
        o = R.T @ (c - p)
        tau = float(np.arctan2(v[2], v[1]))
        tau = float(np.clip(tau, TILT - 0.5, TILT + 0.5))
        o2 = _R_tilt(tau) @ o
        pz = z_des - o2[2]
        pz = float(np.clip(pz, 0.03, 0.7))
        return self.kin.canon(pz, tau)

    def _step(self, state):
        kin = self.kin
        if self.phase == "select":
            todo = [r for r in self.rods if not self._in_region(state, r, 0.003)
                    and self.attempts.get(r, 0) < 5]
            if not todo:
                left = [r for r in self.rods if not self._in_region(state, r, 0.003)]
                if left:  # keep trying rather than idling
                    for r in left:
                        self.attempts[r] = 0
                self.q_t = kin.canon(0.3)
                return
            base, _ = self._read(state)
            todo.sort(key=lambda r: np.linalg.norm(self._rod(state, r)[0][:2] - base[:2]))
            self.cur = todo[0]
            self.plan = None
            self.staged = False
            self.attempts[self.cur] = self.attempts.get(self.cur, 0) + 1
            self.grip = 0.0
            self.q_t = kin.canon(0.3)
            self._set_phase("approach")
            return
        rod = self.cur
        c, u, _ = self._rod(state, rod)
        if self.phase == "approach":
            self.base_step = 0.15
            if self.t % 5 == 0 or self.plan is None:
                self.plan = self._grasp_plan(state, rod)
            self.base_t, self.gamma = self.plan[0].copy(), self.plan[1]
            # near the cupboards: first line up 0.25 behind, then drive straight in
            # (the extended arm must not sweep through rods sticking out of slots)
            hd = np.array([np.cos(self.base_t[2]), np.sin(self.base_t[2])])
            g_x = self.base_t[0] + REACH * hd[0]
            staging = g_x > self.front_x - 0.3 and not self.staged
            if staging:
                stage = self.base_t.copy()
                stage[:2] -= 0.25 * hd
                self.base_t = stage
                be, ye = self._base_err(state)
                if be < 0.03 and ye < 0.05:
                    self.staged = True
            # big base rotations only well away from the cupboards (arm sweep)
            base, _ = self._read(state)
            if abs(_wrap(self.base_t[2] - base[2])) > 0.2:
                self.base_t[0] = min(self.base_t[0], self.front_x - 0.85)
            self.q_t = kin.canon(c[2] + PINCH_ABOVE_ROD + 0.07, TILT, self.gamma)
            be, ye = self._base_err(state)
            if (staging or abs(_wrap(self.base_t[2] - base[2])) > 0.2) and self.t <= 80:
                return
            if (be < 0.01 and ye < 0.02 and self._arm_err(state) < 0.05) or self.t > 80:
                self._set_phase("descend")
            return
        if self.phase == "descend":
            self.q_t = kin.canon(c[2] + PINCH_ABOVE_ROD, TILT, self.gamma)
            if self._arm_err(state) < 0.02 or self.t > 40:
                self._set_phase("close")
            return
        if self.phase == "close":
            self.grip = 1.0
            if self.t > 3:
                self.grasp_z = c[2]
                base, _ = self._read(state)
                self.extract_x = base[0] - 0.3
                self._set_phase("extract" if c[2] > 0.025 else "lift")
            return
        if self.phase == "extract":
            # rod was resting in/on the cupboard: pull it straight back first
            base, _ = self._read(state)
            self.base_step = 0.03
            self.q_t = kin.canon(self.grasp_z + PINCH_ABOVE_ROD + 0.005, TILT, self.gamma)
            self.base_t = np.array([self.extract_x, base[1], base[2]])
            be, _ = self._base_err(state)
            if be < 0.02 or self.t > 25:
                self._set_phase("lift")
            return
        if self.phase == "lift":
            self.q_t = kin.canon(0.3, TILT, self.gamma)
            if (self._arm_err(state) < 0.05 or self.t > 30
                    or (c[2] > self.grasp_z + 0.08 and self.t > 4)):
                self._set_phase("carry")
            return
        cp, rod_z = self._target_zone(state, rod)
        front = cp[0] + CUP_FRONT_DX
        held = self._held(state, rod)
        if self.phase in ("carry", "insert", "lower") and not held:
            self._set_phase("select")
            self.grip = 0.0
            return
        if self.phase == "carry":
            self.base_yaw_step = 0.1
            if self.t < 3:
                self.carry_low = False
                self.q_t = kin.canon(0.3, TILT, 0.0)
                self.base_t = self._read(state)[0]
                return
            base, _ = self._read(state)
            aligned = abs(u[1]) < 0.05
            back = 0.0 if aligned else 0.25
            self.base_t = self._base_for_rod(
                state, rod, np.array([front - 0.2 - back - ROD_HALF, cp[1]]))
            be, ye = self._base_err(state)
            # travel high (clear of floor rods); lower only once in front of the slot
            z = rod_z + INSERT_DZ
            if aligned and be < 0.03:
                self.carry_low = True  # latch (avoid high/low oscillation)
            if not self.carry_low:
                z = max(z, 0.25)
            self.q_t = self._arm_for_rod(state, rod, z)
            if (aligned and be < 0.01 and ye < 0.02 and self._arm_err(state) < 0.03
                    and abs(u[2]) < 0.1) or self.t > 80:
                self._set_phase("insert")
            return
        if self.phase == "insert":
            # fast while the rod is still outside the cupboard
            self.base_step = 0.06 if c[0] + ROD_HALF < front - 0.08 else 0.03
            # start lowering once the rod's front end is inside the cupboard
            dz = INSERT_DZ if c[0] + ROD_HALF < front + 0.03 else 0.003
            self.q_t = self._arm_for_rod(state, rod, rod_z + dz)
            cx = front + 0.06
            self.base_t = self._base_for_rod(state, rod, np.array([cx, cp[1]]))
            base, _ = self._read(state)
            xcap = front + PINCH_MAX_DX - REACH * np.cos(base[2])
            self.base_t[0] = min(self.base_t[0], xcap)
            stuck = base[0] > xcap - 0.005 and c[0] > front + 0.03
            if abs(cx - c[0]) < 0.01 or stuck or self.t > 80:
                self._set_phase("lower")
            return
        if self.phase == "lower":
            self.q_t = self._arm_for_rod(state, rod, rod_z + 0.003)
            if (self._arm_err(state) < 0.02 and abs(c[2] - rod_z) < 0.01) or self.t > 25:
                self._set_phase("release")
            return
        if self.phase == "release":
            self.grip = 0.0
            if self.t > 3:
                self._set_phase("retreat")
            return
        if self.phase == "retreat":
            self.base_step = 0.15
            base, _ = self._read(state)
            self.base_t = np.array([front - 0.06 - GRIP_OFF - REACH - 0.15, base[1], 0.0])
            if self.t > 3:
                self.q_t = kin.canon(0.3)
            be, _ = self._base_err(state)
            if be < 0.03 or self.t > 30:
                self._set_phase("select")
            return

    def _base_for_rod(self, state, rod, c_des):
        """Base pose that puts the held rod's center at c_des (xy) with the rod
        axis along +x, assuming the rod is rigidly attached to the base."""
        c, u, _ = self._rod(state, rod)
        base, _ = self._read(state)
        uu = u if np.dot(u[:2], c[:2] - base[:2]) >= 0 else -u
        rod_yaw = np.arctan2(uu[1], uu[0])
        psi = _wrap(base[2] - rod_yaw)
        cb, sb = np.cos(-base[2]), np.sin(-base[2])
        d = c[:2] - base[:2]
        rel = np.array([cb * d[0] - sb * d[1], sb * d[0] + cb * d[1]])
        cp_, sp_ = np.cos(psi), np.sin(psi)
        off = np.array([cp_ * rel[0] - sp_ * rel[1], sp_ * rel[0] + cp_ * rel[1]])
        b = c_des - off
        return np.array([b[0], b[1], psi])
