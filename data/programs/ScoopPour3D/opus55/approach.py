"""Transfer cubes from source (yellow) bin to target (green) bin.

Strategy: pick cubes one at a time with a top-down grasp; the mobile base
does lateral transport (fast), the arm only moves in the base's x/z plane.
"""
import numpy as np

try:
    from . import kin  # type: ignore
except Exception:  # pragma: no cover
    import kin  # noqa

JOINTS = [f"pos_arm_joint{i}" for i in range(1, 8)]
RD = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # top-down, fingers close along world y

# gripper / grasp parameters
G_OPEN = 0.6          # open command during approach
GAP_HALF = 0.017      # half gap between finger pads at G_OPEN
FINGER_T = 0.010      # finger thickness (along closing axis)
FINGER_W = 0.013      # finger half-width (perpendicular to closing axis)
CUBE_R = 0.0075       # cube footprint radius
GRASP_DZ = -0.003     # tool z relative to cube center when grasping
CLOSE_STEPS = 4
HOVER_Z = 0.56
CARRY_Z = 0.58
BASE_X = -0.15


def Rz(t):
    return kin.rotz(t)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------------------------
    def _robot(self, state):
        R = state.get_object_from_name("robot")
        base = np.array([state.get(R, "pos_base_x"), state.get(R, "pos_base_y"),
                         state.get(R, "pos_base_rot")])
        q = np.array([state.get(R, j) for j in JOINTS])
        return base, q

    def _objs(self, state):
        cubes, bins = {}, {}
        for n in state.get_object_names():
            if n.startswith("cube"):
                o = state.get_object_from_name(n)
                cubes[n] = np.array([state.get(o, k) for k in ("x", "y", "z", "qw", "qx", "qy", "qz")])
            elif n.startswith("bin"):
                o = state.get_object_from_name(n)
                bins[n] = np.array([state.get(o, k) for k in ("x", "y", "z", "qw", "qx", "qy", "qz",
                                                               "bb_x", "bb_y", "bb_z")])
        return cubes, bins

    def reset(self, state, info):
        base, q = self._robot(state)
        self.q_int = q.copy()
        self.q_seed = q.copy()
        self.I = np.zeros(7)
        self.grip = G_OPEN
        self.t = 0
        cubes, bins = self._objs(state)
        self.src = None
        self.dst = None
        for n, b in bins.items():
            if "green" in n:
                self.dst = b
            elif "yellow" in n:
                self.src = b
        if self.dst is None:
            self.dst = list(bins.values())[0]
        self.phase = "select"
        self.cur = None
        self.pt = 0
        self.failed = {}
        self.yaw = 0.0
        self.tool_err = 0.0
        self.tool_p = np.zeros(3)
        self.tol = 0.01
        self.hold = None
        self.zhist = []

    # ------------------------------------------------------------------
    def _in_bin(self, p, b, margin=0.0):
        return (abs(p[0] - b[0]) < b[7] / 2 - margin and abs(p[1] - b[1]) < b[8] / 2 - margin)

    def _bin_of(self, p):
        for b in (self.src, self.dst):
            if b is not None and self._in_bin(p, b):
                return b
        return None

    def _grasp_score(self, n, p, cubes, yaw):
        """Clearance score of grasping cube n with gripper yaw (closing axis angle)."""
        u = np.array([-np.sin(yaw), np.cos(yaw)])  # closing axis (world y at yaw 0)
        v = np.array([np.cos(yaw), np.sin(yaw)])
        clear = 0.05
        for m, p2 in cubes.items():
            if m == n:
                continue
            d = p2[:2] - p[:2]
            if abs(p2[2] - p[2]) > 0.03:
                continue
            du, dv = abs(d @ u), abs(d @ v)
            # finger slab: |u| in [GAP_HALF, GAP_HALF+FINGER_T], |v| < FINGER_W
            if dv < FINGER_W + CUBE_R:
                if du < GAP_HALF - CUBE_R:
                    # inside the gripper gap: would be captured / interfere
                    c = max(du - 0.012, -0.01) if du > 0.012 else -0.02
                    clear = min(clear, c)
                elif du < GAP_HALF + FINGER_T + CUBE_R:
                    clear = min(clear, -0.03)
                else:
                    clear = min(clear, du - (GAP_HALF + FINGER_T + CUBE_R))
            else:
                clear = min(clear, dv - (FINGER_W + CUBE_R) + max(0.0, du - GAP_HALF - FINGER_T - CUBE_R))
        # walls of containing bin
        b = self._bin_of(p)
        if b is not None:
            hx, hy = b[7] / 2 - 0.01, b[8] / 2 - 0.01
            for su in (-1, 1):
                for sv in (-1, 1):
                    corner = p[:2] + su * (GAP_HALF + FINGER_T) * u + sv * FINGER_W * v
                    mx = hx - abs(corner[0] - b[0])
                    my = hy - abs(corner[1] - b[1])
                    clear = min(clear, mx + 0.01, my + 0.01)
        return clear

    def _select(self, cubes):
        best, bs, byaw = None, -1e9, 0.0
        for n, p in cubes.items():
            if self._in_bin(p, self.dst):
                continue
            cy = 2 * np.arctan2(p[6], p[3])
            face = (cy + np.pi / 4) % (np.pi / 2) - np.pi / 4
            cands = []
            for k in range(-4, 5):
                for off in (0.0, 0.25, -0.25, 0.45, -0.45):
                    y = face + k * np.pi / 2 + off
                    if abs(y) > 1.6 or abs(y - self.yaw) > np.pi / 2 + 0.01:
                        continue
                    cands.append((y, off))
            for y, off in cands:
                s = self._grasp_score(n, p, cubes, y)
                s = min(s, 0.03) - 0.02 * abs(off) - 0.012 * abs(y - self.yaw) - 0.002 * abs(y)
                s -= 0.03 * self.failed.get(n, 0)
                s -= 1.0 * max(0.0, p[2] - 0.495)
                if s > bs:
                    bs, best, byaw = s, n, y
        return best, byaw

    def _arm_target(self, base_t, p_world, R_world):
        parm = kin.world_to_arm(p_world, base_t)
        Rarm = Rz(-base_t[2]) @ R_world
        q, err = kin.ik_arm(parm, Rarm, self.q_seed, iters=100, lam=0.05)
        self.q_seed = q
        return q

    def _step_to(self, base, q, base_t, p_world, R_world, vmax=0.1, integ=True):
        a = np.zeros(11, dtype=np.float32)
        d = np.asarray(base_t, float) - base
        d[2] = wrap(d[2])
        cmd = np.array([d[0] / 0.87, d[1] / 0.87, d[2] / 0.99])
        a[0:3] = np.clip(cmd, -0.1, 0.1)
        qt = self._arm_target(np.asarray(base_t, float), p_world, R_world)
        # integral correction for steady-state tracking error
        if integ and np.max(np.abs(qt + self.I - self.q_int)) < 0.05:
            e = qt - q
            e = np.where(np.abs(e) < 0.1, e, 0.0)
            self.I = np.clip(self.I + 0.4 * e, -0.1, 0.1)
        qc = qt + self.I
        dq = np.clip((qc - self.q_int) / 0.25, -vmax, vmax)
        a[3:10] = dq
        self.q_int = self.q_int + 0.25 * dq
        a[10] = self.grip
        pw, _ = kin.fk_world(base, q)
        self.tool_p = pw
        self.tool_err = np.linalg.norm(pw - np.asarray(p_world))
        arm_done = np.max(np.abs(qc - self.q_int)) < 1e-3 and self.tool_err < self.tol
        base_done = np.all(np.abs(d[:2]) < 0.006) and abs(d[2]) < 0.02
        return a, arm_done and base_done

    def _set(self, phase):
        self.phase = phase
        self.pt = 0

    def get_action(self, state):
        try:
            a = self._get_action(state)
            a = np.asarray(a, dtype=np.float32)
            if not np.all(np.isfinite(a)):
                raise ValueError("non-finite action")
            return a
        except Exception:
            a = np.zeros(11, dtype=np.float32)
            a[10] = getattr(self, "grip", 0.0)
            try:
                self.phase = "select"
                self.q_seed = self._robot(state)[1].copy()
                self.q_int = self.q_seed.copy()
                self.I = np.zeros(7)
            except Exception:
                pass
            return a

    def _get_action(self, state):
        base, q = self._robot(state)
        cubes, bins = self._objs(state)
        self.t += 1
        self.pt += 1
        if self.phase == "select":
            self.cur, self.yaw = self._select(cubes)
            if self.cur is None:
                a = np.zeros(11, dtype=np.float32)
                a[10] = self.grip
                return a
            self._set("approach")
            self.pt = 1
        c = cubes[self.cur]
        self.tol = {"approach": 0.015, "descend": 0.006, "close": 0.01, "lift": 0.02,
                    "carry": 0.02, "release": 0.02}.get(self.phase, 0.01)
        Rg = Rz(self.yaw) @ RD
        if self.phase == "approach":
            wide = self.failed.get(self.cur, 0) % 2 == 1
            self.grip = 0.0 if wide else G_OPEN
            bt = np.array([BASE_X, c[1], 0.0])
            a, done = self._step_to(base, q, bt, np.array([c[0], c[1], HOVER_Z]), Rg)
            xy_ok = np.linalg.norm(self.tool_p[:2] - c[:2]) < 0.01 and abs(self.tool_p[2] - HOVER_Z) < 0.03
            if done or (xy_ok and abs(base[1] - c[1]) < 0.006) or self.pt > 60:
                self._set("descend")
                self.hold = np.array([c[0], c[1], c[2] + (-0.006 if wide else GRASP_DZ)])
                self.zhist = []
            return a
        if self.phase == "descend":
            bt = np.array([BASE_X, self.hold[1], 0.0])
            pt = self.hold.copy()
            fast = self.tool_p[2] > self.hold[2] + 0.025
            if fast:
                pt[2] = self.hold[2] + 0.01
            a, done = self._step_to(base, q, bt, pt, Rg, vmax=0.1 if fast else 0.06)
            done = done and not fast
            self.zhist.append(self.tool_p[2])
            blocked = (len(self.zhist) > 6 and self.zhist[-6] - self.zhist[-1] < 0.003
                       and self.tool_p[2] > self.hold[2] + 0.02)
            if blocked:
                self.failed[self.cur] = self.failed.get(self.cur, 0) + 1
                self.hold = np.array([self.hold[0], self.hold[1], HOVER_Z])
                self._set("retreat")
                return a
            stalled = (len(self.zhist) > 5 and self.zhist[-5] - self.zhist[-1] < 0.0015
                       and self.tool_p[2] < self.hold[2] + 0.015)
            if done or stalled or self.pt > 30:
                self._set("close")
            return a
        if self.phase == "close":
            self.grip = 1.0
            bt = np.array([BASE_X, self.hold[1], 0.0])
            a, done = self._step_to(base, q, bt, self.hold, Rg, integ=False)
            if self.pt >= CLOSE_STEPS:
                self._set("lift")
            return a
        if self.phase in ("lift", "retreat"):
            p = self.hold.copy()
            p[2] = HOVER_Z
            a, done = self._step_to(base, q, np.array([BASE_X, self.hold[1], 0.0]), p, Rg)
            if self.phase == "lift" and self.pt >= 5:
                # grasp check: cube should rise with the tool
                if self.tool_p[2] - self.hold[2] > 0.03 and not self._lifted(cubes):
                    self.failed[self.cur] = self.failed.get(self.cur, 0) + 1
                    self._set("retreat")
                    self.grip = G_OPEN
                    return a
            if (self.tool_p[2] > HOVER_Z - 0.012) or self.pt > 30:
                if self.phase == "lift" and self._lifted(cubes, 0.515):
                    self._set("carry")
                else:
                    if self.phase == "lift":
                        self.failed[self.cur] = self.failed.get(self.cur, 0) + 1
                    self.grip = G_OPEN
                    self._set("select")
            return a
        if self.phase in ("carry", "release"):
            tgt = self._drop_point()
            bt = np.array([BASE_X, tgt[1], 0.0])
            p = np.array([tgt[0], tgt[1], CARRY_Z])
            if self.phase == "carry":
                if self.tool_p[2] < 0.545 and self.pt < 10:
                    bt = np.array([BASE_X, self.hold[1], 0.0])
                    p = np.array([self.hold[0], self.hold[1], CARRY_Z])
                a, done = self._step_to(base, q, bt, p, Rg)
                if abs(base[1] - bt[1]) < 0.02 or self.pt > 30:
                    self._set("release")
                return a
            self.grip = G_OPEN
            a, done = self._step_to(base, q, bt, p, Rg)
            if self.pt >= 2:
                self._set("select")
            return a
        a = np.zeros(11, dtype=np.float32)
        a[10] = self.grip
        return a

    def _lifted(self, cubes, zmin=None):
        tp = self.tool_p
        for p in cubes.values():
            if np.linalg.norm(p[:2] - tp[:2]) < 0.05 and p[2] > tp[2] - 0.03 and (zmin is None or p[2] > zmin):
                return True
        return False

    def _drop_point(self):
        b = self.dst
        return np.array([b[0], b[1], b[2]])
