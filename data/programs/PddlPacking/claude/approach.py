"""PR2Packed approach: pick every block from the table and pack them on the plate."""
import numpy as np
import fk

CARRY_Z = 0.95
PLACE_Z = 1.05
GRASP_CLR = 0.05          # tool offset above block centre along -approach
NOTCH_X = -0.43
WIDE_X = -0.66
PRE_BACK = 0.13           # pre-grasp standoff along approach axis

SEEDS = [
    np.array([0.6772, -0.3431, 1.2, -1.4669, 1.2422, -1.9544, 2.2225]),
    np.array([0.3, -0.4, 1.3, -0.7, 1.5, -2.0, 3.3]),
    np.array([1.0, 0.0, 1.5, -1.2, 1.0, -1.5, 1.5]),
    np.array([0.0, 0.0, 1.5, -1.0, 0.0, -1.5, 0.0]),
    np.array([-0.4, 0.3, 0.6, -1.8, 1.0, -1.2, 1.0]),
]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def yaw_of(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def quat_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def align_angle(yaw, ref):
    best, bd = ref, 1e9
    for k in range(-4, 5):
        c = yaw + k * (np.pi / 2)
        d = abs(wrap(c - ref))
        if d < bd:
            bd, best = d, c
    return best


def tool_R(ang, tilt=0.0, tiltdir=0.0):
    """Tool frame: x = approach axis, y = finger axis."""
    x = np.array([np.sin(tilt) * np.cos(tiltdir), np.sin(tilt) * np.sin(tiltdir),
                  -np.cos(tilt)])
    y = np.array([np.cos(ang), np.sin(ang), 0.0])
    y = y - x * float(np.dot(x, y))
    n = np.linalg.norm(y)
    if n < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
        y = y - x * float(np.dot(x, y))
        n = np.linalg.norm(y)
    y /= n
    return np.column_stack([x, y, np.cross(x, y)])


ROTX_PI = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
BLOCK_ORIENTS = [rotz(k * np.pi / 2) for k in range(4)] + \
                [rotz(k * np.pi / 2) @ ROTX_PI for k in range(4)]


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def wrap_joints(d):
    d = np.array(d, dtype=float)
    d[4] = wrap(d[4])
    d[6] = wrap(d[6])
    return d


class Robot:
    RF = ["base_x", "base_y", "base_rot", "joint_1", "joint_2", "joint_3",
          "joint_4", "joint_5", "joint_6", "joint_7", "gripper_opening",
          "grasp_active"]
    BF = ["pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz",
          "pose_qw", "grasp_active", "half_extent_x", "half_extent_y",
          "half_extent_z"]

    def __init__(self, state):
        r = state.get_object_from_name("robot")
        self.r = np.array([state.get(r, f) for f in self.RF])
        self.blocks = {}
        for n in sorted(state.get_object_names()):
            if n.startswith("block"):
                b = state.get_object_from_name(n)
                self.blocks[n] = np.array([state.get(b, f) for f in self.BF])
        p = state.get_object_from_name("plate")
        self.plate = np.array([state.get(p, f) for f in
                               ["pose_x", "pose_y", "pose_z", "half_extent_x",
                                "half_extent_y", "half_extent_z"]])

    @property
    def base(self):
        return self.r[:3]

    @property
    def q(self):
        return self.r[3:10]

    @property
    def holding(self):
        return self.r[11] > 0.5

    def held_name(self):
        for n, b in self.blocks.items():
            if b[7] > 0.5:
                return n
        return None

    def tool(self):
        return fk.fk_world(self.r[:3], self.q)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------ planning
    def base_candidates(self):
        c = [(NOTCH_X, y, 0.0) for y in np.linspace(-0.21, 0.21, 8)]
        c += [(WIDE_X, y, 0.0) for y in np.linspace(-0.9, 0.9, 19)]
        return c

    def prefilter(self, base, pw, Rw=None):
        bx, by, rot = base
        Rb = rotz(rot)
        sh = np.array([bx, by, 0.0]) + Rb @ np.array([-0.05, 0.188, 0.990775])
        w = np.asarray(pw, dtype=float)
        if Rw is not None:
            w = w - fk.TOOL[0] * Rw[:, 0]
        d = w - sh
        dh = float(np.hypot(d[0], d[1]))
        r = np.hypot(max(dh - 0.1, 0.0), d[2])
        if r > 0.716 or r < 0.09:
            return False
        loc = Rb.T @ (np.asarray(pw) - np.array([bx, by, 0.0]))
        if loc[0] < 0.05:
            return False
        ang = np.arctan2(d[1], d[0]) - rot
        return -1.0 < wrap(ang) < 2.5

    def ik_at(self, base, pw, Rw, seeds=None, iters=90):
        if not self.prefilter(base, pw, Rw):
            return None
        bx, by, rot = base
        Rb = rotz(rot)
        tp = Rb.T @ (np.asarray(pw) - np.array([bx, by, 0.0]))
        Rl = Rb.T @ Rw
        for sd in (seeds or SEEDS):
            q, ok = fk.ik(tp, Rl, sd, iters=iters)
            if ok:
                return q
        return None

    def base_cost(self, cur, cand):
        return abs(cand[0] - cur[0]) + abs(cand[1] - cur[1]) + 0.3 * abs(wrap(cand[2] - cur[2]))

    def grasp_poses(self, b):
        """Yield (tool_pos, R_world) grasp candidates for a block, best first."""
        byaw = yaw_of(*b[3:7])
        angs = [align_angle(byaw, np.pi / 2), align_angle(byaw, 0.0)]
        out = []
        bd0 = np.arctan2(b[1], b[0] + 0.43)
        for tilt, tdirs in ((0.0, [0.0]), (0.32, None), (0.5, None), (0.7, None)):
            dirs = tdirs if tdirs is not None else [
                bd0, np.pi, bd0 - 0.6, bd0 + 0.6, 0.0,
                -np.pi / 2 if b[1] < 0 else np.pi / 2]
            for td in dirs:
                for ang in angs:
                    Rw = tool_R(ang, tilt, td)
                    gp = np.array([b[0], b[1], b[2]]) - GRASP_CLR * Rw[:, 0]
                    out.append((gp, Rw))
        return out

    def plan_pick(self, R, name, exclude=()):
        b = R.blocks[name]
        cands = [(i, c) for i, c in enumerate(self.grasp_poses(b))
                 if i not in exclude]
        bases = sorted(self.base_candidates(), key=lambda c: self.base_cost(R.base, c))
        # prefer vertical grasps from any base before tilted ones
        for k in range(3):
            group = [c for c in cands if abs(c[1][1][2, 0] + 1.0) < 1e-9] if k == 0 else cands
            for base in bases:
                for idx, (gp, Rw) in group:
                    pre = gp - PRE_BACK * Rw[:, 0]
                    pre[2] = max(pre[2], CARRY_Z)
                    if self.ik_at(base, gp, Rw) is None:
                        continue
                    qpre = self.ik_at(base, pre, Rw)
                    if qpre is None:
                        continue
                    return base, gp, Rw, pre, qpre, idx
            if k == 0:
                continue
            break
        return None

    def plate_reachable(self, base, cell):
        pw = np.array([cell[0], cell[1], CARRY_Z])
        for ang in (np.pi / 2, 0.0):
            if self.ik_at(base, pw, tool_R(ang)) is not None:
                return True
        return False

    def plan_place_base_far(self, R, cell):
        best = None
        for base in self.base_candidates():
            if self.base_cost(R.base, base) < 0.08:
                continue
            if self.plate_reachable(base, cell):
                c = self.base_cost(R.base, base)
                if best is None or c < best[0]:
                    best = (c, base)
        return None if best is None else best[1]

    def plan_place_base(self, R, cell):
        for base in sorted(self.base_candidates(), key=lambda c: self.base_cost(R.base, c)):
            if self.plate_reachable(base, cell):
                return base
        return None

    def base_path(self, cur, dst):
        wps = []
        if abs(dst[1] - cur[1]) > 0.02:
            rx = min(cur[0], dst[0], -0.78)
            if cur[0] > rx + 1e-6:
                wps.append((rx, cur[1], dst[2]))
            wps.append((rx, dst[1], dst[2]))
        wps.append(tuple(dst))
        return wps

    # ------------------------------------------------ cells
    def make_cells(self, R):
        n = len(R.blocks)
        px, py = R.plate[0], R.plate[1]
        hx, hy = R.plate[3], R.plate[4]
        bh = 0.035
        for b in R.blocks.values():
            bh = max(bh, b[8], b[9])
        diag = 2.0 * bh * np.sqrt(2.0)
        # loose (rotation free) layouts: block yaw can stay arbitrary
        loose = None
        USE_LOOSE = False
        if n == 1:
            loose = [(px, py)]
        elif n == 2 and hy - 0.5 * (diag + 0.015) - 0.5 * diag > -1e-9:
            d = 0.5 * (diag + 0.015)
            loose = [(px, py - d), (px, py + d)]
        elif n <= 4:
            d = 0.5 * (diag + 0.015)
            if hx - d - 0.5 * diag > -1e-9 and hy - d - 0.5 * diag > -1e-9:
                cs = [(px - d, py - d), (px + d, py - d), (px - d, py + d),
                      (px + d, py + d)]
                loose = cs[:n] if n == 4 else cs[:n]
        if loose is not None and USE_LOOSE:
            self.free_yaw = True
            return loose
        self.free_yaw = False
        return self.make_grid_cells(R)

    def make_grid_cells(self, R):
        px, py, hx, hy = R.plate[0], R.plate[1], R.plate[3], R.plate[4]
        bh = 0.035
        for b in R.blocks.values():
            bh = max(bh, b[8], b[9])
        n = len(R.blocks)
        pitch = 2 * bh + 0.018
        kx = max(1, int(np.floor((2 * hx - 2 * bh) / pitch)) + 1)
        ky = max(1, int(np.floor((2 * hy - 2 * bh) / pitch)) + 1)
        offs_x = [(i - (kx - 1) / 2.0) * pitch for i in range(kx)]
        offs_y = [(i - (ky - 1) / 2.0) * pitch for i in range(ky)]
        if n <= ky:
            cells = [(px, py + o) for o in offs_y]
        else:
            cells = [(px + ox, py + oy) for oy in offs_y for ox in offs_x]
        cells.sort(key=lambda c: (-(c[0] - px), abs(c[1] - py)))
        return cells

    def on_plate(self, R, b):
        px, py, hx, hy = R.plate[0], R.plate[1], R.plate[3], R.plate[4]
        return (abs(b[0] - px) <= hx - b[8] + 1e-6 and
                abs(b[1] - py) <= hy - b[9] + 1e-6)

    # ------------------------------------------------ episode
    def reset(self, state, info):
        R = Robot(state)
        self.cells = self.make_cells(R)
        self.phase = "select"
        self.prev_r = None
        self.last_action = None
        self.target = None
        self.stuck = 0
        self.base_wp = []
        self.grasp_tries = 0
        self.cell_try = 0
        self.failed = set()
        self.grasp_excl = {}
        self.step_i = 0
        self.Rgrasp = None
        self.gp = None
        self.pre = None
        self.cell = None
        self.place_fail = 0
        self.arm_target = None

    # ------------------------------------------------ low level
    def act_arm(self, dq, grip=0.0):
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = np.clip(dq, -0.2, 0.2)
        a[10] = grip
        return a

    def act_base(self, db):
        a = np.zeros(11, dtype=np.float32)
        a[:3] = np.clip(db, -0.2, 0.2)
        return a

    def cart_step(self, R, pw, Rw, step_len=0.10, ptol=3e-3, rtol=1e-2,
                  rot_step=0.25):
        base = R.base
        Rb = rotz(base[2])
        tp = Rb.T @ (np.asarray(pw) - np.array([base[0], base[1], 0.0]))
        Rl = Rb.T @ Rw
        q = R.q
        p, Rc, _ = fk.fk_arm(q)
        ep = tp - p
        er = fk.so3_error(Rc, Rl)
        dist = float(np.linalg.norm(ep))
        if dist < ptol and np.linalg.norm(er) < rtol:
            return None, True
        sub = p + ep * min(1.0, step_len / max(dist, 1e-9))
        rn = float(np.linalg.norm(er))
        Rsub = Rl if rn < rot_step else rodrigues(er * (rot_step / rn)) @ Rc
        qd, ok = fk.ik(sub, Rsub, q, iters=60)
        dq = wrap_joints(qd - q)
        if np.abs(dq).max() < 1e-7:
            return None, False
        if np.linalg.norm(dq) > 0.9:
            # local Jacobian fallback: avoid branch jumps
            J = fk.jacobian(q)
            e6 = np.concatenate([sub - p, er * min(1.0, rot_step / max(rn, 1e-9))])
            lam = 0.08
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e6)
            dq = wrap_joints(dq)
            if np.abs(dq).max() < 1e-7:
                return None, False
        return dq, False

    # ------------------------------------------------ main
    def get_action(self, state):
        R = Robot(state)
        self.step_i += 1
        rejected = False
        if self.prev_r is not None and self.last_action is not None:
            if np.allclose(self.prev_r[:10], R.r[:10], atol=1e-9) and \
                    np.abs(self.last_action[:10]).max() > 1e-9:
                rejected = True
        self.prev_r = R.r.copy()
        if rejected:
            self.rej_run = getattr(self, "rej_run", 0) + 1
        else:
            self.rej_run = 0
        if getattr(self, "escape", 0) > 0 or self.rej_run >= 8:
            return self.do_escape(R)
        act = None
        for _ in range(8):
            act = self.dispatch(R, rejected)
            rejected = False
            if act is not None:
                break
        if act is None:
            act = np.zeros(11, dtype=np.float32)
        self.last_action = np.asarray(act, dtype=np.float64)
        return np.asarray(act, dtype=np.float32)

    def do_escape(self, R):
        """Everything is being rejected: shake loose and replan."""
        if getattr(self, "escape", 0) <= 0:
            self.escape = 9
            self.rng = getattr(self, "rng", np.random.default_rng(0))
        self.escape -= 1
        a = np.zeros(11, dtype=np.float32)
        k = self.escape
        if k >= 5:
            a[0] = -0.2
            a[1] = 0.1 if k % 2 else -0.1
        else:
            a[3:10] = self.rng.uniform(-0.18, 0.18, 7)
        if self.escape <= 0:
            self.phase = "select"
            self.stuck = 0
            self.rej_run = 0
            self._pkey = None
            self.grasp_excl = {}
            self.failed = set()
        self.last_action = np.asarray(a, dtype=np.float64)
        return a

    def dispatch(self, R, rejected):
        ph = self.phase
        if ph == "select":
            return self.ph_select(R)
        if ph == "base":
            return self.ph_base(R, rejected)
        if ph == "pregrasp":
            return self.ph_cart(R, self.pre, self.Rgrasp, "descend", rejected,
                                step_len=0.16)
        if ph == "descend":
            return self.ph_cart(R, self.gp, self.Rgrasp, "close", rejected,
                                step_len=0.14, ptol=2e-3)
        if ph == "close":
            return self.ph_close(R)
        if ph == "lift":
            return self.ph_cart(R, self.lift_pt, self.Rgrasp, "place_base",
                                rejected, step_len=0.16)
        if ph == "place_base":
            return self.ph_place_base(R, rejected)
        if ph == "place":
            return self.ph_place(R, rejected)
        if ph == "goto_q":
            return self.ph_goto_q(R, rejected)
        if ph == "open":
            return self.ph_open(R)
        return np.zeros(11, dtype=np.float32)

    def ph_select(self, R):
        held = R.held_name()
        if held is not None:
            self.target = held
            self.pick_cell(R)
            self.phase = "place_base"
            self.stuck = 0
            return None
        todo = [n for n, b in R.blocks.items()
                if not self.on_plate(R, b) and n not in self.failed]
        if not todo:
            self.phase = "done"
            return np.zeros(11, dtype=np.float32)
        todo.sort(key=lambda n: abs(R.blocks[n][1] - R.base[1]) +
                  abs(R.blocks[n][0] - R.base[0]))
        self.target = todo[0]
        excl = self.grasp_excl.get(self.target, set())
        plan = self.plan_pick(R, self.target, excl)
        if plan is None:
            self.failed.add(self.target)
            return None
        base, self.gp, self.Rgrasp, self.pre, qpre, self.grasp_idx = plan
        self.arm_target = qpre
        near = 9.9
        for n2, b2 in R.blocks.items():
            if n2 == self.target or b2[7] > 0.5:
                continue
            near = min(near, float(np.hypot(b2[0] - self.gp[0], b2[1] - self.gp[1])))
        lz = PLACE_Z if near < 0.22 else 0.93
        lp = np.array([self.pre[0], self.pre[1], max(self.pre[2], lz)])
        while lp[2] > self.pre[2] + 0.01:
            if self.ik_at(base, lp, self.Rgrasp) is not None:
                break
            lp[2] -= 0.05
        self.lift_pt = lp
        self.pick_cell(R)
        self.grasp_tries = 0
        self.stuck = 0
        self.base_wp = self.base_path(R.base, base)
        self.phase = "base"
        self.next_after_base = "pregrasp"
        return None

    def pick_cell(self, R):
        b = R.blocks[self.target]
        used = [(bb[0], bb[1]) for n, bb in R.blocks.items()
                if n != self.target and bb[7] < 0.5 and self.on_plate(R, bb)]
        free = [c for c in self.cells
                if all(max(abs(c[0] - u[0]), abs(c[1] - u[1])) > 2 * b[8] + 0.012
                       for u in used)]
        if not free:
            free = list(self.cells)
        free.sort(key=lambda c: abs(c[1] - b[1]) + abs(c[0] - b[0]))
        self.cell = free[0]
        self.cell_try = 0
        self.place_fail = 0

    def ph_base(self, R, rejected):
        if rejected:
            self.stuck += 1
            if self.stuck > 5:
                self.stuck = 0
                self.base_wp = []
                self.phase = self.next_after_base
                return None
            cur = R.base
            self.base_wp = [(min(cur[0], -0.85), cur[1], cur[2])] + self.base_wp
        while self.base_wp:
            d = np.array(self.base_wp[0]) - R.base
            d[2] = wrap(d[2])
            if np.abs(d).max() < 1e-4:
                self.base_wp.pop(0)
                continue
            a = self.act_base(d)
            qt = getattr(self, "arm_target", None)
            if qt is not None and self.stuck == 0:
                a[3:10] = np.clip(wrap_joints(qt - R.q), -0.2, 0.2)
            return a
        self.phase = self.next_after_base
        self.stuck = 0
        return None

    def progress(self, R, pw, Rw, key):
        p, Rc, _ = R.tool()
        e = float(np.linalg.norm(np.asarray(pw) - p)) + \
            0.15 * float(np.linalg.norm(fk.so3_error(Rc, Rw)))
        if getattr(self, "_pkey", None) != key:
            self._pkey = key
            self._perr = e
            self._pcnt = 0
            return False
        if e < self._perr - 2e-3:
            self._perr = e
            self._pcnt = 0
            return False
        self._pcnt += 1
        return self._pcnt >= 4

    def will_converge(self, R, dq, pw, Rw, ptol):
        q = R.q + np.clip(dq, -0.2, 0.2)
        p, Rc, _ = fk.fk_arm(q)
        base = R.base
        Rb = rotz(base[2])
        tp = Rb.T @ (np.asarray(pw) - np.array([base[0], base[1], 0.0]))
        Rl = Rb.T @ Rw
        return (np.linalg.norm(tp - p) < ptol and
                np.linalg.norm(fk.so3_error(Rc, Rl)) < 1.2e-2)

    def ph_cart(self, R, pw, Rw, nxt, rejected, step_len=0.10, ptol=3e-3):
        if rejected:
            self.stuck += 1
            if self.stuck > 12:
                self.stuck = 0
                self.phase = nxt
                return None
            step_len = 0.03 if self.stuck < 6 else 0.015
        dq, conv = self.cart_step(R, pw, Rw, step_len=step_len, ptol=ptol)
        if conv:
            self.phase = nxt
            self.stuck = 0
            self._pkey = None
            return None
        stalled = self.progress(R, pw, Rw, ("cart", self.phase, self.target))
        if dq is None or stalled:
            self.stuck = 0
            self._pkey = None
            return self.recover(R, pw, Rw, nxt)
        if nxt == "close" and self.grasp_tries == 0 and \
                self.will_converge(R, dq, pw, Rw, ptol):
            self.phase = "close"
            self.grasp_tries = 1
            self._pkey = None
            self.stuck = 0
            return self.act_arm(dq, grip=-1.0)
        return self.act_arm(dq)

    def recover(self, R, pw, Rw, nxt, replan_base=False):
        """Cartesian servo stalled: try a joint space jump to a full IK solution."""
        q = self.ik_at(tuple(R.base), pw, Rw, iters=200)
        if q is not None and np.abs(wrap_joints(q - R.q)).max() > 1e-4:
            self.goto_q = q
            self.after_goto = nxt
            self.phase = "goto_q"
            self.stuck = 0
            return None
        if replan_base:
            base = self.plan_place_base_far(R, self.cell)
            if base is not None and self.base_cost(R.base, base) > 0.02:
                self.base_wp = self.base_path(R.base, base)
                self.next_after_base = "place"
                self.phase = "base"
                return None
        self.phase = nxt
        return None

    def place_recover(self, R, tgt, Rdes):
        self.place_fail = getattr(self, "place_fail", 0) + 1
        if self.place_fail <= 1:
            q = self.ik_at(tuple(R.base), tgt, Rdes, iters=200)
            if q is not None and np.abs(wrap_joints(q - R.q)).max() > 1e-4:
                self.goto_q = q
                self.after_goto = "place"
                self.phase = "goto_q"
                return None
        if self.place_fail <= 3:
            base = self.plan_place_base_far(R, self.cell)
            if base is not None:
                self.base_wp = self.base_path(R.base, base)
                self.next_after_base = "place"
                self.phase = "base"
                return None
        self.next_cell(R)
        self.place_fail = 0
        self.phase = "place"
        return None

    def ph_goto_q(self, R, rejected):
        if rejected:
            self.stuck += 1
            if self.stuck > 6:
                self.stuck = 0
                self.phase = self.after_goto
                return None
        d = wrap_joints(self.goto_q - R.q)
        if np.abs(d).max() < 1e-4:
            self.phase = self.after_goto
            self.stuck = 0
            return None
        sc = 1.0 if self.stuck < 3 else 0.25
        return self.act_arm(d * sc)

    def ph_close(self, R):
        if R.holding:
            b = R.blocks[R.held_name()]
            self.blk_R0 = quat_R(b[3:7])
            self.target = R.held_name()
            self.phase = "lift"
            self.stuck = 0
            return None
        self.grasp_tries += 1
        if self.grasp_tries > 2:
            self.grasp_excl.setdefault(self.target, set()).add(self.grasp_idx)
            if len(self.grasp_excl[self.target]) > 12:
                self.failed.add(self.target)
            self.grasp_tries = 0
            self.phase = "select"
            a = np.zeros(11, dtype=np.float32)
            a[10] = 1.0
            return a
        if self.grasp_tries > 1:
            self.gp = self.gp - 0.012 * self.Rgrasp[:, 0]
            self.phase = "descend"
            a = np.zeros(11, dtype=np.float32)
            a[10] = 1.0
            return a
        a = np.zeros(11, dtype=np.float32)
        a[10] = -1.0
        return a

    def ph_place_base(self, R, rejected):
        if self.plate_reachable(R.base, self.cell):
            self.phase = "place"
            self.stuck = 0
            return None
        base = self.plan_place_base(R, self.cell)
        if base is None:
            self.phase = "place"
            return None
        self.base_wp = self.base_path(R.base, base)
        self.next_after_base = "place"
        self.arm_target = None
        self.phase = "base"
        return None

    def ph_place(self, R, rejected):
        if not R.holding:
            self.phase = "select"
            return None
        b = R.blocks[self.target]
        p_tool, Rw, _ = R.tool()
        Rblk = quat_R(b[3:7])
        Rrel = Rw.T @ Rblk
        # desired tool rotation so that the block ends up level and axis aligned
        if getattr(self, "free_yaw", False) and getattr(self, "blk_R0", None) is not None:
            Rdes = self.blk_R0 @ Rrel.T
            bd = float(np.linalg.norm(fk.so3_error(Rw, Rdes)))
        else:
          best, bd = None, 1e9
          for Rb_des in BLOCK_ORIENTS:
            Rd = Rb_des @ Rrel.T
            d = np.linalg.norm(fk.so3_error(Rw, Rd))
            if d < bd:
                bd, best = d, Rd
          Rdes = best
        off_t = Rw.T @ (np.array([b[0], b[1], b[2]]) - p_tool)
        off_w = Rdes @ off_t
        pz = PLACE_Z
        tgt = np.array([self.cell[0] - off_w[0], self.cell[1] - off_w[1], pz])
        err = np.linalg.norm(np.array([b[0], b[1]]) - np.array(self.cell))
        if err < 0.005 and bd < 0.02:
            self.phase = "open"
            self.stuck = 0
            return None
        if rejected:
            self.stuck += 1
            if self.stuck > 12:
                self.stuck = 0
                self.phase = "open"
                return None
        dq, conv = self.cart_step(R, tgt, Rdes, step_len=0.18, ptol=2e-3, rtol=6e-3,
                                  rot_step=0.35)
        if conv:
            self.phase = "open"
            self.stuck = 0
            self._pkey = None
            return None
        stalled = self.progress(R, tgt, Rdes, ("place", self.target, tuple(self.cell)))
        if dq is None or stalled:
            self.stuck = 0
            self._pkey = None
            return self.place_recover(R, tgt, Rdes)
        if self.cell_try == 0 and self.will_converge(R, dq, tgt, Rdes, 2e-3):
            self.phase = "open"
            self.cell_try = 1
            self._pkey = None
            self.stuck = 0
            return self.act_arm(dq, grip=1.0)
        return self.act_arm(dq)

    def ph_open(self, R):
        if not R.holding:
            self.phase = "select"
            self.stuck = 0
            self.place_fail = 0
            return None
        b = R.blocks[self.target]
        err = np.hypot(b[0] - self.cell[0], b[1] - self.cell[1])
        if err > 0.012:
            self.phase = "place"
            return None
        self.cell_try += 1
        if self.cell_try > 2:
            self.cell_try = 0
            self.next_cell(R)
            self.phase = "place"
            return None
        a = np.zeros(11, dtype=np.float32)
        a[10] = 1.0
        return a

    def next_cell(self, R):
        used = [(bb[0], bb[1]) for n, bb in R.blocks.items()
                if n != self.target and bb[7] < 0.5 and self.on_plate(R, bb)]
        b = R.blocks[self.target]
        free = [c for c in self.cells if c != self.cell and
                all(max(abs(c[0] - u[0]), abs(c[1] - u[1])) > 2 * b[8] + 0.012
                    for u in used)]
        if free:
            self.cell = free[0]
