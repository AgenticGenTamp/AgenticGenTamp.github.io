"""PR2Blocked: kinematic pick-and-place policy.

Hand-rolled (calibrated) PR2 left-arm FK/IK + base placement search + grid A*
base path planning + a verifying state machine executor.
"""
import numpy as np

# ---------------------------------------------------------------- kinematics
JOINT_LOW = np.array([-0.7146, -0.5236, -0.8, -2.3213, -np.inf, -2.094, -np.inf])
JOINT_HIGH = np.array([2.2854, 1.3963, 3.9, 0.0, np.inf, 0.0, np.inf])
CONT = (4, 6)
SHOULDER_XY = np.array([-0.05, 0.188])
TORSO_Z = 0.99068
L_UP, L_FORE, L_TOOL = 0.4, 0.321, 0.18
MAXD = 0.2          # per-step joint / base delta
GRASP_BACK = 0.02   # tool origin behind block centre along approach
GRASP_UP = 0.03     # tool origin above block centre


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def arm_chain(q):
    """Return (points, R_tool, axes) with joint origins/axes in the base frame."""
    p = np.array([SHOULDER_XY[0], SHOULDER_XY[1], TORSO_Z])
    R = np.eye(3)
    origins = []
    axes = []
    # joint 1: z axis at shoulder
    origins.append(p.copy()); axes.append(R[:, 2].copy())
    R = R @ _rz(q[0])
    p1 = p + R @ np.array([0.1, 0.0, 0.0])
    origins.append(p1.copy()); axes.append(R[:, 1].copy())
    R = R @ _ry(q[1])
    origins.append(p1.copy()); axes.append(R[:, 0].copy())
    R = R @ _rx(q[2])
    p2 = p1 + R @ np.array([L_UP, 0.0, 0.0])
    origins.append(p2.copy()); axes.append(R[:, 1].copy())
    R = R @ _ry(q[3])
    origins.append(p2.copy()); axes.append(R[:, 0].copy())
    R = R @ _rx(q[4])
    p3 = p2 + R @ np.array([L_FORE, 0.0, 0.0])
    origins.append(p3.copy()); axes.append(R[:, 1].copy())
    R = R @ _ry(q[5])
    origins.append(p3.copy()); axes.append(R[:, 0].copy())
    R = R @ _rx(q[6])
    p4 = p3 + R @ np.array([L_TOOL, 0.0, 0.0])
    return [p, p1, p2, p3, p4], R, (origins, axes)


def arm_fk(q):
    pts, R, _ = arm_chain(q)
    return pts[4], R


def world_fk(q, base):
    p, R = arm_fk(q)
    Rb = _rz(base[2])
    return Rb @ p + np.array([base[0], base[1], 0.0]), Rb @ R


def world_chain(q, base):
    pts, R, _ = arm_chain(q)
    Rb = _rz(base[2])
    off = np.array([base[0], base[1], 0.0])
    return [Rb @ p + off for p in pts], Rb @ R


def _rot_err(R, tgt_R):
    Re = R @ tgt_R.T
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    s = np.linalg.norm(v)
    c = (np.trace(Re) - 1) / 2.0
    ang = np.arctan2(s / 2.0, c)
    return v / (s + 1e-12) * ang if s > 1e-9 else np.zeros(3)


def pose_err(q, base, tgt_p, tgt_R, w_rot=1.0):
    p, R = world_fk(q, base)
    return np.concatenate([p - tgt_p, w_rot * _rot_err(R, tgt_R)])


def jacobian(q, base):
    """Geometric Jacobian of the tool pose in world coordinates."""
    pts, R, (origins, axes) = arm_chain(q)
    Rb = _rz(base[2])
    tip = Rb @ pts[4]
    J = np.zeros((6, 7))
    for i in range(7):
        a = Rb @ axes[i]
        o = Rb @ origins[i]
        J[:3, i] = np.cross(a, tip - o)
        J[3:, i] = a
    return J


def clip_q(q):
    q = np.array(q, dtype=float)
    for i in range(7):
        if i in CONT:
            q[i] = (q[i] + np.pi) % (2 * np.pi) - np.pi
        else:
            q[i] = min(max(q[i], JOINT_LOW[i]), JOINT_HIGH[i])
    return q


def dq_wrap(dq):
    dq = np.array(dq, dtype=float)
    for i in CONT:
        dq[i] = (dq[i] + np.pi) % (2 * np.pi) - np.pi
    return dq


def _solve_one(q, base, tgt_p, tgt_R, iters, w_rot):
    q = clip_q(q)
    for _ in range(iters):
        p, R = world_fk(q, base)
        e = np.concatenate([p - tgt_p, w_rot * _rot_err(R, tgt_R)])
        if np.linalg.norm(e) < 1e-9:
            break
        J = jacobian(q, base)
        J[3:] *= w_rot
        try:
            step = np.linalg.solve(J.T @ J + 0.02 * np.eye(7), -J.T @ e)
        except np.linalg.LinAlgError:
            break
        n = np.linalg.norm(step)
        if n > 0.4:
            step *= 0.4 / n
        qn = clip_q(q + step)
        if np.max(np.abs(qn - q)) < 1e-9:
            break
        q = qn
    return q


def ik_solutions(tgt_p, tgt_R, base, q0, seeds=6, iters=45, rng=None, w_rot=0.5,
                 tol=0.004):
    if rng is None:
        rng = np.random.default_rng(0)
    lo = np.where(np.isfinite(JOINT_LOW), JOINT_LOW, -np.pi)
    hi = np.where(np.isfinite(JOINT_HIGH), JOINT_HIGH, np.pi)
    out = []
    cands = [np.array(q0, dtype=float)]
    for _ in range(seeds - 1):
        cands.append(rng.uniform(lo, hi))
    for c in cands:
        q = _solve_one(c, base, tgt_p, tgt_R, iters, w_rot)
        e = pose_err(q, base, tgt_p, tgt_R, 1.0)
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 0.02:
            if not any(np.max(np.abs(dq_wrap(q - o))) < 0.05 for o in out):
                out.append(q)
    return out


def grasp_R(yaw):
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack([x, np.cross(z, x), z])


# ------------------------------------------------------------------- world
class Rect:
    def __init__(self, cx, cy, hx, hy, top):
        self.c = np.array([cx, cy])
        self.h = np.array([hx, hy])
        self.top = top

    def dist_xy(self, p):
        d = np.abs(np.asarray(p)[:2] - self.c) - self.h
        return np.linalg.norm(np.maximum(d, 0)) + min(max(d[0], d[1]), 0.0)

    def inside(self, p, margin=0.0):
        return self.dist_xy(p) < margin


def seg_points(a, b, n):
    return [a + (b - a) * t for t in np.linspace(0, 1, n)]


class GeneratedApproach:
    BASE_MARGIN = 0.40          # base centre clearance from a table footprint
    WORLD_LIM = 4.9

    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------- state access
    def _robot(self, state):
        r = state.get_object_from_name("robot")
        g = lambda f: state.get(r, f)
        return dict(base=np.array([g("base_x"), g("base_y"), g("base_rot")]),
                    q=np.array([g("joint_%d" % i) for i in range(1, 8)]),
                    grip=g("gripper_opening"), holding=g("grasp_active") > 0.5)

    def _blocks(self, state):
        out = {}
        for name in sorted(state.get_object_names()):
            o = state.get_object_from_name(name)
            if o.type.name == "block":
                out[name] = np.array([state.get(o, "pose_x"), state.get(o, "pose_y"),
                                      state.get(o, "pose_z")])
        return out

    def _surfaces(self, state):
        out = {}
        for name in sorted(state.get_object_names()):
            o = state.get_object_from_name(name)
            if o.type.name == "surface":
                out[name] = np.array([state.get(o, f) for f in
                                      ("pose_x", "pose_y", "pose_z",
                                       "half_extent_x", "half_extent_y",
                                       "half_extent_z")])
        return out

    # ------------------------------------------------------------- planning
    def reset(self, state, info):
        self.step_i = 0
        self.fail = 0
        self.prev_robot = None
        self.rej_streak = 0
        surf = self._surfaces(state)
        self.tables = {}
        for nm in ("near_table", "far_table"):
            if nm in surf:
                s = surf[nm]
                self.tables[nm] = Rect(s[0], s[1], s[3], s[4], s[2] + s[5] * 2)
        s = surf["plate"]
        self.plate = Rect(s[0], s[1], s[3], s[4], s[2] + s[5] * 2)
        self.plate_z = s[2] + s[5]
        self.table_top = 0.73
        self.last_cmd = None
        self._plan(state)

    # base path planning -----------------------------------------------------
    def _base_blocked(self, p, margin=None):
        m = self.BASE_MARGIN if margin is None else margin
        if abs(p[0]) > self.WORLD_LIM or abs(p[1]) > self.WORLD_LIM:
            return True
        return any(t.dist_xy(p) < m for t in self.tables.values())

    def _base_path(self, start, goal):
        """Chebyshev-metric A* on a 0.1 m grid; returns list of xy waypoints."""
        res = 0.1
        start = np.asarray(start, dtype=float)[:2]
        goal = np.asarray(goal, dtype=float)[:2]
        if not self._seg_blocked(start, goal):
            return [goal]
        import heapq
        s = (int(round(start[0] / res)), int(round(start[1] / res)))
        g = (int(round(goal[0] / res)), int(round(goal[1] / res)))
        def h(a):
            return max(abs(a[0] - g[0]), abs(a[1] - g[1]))
        openq = [(h(s), 0, s)]
        came = {s: None}
        cost = {s: 0}
        found = False
        while openq:
            _, c, cur = heapq.heappop(openq)
            if cur == g:
                found = True
                break
            if c > cost.get(cur, 1e9):
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cur[0] + dx, cur[1] + dy)
                    pt = np.array([nb[0] * res, nb[1] * res])
                    if nb != g and self._base_blocked(pt):
                        continue
                    nc = c + 1
                    if nc < cost.get(nb, 1e9):
                        cost[nb] = nc
                        came[nb] = cur
                        heapq.heappush(openq, (nc + h(nb), nc, nb))
        if not found:
            return [goal]
        path = []
        cur = g
        while cur is not None:
            path.append(np.array([cur[0] * res, cur[1] * res]))
            cur = came[cur]
        path.reverse()
        path[-1] = goal
        # string pull
        out = []
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and self._seg_blocked(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _seg_blocked(self, a, b, margin=None):
        n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / 0.05) + 1)
        return any(self._base_blocked(a + (np.asarray(b) - np.asarray(a)) * t, margin)
                   for t in np.linspace(0, 1, n))

    # arm feasibility -------------------------------------------------------
    def _arm_ok(self, q, base, tip_free=0.22):
        """Reject configurations whose arm links would hit the table or torso."""
        pts, _ = world_chain(q, base)
        elbow, tip = pts[2], pts[4]
        if elbow[2] < 0.74:
            return False
        samples = seg_points(pts[1], pts[2], 5) + seg_points(pts[2], pts[3], 6)
        for p in samples:
            if np.linalg.norm(p - tip) < tip_free:
                continue
            if p[2] < 0.78:
                for t in self.tables.values():
                    if t.dist_xy(p) < 0.09:
                        return False
            if p[2] < 0.95 and np.linalg.norm(p[:2] - np.asarray(base)[:2]) < 0.26:
                return False
        return True

    def _arm_cost(self, q, base, keepout=None):
        pts, _ = world_chain(q, base)
        cost = -0.4 * min(pts[2][2], 1.15)
        tip = pts[4]
        for p in seg_points(pts[1], pts[2], 5) + seg_points(pts[2], pts[3], 6):
            if np.linalg.norm(p - tip) < 0.22:
                continue
            for t in self.tables.values():
                d = t.dist_xy(p)
                if d < 0.16 and p[2] < 0.86:
                    cost += 3.0 * (0.86 - p[2])
        return cost

    # ------------------------------------------------------------ grasping
    def grasp_pose(self, block, d):
        return np.array([block[0] - d[0] * GRASP_BACK,
                         block[1] - d[1] * GRASP_BACK,
                         block[2] + GRASP_UP])

    def _cart_path(self, base, q_start, pts, R, max_jump=0.8):
        out = []
        q = np.array(q_start, dtype=float)
        for p in pts:
            sols = ik_solutions(p, R, base, q, seeds=1, iters=70, rng=self.rng)
            if not sols:
                sols = ik_solutions(p, R, base, q, seeds=4, iters=90, rng=self.rng)
            if not sols:
                return None
            best = min(sols, key=lambda s: np.max(np.abs(dq_wrap(s - q))))
            if np.max(np.abs(dq_wrap(best - q))) > max_jump:
                return None
            out.append(best)
            q = best
        return out

    def _aligned_bases(self, block, d, backs=(0.85, 0.82, 0.88, 0.79, 0.90, 0.75)):
        """Base poses that put the shoulder on the approach axis."""
        perp = np.array([-d[1], d[0]])
        yaw = np.arctan2(d[1], d[0])
        out = []
        for back in backs:
            p = np.asarray(block)[:2] - d[:2] * back - perp * 0.188
            if self._base_blocked(p):
                continue
            out.append(np.array([p[0], p[1], yaw]))
        return out

    def _ring_bases(self, block, d, radii=(0.70, 0.80, 0.62, 0.86, 0.55)):
        yaw0 = np.arctan2(d[1], d[0])
        out = []
        for r in radii:
            for k in range(15):
                ang = yaw0 + np.pi + ((k + 1) // 2) * (1 if k % 2 else -1) * 0.35
                p = np.asarray(block)[:2] + r * np.array([np.cos(ang), np.sin(ang)])
                if self._base_blocked(p):
                    continue
                face = np.arctan2(block[1] - p[1], block[0] - p[0])
                for dyaw in (0.0, 0.5, -0.5):
                    out.append(np.array([p[0], p[1], face + dyaw]))
        return out

    def _base_candidates(self, block, d):
        return self._aligned_bases(block, d) + self._ring_bases(block, d)

    def _grasp_options(self, base, block, d, q_seed, approach=0.20, strict=True):
        """Ranked list of joint-waypoint sequences [pregrasp .. grasp]."""
        yaw = np.arctan2(d[1], d[0])
        R = grasp_R(yaw)
        p_g = self.grasp_pose(block, d)
        offs = [approach]
        x = approach
        while x > 1e-9:
            x = max(x - 0.05, 0.0)
            offs.append(x)
        pts = [p_g - d * o for o in offs]
        sols = ik_solutions(pts[0], R, base, q_seed, seeds=12, rng=self.rng)
        out = []
        for qp in sols:
            if strict and not self._arm_ok(qp, base):
                continue
            seq = self._cart_path(base, qp, pts[1:], R, max_jump=0.9)
            if seq is None:
                continue
            if strict and not all(self._arm_ok(q, base) for q in seq):
                continue
            cost = (np.max(np.abs(dq_wrap(qp - q_seed))) / MAXD
                    + sum(self._arm_cost(q, base) for q in [qp] + seq) / (1 + len(seq)))
            out.append((cost, [qp] + seq, R))
        out.sort(key=lambda t: t[0])
        return out

    # ------------------------------------------------------------- top plan
    def _plan(self, state):
        blocks = self._blocks(state)
        g0 = blocks["green0"]
        bl = blocks["blocker"]
        d = g0[:2] - bl[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        self.dir = np.array([d[0], d[1], 0.0])
        self.spares = sorted([n for n in blocks
                              if n.startswith("green") and n != "green0"])
        self.tasks = [("pick", "blocker"), ("dump", "blocker"),
                      ("pick", "green0"), ("place", "green0")]
        self.task_i = 0
        self.queue = []
        self.retry = 0
        self.tgt_idx = 0
        self.pending = None
        self.base_pose = None
        self.base_alt = 0
        self._opt_cache = {}
        self.cur_R = grasp_R(np.arctan2(d[1], d[0]))
        r = self._robot(state)
        self._choose_base(state, r)

    def _choose_base(self, state, r):
        """Pick a base pose that serves both the blocker and the green0 grasp."""
        blocks = self._blocks(state)
        g0 = blocks["green0"]
        bl = blocks["blocker"]
        d = self.dir
        cands = self._base_candidates(g0, d)
        scored = []
        for base in cands:
            o1 = self._grasp_options(base, bl, d, r["q"], approach=0.16)
            if not o1:
                continue
            o2 = self._grasp_options(base, g0, d, o1[0][1][-1], approach=0.20)
            if not o2:
                continue
            drive = np.max(np.abs(base[:2] - r["base"][:2])) / MAXD
            scored.append((o1[0][0] + o2[0][0] + 0.3 * drive, base))
            self._opt_cache[("blocker", tuple(np.round(base, 4)))] = o1
            self._opt_cache[("green0", tuple(np.round(base, 4)))] = o2
            if len(scored) >= 2:
                break
        if scored:
            scored.sort(key=lambda t: t[0])
            k = min(self.base_alt, len(scored) - 1)
            self.base_pose = scored[k][1]
        else:
            self.base_pose = cands[0] if cands else r["base"]
        return self.base_pose

    # queue helpers
    def _q_move(self, base, q, tol=0.02):
        return dict(kind="move", base=np.array(base, dtype=float),
                    q=np.array(q, dtype=float), tol=tol)

    def _q_grip(self, val, check=None):
        return dict(kind="grip", val=val, check=check)

    # ------------------------------------------------------------ task build
    def _build_task(self, state):
        kind, name = self.tasks[self.task_i]
        r = self._robot(state)
        if kind == "pick":
            self._build_pick(state, r, name)
        else:
            self._build_transfer(state, r, name, kind)

    def _build_pick(self, state, r, name):
        blocks = self._blocks(state)
        block = blocks[name]
        d = self.dir
        base = self.base_pose
        key = (name, tuple(np.round(base, 4)))
        opts = self._opt_cache.get(key, []) if self.retry == 0 else []
        for appr in ((0.20, 0.16, 0.24) if not opts else ()):
            opts = self._grasp_options(base, block, d, r["q"], approach=appr)
            if opts:
                break
        if not opts:
            self.base_alt += 1
            base = self._choose_base(state, r)
            opts = self._grasp_options(base, block, d, r["q"], approach=0.20)
        if not opts:
            for appr in (0.20, 0.16):
                opts = self._grasp_options(base, block, d, r["q"], approach=appr,
                                           strict=False)
                if opts:
                    break
        if not opts:
            self.queue = [self._q_move(r["base"], r["q"])]
            return
        _, seq, R = opts[min(self.retry, len(opts) - 1)]
        self.cur_R = R
        items = []
        path = self._base_path(r["base"][:2], base[:2])
        for wp in path[:-1]:
            items.append(self._q_move(np.array([wp[0], wp[1], base[2]]), seq[0],
                                      tol=0.06))
        items.append(self._q_move(base, seq[0], tol=0.01))
        for q in seq[1:-1]:
            items.append(self._q_move(base, q, tol=0.01))
        items.append(self._q_move(base, seq[-1], tol=0.0015))
        items.append(self._q_grip(-1.0, check=("grasped", name)))
        self.queue = items

    def _transfer_targets(self, state, kind, name):
        blocks = self._blocks(state)
        d = self.dir
        if kind == "place":
            c, h = self.plate.c, self.plate.h
            out = []
            for fx in (0.0, -0.35, 0.35, -0.65, 0.65):
                for fy in (0.0, -0.35, 0.35, 0.65, -0.65):
                    p = c + np.array([fx * h[0], fy * h[1]])
                    if any(nm != name and b[2] > 0.5 and
                           np.linalg.norm(b[:2] - p) < 0.13
                           for nm, b in blocks.items()):
                        continue
                    out.append(p)
            out.sort(key=lambda p: np.linalg.norm(p - blocks[name][:2]))
            return out
        g0 = blocks["green0"]
        perp = np.array([-d[1], d[0]])
        out = []
        for lat in (0.32, 0.40, 0.26, 0.48):
            for s in (1.0, -1.0):
                for back in (0.10, 0.0, 0.22, -0.10):
                    p = blocks[name][:2] + perp * s * lat - d[:2] * back
                    if np.linalg.norm(p - g0[:2]) < 0.28:
                        continue
                    if any(nm != name and b[2] > 0.5 and
                           np.linalg.norm(b[:2] - p) < 0.13
                           for nm, b in blocks.items()):
                        continue
                    if self.plate.dist_xy(p) < 0.07:
                        continue
                    dt = self.tables["near_table"].dist_xy(p)
                    if -0.075 < dt < 0.075:
                        continue
                    out.append(p)
        return out

    def _build_transfer(self, state, r, name, kind):
        d = self.dir
        R = self.cur_R
        p_tool, _ = world_fk(r["q"], r["base"])
        carry_z = max(p_tool[2] + 0.17, 1.0)
        targets = self._transfer_targets(state, kind, name)
        if not targets:
            self.queue = [self._q_move(r["base"], r["q"])]
            return
        tgt = targets[min(self.tgt_idx, len(targets) - 1)]
        base = r["base"]
        items = []
        p_up = np.array([p_tool[0], p_tool[1], carry_z])
        seq_up = self._cart_path(base, r["q"], [p_up], R)
        q_up = r["q"]
        if seq_up is not None:
            items.append(self._q_move(base, seq_up[0], tol=0.01))
            q_up = seq_up[0]
        else:
            p_up = p_tool
        p_rel = np.array([tgt[0] - d[0] * GRASP_BACK, tgt[1] - d[1] * GRASP_BACK,
                          p_up[2]])
        dist = np.linalg.norm(p_rel[:2] - p_up[:2])
        n = max(1, int(np.ceil(dist / 0.17)))
        pts = [p_up + (p_rel - p_up) * (i / n) for i in range(1, n + 1)]
        seq = self._cart_path(base, q_up, pts, R, max_jump=0.9)
        use_base = seq is None
        if not use_base:
            for q in seq:
                items.append(self._q_move(base, q, tol=0.02))
        else:
            delta = p_rel[:2] - p_up[:2]
            nb = np.array([base[0] + delta[0], base[1] + delta[1], base[2]])
            if self._base_blocked(nb[:2]) or self._seg_blocked(base[:2], nb[:2]):
                self.tgt_idx += 1
                self.queue = [self._q_move(base, q_up, tol=0.02)]
                return
            items.append(self._q_move(nb, q_up, tol=0.01))
        items.append(self._q_grip(1.0, check=("released", name)))
        self.queue = items

    # ----------------------------------------------------------------- exec
    def get_action(self, state):
        r = self._robot(state)
        a = np.zeros(11, dtype=np.float32)
        rejected = False
        if self.prev_robot is not None and self.last_cmd is not None:
            moved = np.max(np.abs(np.concatenate([r["base"], r["q"]]) -
                                  self.prev_robot))
            if np.max(np.abs(self.last_cmd[:10])) > 1e-4 and moved < 1e-7:
                rejected = True
        if self.pending is not None:
            what, name = self.pending
            self.pending = None
            ok = (r["holding"] if what == "grasped" else not r["holding"])
            if ok:
                self.retry = 0
                self.tgt_idx = 0
                self.task_i = min(self.task_i + 1, len(self.tasks) - 1)
            else:
                self.retry += 1
                if what == "released":
                    self.tgt_idx += 1
            self.queue = []
        self.rej_streak = self.rej_streak + 1 if rejected else 0
        if self.rej_streak > 5:
            self.queue = []
            self.rej_streak = 0
            self.retry += 1
        if not self.queue:
            self._build_task(state)
        # pop reached waypoints
        while self.queue:
            it = self.queue[0]
            if it["kind"] == "grip":
                break
            db = it["base"] - r["base"]
            db[2] = (db[2] + np.pi) % (2 * np.pi) - np.pi
            dq = dq_wrap(it["q"] - r["q"])
            if max(np.max(np.abs(db)), np.max(np.abs(dq))) < it["tol"]:
                self.queue.pop(0)
                if not self.queue:
                    self._build_task(state)
            else:
                break
        it = self.queue[0]
        if it["kind"] == "grip":
            a[10] = it["val"]
            self.pending = it["check"]
            self.queue.pop(0)
        else:
            db = it["base"] - r["base"]
            db[2] = (db[2] + np.pi) % (2 * np.pi) - np.pi
            dq = dq_wrap(it["q"] - r["q"])
            a[:3] = np.clip(db, -MAXD, MAXD)
            a[3:10] = np.clip(dq, -MAXD, MAXD)
            if rejected:
                k = self.rej_streak
                if k == 1:
                    a[:3] = 0.0
                elif k == 2:
                    a[3:10] = 0.0
                elif k == 3:
                    a[:10] *= 0.35
                else:
                    a[:3] = 0.0
                    a[3:10] = np.clip(dq, -MAXD, MAXD) * 0.3 + \
                        self.rng.uniform(-0.06, 0.06, 7)
        self.prev_robot = np.concatenate([r["base"], r["q"]])
        self.last_cmd = a.copy()
        self.step_i += 1
        return a
