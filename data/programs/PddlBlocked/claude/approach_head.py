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
GRASP_OFF = 0.03    # tool origin sits this far behind the block centre


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
    """Return (points, R_tool) for shoulder, elbow, wrist, tool in base frame."""
    p = np.array([SHOULDER_XY[0], SHOULDER_XY[1], TORSO_Z])
    R = _rz(q[0])
    p1 = p + R @ np.array([0.1, 0.0, 0.0])
    R = R @ _ry(q[1]) @ _rx(q[2])
    p2 = p1 + R @ np.array([L_UP, 0.0, 0.0])
    R = R @ _ry(q[3]) @ _rx(q[4])
    p3 = p2 + R @ np.array([L_FORE, 0.0, 0.0])
    R = R @ _ry(q[5]) @ _rx(q[6])
    p4 = p3 + R @ np.array([L_TOOL, 0.0, 0.0])
    return [p, p1, p2, p3, p4], R


def arm_fk(q):
    pts, R = arm_chain(q)
    return pts[4], R


def world_fk(q, base):
    p, R = arm_fk(q)
    Rb = _rz(base[2])
    return Rb @ p + np.array([base[0], base[1], 0.0]), Rb @ R


def world_chain(q, base):
    pts, R = arm_chain(q)
    Rb = _rz(base[2])
    off = np.array([base[0], base[1], 0.0])
    return [Rb @ p + off for p in pts], Rb @ R


def pose_err(q, base, tgt_p, tgt_R, w_rot=1.0):
    p, R = world_fk(q, base)
    Re = R @ tgt_R.T
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    s = np.linalg.norm(v)
    c = (np.trace(Re) - 1) / 2.0
    ang = np.arctan2(s / 2.0, c)
    er = v / (s + 1e-12) * ang if s > 1e-9 else np.zeros(3)
    return np.concatenate([p - tgt_p, w_rot * er])


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
        e = pose_err(q, base, tgt_p, tgt_R, w_rot)
        J = np.zeros((6, 7))
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = 1e-5
            J[:, i] = (pose_err(q + dq, base, tgt_p, tgt_R, w_rot) - e) / 1e-5
        try:
            step = np.linalg.solve(J.T @ J + 0.05 * np.eye(7), -J.T @ e)
        except np.linalg.LinAlgError:
            break
        n = np.linalg.norm(step)
        if n > 0.4:
            step *= 0.4 / n
        q = clip_q(q + step)
        if n < 1e-8:
            break
    return q


def ik_solutions(tgt_p, tgt_R, base, q0, seeds=6, iters=90, rng=None, w_rot=0.5,
                 tol=0.004):
    """Return a list of distinct joint solutions reaching the target pose."""
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

    # arm safety heuristic ---------------------------------------------------
    def _arm_cost(self, q, base, ignore_tip=0.28, keepout=None):
        pts, _ = world_chain(q, base)
        cost = 0.0
        tip = pts[4]
        samples = []
        samples += seg_points(pts[1], pts[2], 5)
        samples += seg_points(pts[2], pts[3], 5)
        samples += seg_points(pts[3], pts[4], 4)
        for p in samples:
            if np.linalg.norm(p - tip) < ignore_tip:
                continue
            for t in self.tables.values():
                d = t.dist_xy(p)
                if d < 0.05 and p[2] < self.table_top + 0.20:
                    cost += 6.0 * (self.table_top + 0.20 - p[2])
            if p[2] < 0.35:
                cost += 4.0 * (0.35 - p[2])
            if keepout is not None:
                for c, rad, ztop in keepout:
                    if p[2] < ztop and np.linalg.norm(p[:2] - c[:2]) < rad:
                        cost += 3.0
        return cost

    def _keepouts(self, state, exclude=()):
        """Cylinders around the pen / other blocks the arm should not cross."""
        ks = []
        for nm, b in self._blocks(state).items():
            if nm in exclude:
                continue
            r = 0.16 if nm == "green0" else 0.075
            ks.append((b[:2], r, 0.90))
        return ks

