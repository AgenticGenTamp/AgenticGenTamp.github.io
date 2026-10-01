"""Policy for kinder/BaseMotion3D-v0.

Black-box findings (all verified against the live environment):

  * The episode terminates as soon as the robot BASE (obs[0], obs[1]) is within
    0.05 (euclidean, xy only) of the target (obs[19], obs[20]).  The arm joints,
    the base rotation, the gripper and the target z are all irrelevant to the
    goal test (verified by measuring the termination radius for many different
    arm configurations: always exactly 0.05 in every direction).
  * Reward is -1 per step, so an optimal policy needs the fewest steps.  Base
    deltas are box bounded by +-0.4 per axis, so the minimum step count is
    ceil(Linf(target - start) / 0.4); saturating both axes achieves it.
  * The world has static obstacles (identical for every seed).  A colliding
    step is rejected entirely - the state does not change at all - so a blocked
    move just wastes a timestep.  The robot starts at (0, 0) and targets lie in
    [-2, 2]^2, where the only obstacle in the way is a wall whose inflated
    free-space boundary is y >= -1.9088 for x in [-0.755, 1.44] (with a
    diagonal edge running down-left from its left corner).  Straight lines from
    the start to any target never cross it, so no global planning is needed.
  * Base rotation and arm configuration do not change the collision boundary,
    so targets more than 0.05 beyond the wall are simply unreachable (~0.6% of
    seeds); for those we get as close as possible and idle.
  * If any commanded joint delta would violate a joint limit, the WHOLE action
    is rejected (including the base motion), so all joint deltas stay zero.

The controller drives straight at the (obstacle-clamped) target with saturated
steps, and carries a model-free fallback: a bisection on the step length when a
move is rejected, then a bug-style wall follow, so it still behaves sensibly if
an instance has a layout different from the one probed.
"""

import heapq

import numpy as np

_MAXD = 0.4
_TOL = 0.05



# ---------------------------------------------------------------------------
# Static occupancy map of the world, measured by walking the base over a 0.1 m
# grid with the live environment (the layout is identical for every seed).
# Encoded as "xindex:ylo~yhi,ylo~yhi;...", indices are multiples of 0.1.
# ---------------------------------------------------------------------------
_MAP_ENC = (
    "-34:-22~62;-33:-22~62;-32:-22~62;-31:-22~62;-30:-22~62;-29:-22~62;-28:-22~62;-27:-23~62;-26:-23~62;-25:-23~62;-24:-23~62;-23:-23~62;-22:-23~62;-21:-23~62;-20:-23~62;-19:-23~62;-18:-23~62;-17:-23~62;-16:-23~62;-15:-23~62;-14:-23~62;-13:-23~62;-12:-23~62;-11:-23~62;-10:-23~62;-9:-22~62;-8:-20~62;-7:-19~62;-6:-19~62;-5:-19~62;-4:-19~62;-3:-19~62;-2:-19~62;-1:-19~62;0:-19~62;1:-19~62;2:-19~26;3:-19~24;4:-19~24;5:-19~24;6:-19~24;7:-19~24;8:-19~24;9:-19~24;10:-19~24;11:-19~24;12:-19~24;13:-19~24;14:-19~24;15:-38~24;16:-38~24;17:-38~24;18:-38~24;19:-38~24;20:-38~24;21:-38~24;22:-38~24;23:-37~24;24:-37~24;25:-37~24;26:-37~24;27:-37~24;28:-37~24;29:-37~24"
)
_RES = 0.1


def _decode_map():
    cols = {}
    for part in _MAP_ENC.split(";"):
        k, v = part.split(":")
        cols[int(k)] = tuple(
            tuple(int(z) for z in run.split("~")) for run in v.split(",")
        )
    return cols


_MAP = _decode_map()


def _cell(x, y):
    return (int(np.floor(x / _RES + 0.5)), int(np.floor(y / _RES + 0.5)))


def _map_free(c):
    runs = _MAP.get(c[0])
    if runs is None:
        return False
    j = c[1]
    for a, b in runs:
        if a <= j <= b:
            return True
    return False


class GeneratedApproach:
    """Drive the base to the target; fall back to planning if anything blocks."""

    # Inflated obstacle boundary of the wall that can sit between the start
    # pose and a target (measured on the live environment).
    WALL_Y = -1.9075
    WALL_X_LO = -0.755
    WALL_X_HI = 1.44

    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.observation_space = observation_space
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is None:
            self.dim, self.lo, self.hi = 11, -_MAXD, _MAXD
        else:
            low = np.asarray(low, dtype=np.float64).ravel()
            high = np.asarray(high, dtype=np.float64).ravel()
            self.dim = int(low.shape[0])
            self.lo, self.hi = float(low[0]), float(high[0])
        self.reset(None, None)

    # ------------------------------------------------------------------
    def reset(self, state, info=None):
        self.prev_pos = None
        self.prev_cmd = None
        self.bound = None            # upper bound on the free advance
        self.use_true_target = False
        self.mode = "direct"
        self.path = None
        self.blocked_cells = set()
        self.blocked_pts = []
        self.trust_map = len(_MAP) > 0
        self.known_free = set()
        self.follow_dir = None
        self.follow_side = 1.0
        self.follow_steps = 0
        self.follow_rot = 0
        self.tried_direct = False
        self.hit_dist = None
        self.stuck = 0

    # ---------------------------- helpers -----------------------------
    def _clamp_goal(self, tgt):
        """Nearest point to `tgt` that the obstacle model says is reachable."""
        x = float(tgt[0])
        y = float(tgt[1])
        lim = None
        if self.WALL_X_LO <= x <= self.WALL_X_HI:
            lim = self.WALL_Y
        elif -1.0 <= x < self.WALL_X_LO:
            # diagonal left edge: y = -2.010 + 1.925 * (x + 0.8), plus margin
            lim = -2.006 + 1.925 * (x + 0.8)
        if lim is not None and y < lim:
            y = lim
        return np.array([x, y], dtype=np.float64)

    def _action(self, dx, dy):
        a = np.zeros(self.dim, dtype=np.float32)
        a[0] = float(np.clip(dx, self.lo, self.hi))
        a[1] = float(np.clip(dy, self.lo, self.hi))
        return a

    def _free(self, c):
        if c in self.blocked_cells:
            return False
        if c in self.known_free:
            return True
        return _map_free(c)

    def _seg_ok(self, p, q):
        """Segment check; when the static map is not trusted only learned
        obstacles are respected."""
        if self.trust_map:
            return self._seg_free(p, q)
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        n = max(abs(dx), abs(dy))
        k = int(n / 0.04) + 1
        bad = self.blocked_cells
        pts = self.blocked_pts
        for i in range(k + 1):
            t = float(i) / k
            c = _cell(p[0] + t * dx, p[1] + t * dy)
            if c in bad or c in pts:
                return False
        return True

    def _seg_free(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        n = max(abs(dx), abs(dy))
        k = int(n / 0.04) + 1
        for i in range(k + 1):
            t = float(i) / k
            if not self._free(_cell(p[0] + t * dx, p[1] + t * dy)):
                return False
        return True

    # ---------------------------- planning ----------------------------
    def _astar(self, start, goal, optimistic):
        if start == goal:
            return [goal]
        if optimistic:
            bad = set(self.blocked_cells) | set(self.blocked_pts)
            free = lambda c: c not in bad
        else:
            free = self._free
        openh = [(0.0, start)]
        gs = {start: 0.0}
        came = {}
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        seen = 0
        while openh:
            _, cur = heapq.heappop(openh)
            if cur == goal:
                out = [cur]
                while cur in came:
                    cur = came[cur]
                    out.append(cur)
                out.reverse()
                return out
            seen += 1
            if seen > 40000:
                return None
            gc = gs[cur]
            for dx, dy in nbrs:
                nb = (cur[0] + dx, cur[1] + dy)
                if not free(nb):
                    continue
                ng = gc + (1.41421356 if dx and dy else 1.0)
                if ng < gs.get(nb, 1e18) - 1e-9:
                    gs[nb] = ng
                    came[nb] = cur
                    h = np.hypot(nb[0] - goal[0], nb[1] - goal[1])
                    heapq.heappush(openh, (ng + h, nb))
        return None

    def _nearest_reachable(self, start, tgt_pt):
        """Flood fill from `start`; return the free cell closest to the target."""
        seen = {start}
        stack = [start]
        best = start
        bd = 1e18
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        while stack:
            cur = stack.pop()
            d = np.hypot(cur[0] * _RES - tgt_pt[0], cur[1] * _RES - tgt_pt[1])
            if d < bd:
                bd = d
                best = cur
            for dx, dy in nbrs:
                nb = (cur[0] + dx, cur[1] + dy)
                if nb not in seen and self._free(nb):
                    seen.add(nb)
                    stack.append(nb)
            if len(seen) > 40000:
                break
        return best, bd

    def _nearest_free_cell(self, pos):
        c = _cell(pos[0], pos[1])
        if self._free(c):
            return c
        best = None
        bd = 1e18
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nb = (c[0] + dx, c[1] + dy)
                if self._free(nb):
                    dd = np.hypot(nb[0] * _RES - pos[0], nb[1] * _RES - pos[1])
                    if dd < bd:
                        bd = dd
                        best = nb
        return best

    def _make_path(self, pos, goal):
        start = self._nearest_free_cell(pos)
        gcell = _cell(goal[0], goal[1])
        path = None
        if start is not None:
            path = self._astar(start, gcell, False)
        else:
            start = _cell(pos[0], pos[1])
        if path is None:
            best, bd = self._nearest_reachable(start, goal)
            if bd < 1e17 and best != start:
                path = self._astar(start, best, False)
            if path is None:
                path = self._astar(start, gcell, True)
        if path is None:
            return None
        pts = [np.array([c[0] * _RES, c[1] * _RES]) for c in path]
        gp = np.asarray(goal, dtype=np.float64)
        if np.hypot(pts[-1][0] - gp[0], pts[-1][1] - gp[1]) > 1e-9:
            pts.append(gp)
        return pts

    # ---------------------------- control -----------------------------
    def get_action(self, state):
        s = np.asarray(state, dtype=np.float64).ravel()
        pos = s[0:2].copy()
        true_tgt = s[19:21].copy()

        moved = True
        advance = 0.0
        if self.prev_pos is not None and self.prev_cmd is not None:
            cmd_mag = float(np.hypot(self.prev_cmd[0], self.prev_cmd[1]))
            advance = float(np.hypot(pos[0] - self.prev_pos[0], pos[1] - self.prev_pos[1]))
            if cmd_mag > 1e-9 and advance <= 1e-7:
                moved = False
        cur_cell = _cell(pos[0], pos[1])
        if (abs(pos[0] - cur_cell[0] * _RES) < 0.03
                and abs(pos[1] - cur_cell[1] * _RES) < 0.03):
            self.known_free.add(cur_cell)   # only trust cells we really sit on
        if not moved:
            blocked_len = float(np.hypot(self.prev_cmd[0], self.prev_cmd[1]))
            self.bound = blocked_len if self.bound is None else min(self.bound, blocked_len)
            end = (self.prev_pos[0] + self.prev_cmd[0], self.prev_pos[1] + self.prev_cmd[1])
            ec = _cell(end[0], end[1])
            # only blame the whole 0.1 cell when the rejected endpoint is
            # essentially at its centre - otherwise we would wrongly condemn a
            # cell whose free part we can still use.
            if ec != cur_cell and max(abs(end[0] - ec[0] * _RES),
                                      abs(end[1] - ec[1] * _RES)) <= 0.025:
                self.blocked_cells.add(ec)
            if blocked_len > 0.03 and ec != cur_cell:
                self.blocked_pts.append(ec)
            self.path = None
            self.stuck += 1
        else:
            self.stuck = 0
        if moved and self.bound is not None:
            self.bound = max(self.bound - advance, 0.0)

        goal = true_tgt if self.use_true_target else self._clamp_goal(true_tgt)
        d = goal - pos
        dist = float(np.hypot(d[0], d[1]))
        if dist <= 1e-6 and not self.use_true_target:
            # modelled goal reached but no termination: aim at the real target
            self.use_true_target = True
            self.bound = None
            self.path = None
            self.mode = "direct"
            goal = true_tgt
            d = goal - pos
            dist = float(np.hypot(d[0], d[1]))

        cmd = self._step_cmd(pos, goal, d, dist, moved)

        self.prev_pos = pos
        self.prev_cmd = np.array(cmd, dtype=np.float64)
        return self._action(cmd[0], cmd[1])

    def _reach(self, d, dist):
        """Longest step along direction d that fits in the per-axis box."""
        return min(dist, _MAXD * _linf_scale(d, self.hi))

    def _step_cmd(self, pos, goal, d, dist, moved):
        if dist <= 1e-9:
            return (0.0, 0.0)

        if self.mode == "direct":
            step = self._reach(d, dist)
            if self.bound is not None:
                if dist > _TOL + 0.03 and not self._seg_ok(pos, goal):
                    return self._switch_to_path(pos, goal, d, dist)
                if self.bound < 0.002:
                    if dist <= _TOL + 0.03:
                        return (0.0, 0.0)      # as close as this line allows
                    return self._switch_to_path(pos, goal, d, dist)
                step = min(step, 0.5 * self.bound)
            u = _unit(d)
            return (u[0] * step, u[1] * step)

        if self.mode == "path":
            if self.stuck >= 4:
                self._enter_follow(dist)
                return self._follow_cmd(d, dist)
            if self.path is None:
                self.path = self._make_path(pos, goal)
                if self.path is None:
                    self._enter_follow(dist)
                    return self._follow_cmd(d, dist)
                self.bound = None
            # string pulling: aim at the farthest visible waypoint
            aim = None
            for k in range(len(self.path) - 1, -1, -1):
                # only shortcut along the path when the static map is trusted;
                # otherwise walk it cell by cell so that every rejected move
                # teaches us something about the unknown layout.
                if (self._seg_free(pos, self.path[k]) if self.trust_map
                        else k == 0):
                    aim = self.path[k]
                    self.path = self.path[k:]
                    break
            if aim is None:
                aim = self.path[0]
            ad = aim - pos
            adist = float(np.hypot(ad[0], ad[1]))
            if adist < 1e-6:
                if len(self.path) > 1:
                    self.path = self.path[1:]
                    aim = self.path[0]
                    ad = aim - pos
                    adist = float(np.hypot(ad[0], ad[1]))
                else:
                    if dist <= _TOL + 0.03 or self.bound is not None:
                        return (0.0, 0.0)
                    self._enter_follow(dist)
                    return self._follow_cmd(d, dist)
            step = self._reach(ad, adist)
            if self.bound is not None:
                if self.bound < 0.002:
                    if dist <= _TOL + 0.03:
                        return (0.0, 0.0)
                    self._enter_follow(dist)
                    return self._follow_cmd(d, dist)
                step = min(step, 0.5 * self.bound)
            u = _unit(ad)
            return (u[0] * step, u[1] * step)

        # ------------- last resort: bug-style wall follow ------------------
        self.follow_steps += 1
        if self.follow_steps > 400:
            return (0.0, 0.0)
        if self.tried_direct:
            if moved:
                if dist < self.hit_dist - 1e-3:
                    self.hit_dist = dist
                return self._direct_cmd(d, dist)
            self.tried_direct = False
            return self._follow_cmd(d, dist)
        if not moved:
            self.follow_dir = _rot(self.follow_dir, self.follow_side * np.pi / 4.0)
            self.follow_rot += 1
            if self.follow_rot > 7:
                self.follow_rot = 0
                self.follow_side = -self.follow_side
                self.follow_dir = _rot(_unit(d), self.follow_side * np.pi / 2.0)
            return self._follow_cmd(d, dist)
        self.tried_direct = True
        self.follow_dir = _rot(_unit(d), self.follow_side * np.pi / 2.0)
        self.follow_rot = 0
        return self._direct_cmd(d, dist)

    def _switch_to_path(self, pos, goal, d, dist):
        self.mode = "path"
        self.path = self._make_path(pos, goal)
        self.bound = None
        if self.path is None:
            self._enter_follow(dist)
            return self._follow_cmd(d, dist)
        return self._step_cmd(pos, goal, d, dist, True)

    def _direct_cmd(self, d, dist):
        step = self._reach(d, dist)
        u = _unit(d)
        return (u[0] * step, u[1] * step)

    def _enter_follow(self, dist):
        self.mode = "follow"
        self.follow_steps = 0
        self.follow_rot = 0
        self.tried_direct = False
        self.hit_dist = dist
        self.follow_dir = None

    def _follow_cmd(self, d, dist):
        if self.follow_dir is None:
            self.follow_dir = _rot(_unit(d), self.follow_side * np.pi / 2.0)
        u = self.follow_dir
        step = max(0.05, min(dist, _MAXD))
        return (u[0] * step, u[1] * step)


def _linf_scale(d, hi):
    """Largest scale k such that k * unit(d) fits in the per-axis box."""
    n = float(np.hypot(d[0], d[1]))
    if n < 1e-12:
        return 1.0
    m = max(abs(d[0]), abs(d[1])) / n
    return 1.0 / m if m > 1e-9 else 1.0


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-9:
        return (1.0, 0.0)
    return (float(v[0]) / n, float(v[1]) / n)


def _rot(u, ang):
    c, s = float(np.cos(ang)), float(np.sin(ang))
    return (u[0] * c - u[1] * s, u[0] * s + u[1] * c)
