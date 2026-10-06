"""Scripted pick-and-place planner for Obstruction2D with a fast kinematic model.

Strategy: keep the arm pointing down (theta = -pi/2).
1. Remove obstructions that overlap the target surface (or that otherwise
   block placement) by carrying them to free spots on the table.
2. Grasp the target block and place it on the target surface.
Every candidate motion is validated with a fast axis-aligned collision model
that mirrors the environment's step function. During execution, if the real
state deviates from the prediction, we replan.
"""

import time

import numpy as np

WORLD_W = 1.618
WORLD_H = 1.0
TABLE_TOP = 0.1
BASE_R = 0.1
ARM_MIN = 0.1
ARM_MAX = 0.2
GRIP_H = 0.07  # extent across the arm direction (x when pointing down)
GRIP_W = 0.01  # extent along the arm direction
ARM_HALF_W = 0.0025
MAX_D = 0.05
MAX_ARM = 0.1
EPS = 0.002  # collision safety margin
HELD_EPS = -1e-7  # held blocks may touch (they start touching table/walls)
GRASP_GAP = 0.0125  # block top = robot_y - aj - GRASP_GAP
PLACE_LIFT = 0.008
DOWN = -np.pi / 2


def _aabb_overlap(a, b, eps=EPS):
    # a, b = (x0, y0, x1, y1)
    return (a[0] < b[2] + eps and b[0] < a[2] + eps and
            a[1] < b[3] + eps and b[1] < a[3] + eps)


def _circle_aabb(cx, cy, r, b, eps=EPS):
    px = min(max(cx, b[0]), b[2])
    py = min(max(cy, b[1]), b[3])
    return (px - cx) ** 2 + (py - cy) ** 2 < (r + eps) ** 2


STATIC_BOXES = [
    (0.0, -0.1, WORLD_W, TABLE_TOP),  # table (+ bottom wall)
    (-0.1, -0.1, 0.0, WORLD_H + 0.1),  # left wall
    (WORLD_W, -0.1, WORLD_W + 0.1, WORLD_H + 0.1),  # right wall
    (-0.1, WORLD_H, WORLD_W + 0.1, WORLD_H + 0.1),  # top wall
]


class Model:
    """Kinematic model with robot pointing down and axis-aligned blocks."""

    def __init__(self, rx, ry, aj, vac, blocks, held):
        self.rx, self.ry, self.aj, self.vac = rx, ry, aj, vac
        self.blocks = blocks  # name -> [x, y, w, h]
        self.held = held  # name -> (offx, offy) block pos minus robot pos

    def copy(self):
        return Model(self.rx, self.ry, self.aj, self.vac,
                     {k: list(v) for k, v in self.blocks.items()},
                     dict(self.held))

    @staticmethod
    def box(b):
        return (b[0], b[1], b[0] + b[2], b[1] + b[3])

    def suction_set(self, rx, ry, aj, blocks):
        zone = (rx - GRIP_H / 2, ry - aj - 2 * GRIP_W,
                rx + GRIP_H / 2, ry - aj - GRIP_W)
        return [n for n, b in blocks.items()
                if _aabb_overlap(zone, self.box(b), eps=0.0)]

    def collides(self, rx, ry, aj, blocks, held_names):
        robot_boxes = [
            (rx - GRIP_H / 2, ry - aj - GRIP_W / 2,
             rx + GRIP_H / 2, ry - aj + GRIP_W / 2),  # gripper
            (rx - ARM_HALF_W, ry - aj, rx + ARM_HALF_W, ry),  # arm
        ]
        held_boxes = [self.box(blocks[n]) for n in held_names]
        obstacles = list(STATIC_BOXES) + [self.box(b) for n, b in blocks.items()
                                          if n not in held_names]
        for ob in obstacles:
            if _circle_aabb(rx, ry, BASE_R, ob):
                return True
            for rb in robot_boxes:
                if _aabb_overlap(rb, ob):
                    return True
            for hb in held_boxes:
                if _aabb_overlap(hb, ob, eps=HELD_EPS):
                    return True
        return False

    def step(self, a):
        """Apply action; return True if it succeeded (no collision)."""
        dx, dy, _, darm, vac = a
        nx, ny = self.rx + dx, self.ry + dy
        naj = min(max(self.aj + darm, ARM_MIN), ARM_MAX)
        blocks = self.blocks
        if self.held:
            blocks = {k: list(v) for k, v in self.blocks.items()}
            # tool tip moves with (x, y - aj); held offsets are relative to
            # the tool tip.
            for n, (ox, oy) in self.held.items():
                blocks[n][0] = nx + ox
                blocks[n][1] = ny - naj + oy
        if self.collides(nx, ny, naj, blocks, self.held.keys()):
            return False
        self.rx, self.ry, self.aj, self.vac = nx, ny, naj, vac
        self.blocks = blocks
        if vac > 0.5:
            names = self.suction_set(nx, ny, naj, blocks)
            self.held = {n: (blocks[n][0] - nx, blocks[n][1] - (ny - naj))
                         for n in names}
        else:
            self.held = {}
        return True


def seg_steps(a, b):
    return int(np.ceil(max(abs(b[0] - a[0]) / (MAX_D * 0.999),
                           abs(b[1] - a[1]) / (MAX_D * 0.999),
                           abs(b[2] - a[2]) / (MAX_ARM * 0.999)) - 1e-9))


def move_actions(cur, goal, vac):
    """cur/goal = (x, y, aj). Straight-line interpolation."""
    diff = np.array(goal, dtype=float) - np.array(cur, dtype=float)
    n = seg_steps(cur, goal)
    if n <= 0:
        return []
    st = diff / n
    return [np.array([st[0], st[1], 0.0, st[2], vac], dtype=np.float32)
            for _ in range(n)]


TOP_Y = WORLD_H - BASE_R - 0.004


VARIANTS = [("float", "tall", "float"), ("park", "tall", "float"),
            ("cost", "tall", "float"), ("float", "near", "float"),
            ("park", "near", "float"), ("park", "tall", "cost"),
            ("float", "short", "float"), ("clear", "tall", "float")]


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.plan = []
        self.pred = []
        self.idx = 0
        self.t0 = time.time()
        self.fail_count = 0

    # ---------------------------------------------------------------- parse
    def _parse(self, state):
        robot = None
        blocks = {}
        target = surface = None
        obst = []
        for o in state:
            tn = o.type.name
            if tn == "crv_robot":
                robot = o
            elif tn == "target_surface":
                surface = (state.get(o, "x"), state.get(o, "y"),
                           state.get(o, "width"), state.get(o, "height"))
            elif tn in ("target_block", "rectangle"):
                if tn == "target_block":
                    target = o.name
                elif state.get(o, "static") > 0.5:
                    continue
                else:
                    obst.append(o.name)
                blocks[o.name] = [float(state.get(o, f))
                                  for f in ("x", "y", "width", "height")]
        rx = float(state.get(robot, "x"))
        ry = float(state.get(robot, "y"))
        aj = float(state.get(robot, "arm_joint"))
        vac = float(state.get(robot, "vacuum"))
        m = Model(rx, ry, aj, vac, blocks, {})
        if vac > 0.5:
            names = m.suction_set(rx, ry, aj, blocks)
            m.held = {n: (blocks[n][0] - rx, blocks[n][1] - (ry - aj))
                      for n in names}
        return m, target, surface, sorted(obst)

    # ------------------------------------------------------------- helpers
    def _run(self, m, acts):
        m = m.copy()
        preds = []
        for a in acts:
            if not m.step(a):
                return None, None
            preds.append((m.rx, m.ry, m.aj))
        return m, preds

    def _held_blocks_at(self, m, gx, gy, gaj):
        blocks = m.blocks
        if m.held:
            blocks = {k: list(v) for k, v in m.blocks.items()}
            for n, (ox, oy) in m.held.items():
                blocks[n][0] = gx + ox
                blocks[n][1] = gy - gaj + oy
        return blocks

    def _fast_ok_wps(self, m, cur, wps, vac):
        held = m.held
        obstacles = self._obstacles(m)
        lh = getattr(self, "_last_hit", None)
        if lh is not None and lh in obstacles:
            obstacles.remove(lh)
            obstacles.insert(0, lh)
        held_dims = [(ox, oy, m.blocks[n][2], m.blocks[n][3])
                     for n, (ox, oy) in held.items()]
        x, y, aj = m.rx, m.ry, m.aj
        hw = GRIP_H / 2
        c = cur
        segs = []
        for w in wps:
            n = seg_steps(c, w)
            if n > 0:
                segs.append((n, float(np.float32((w[0] - c[0]) / n)),
                             float(np.float32((w[1] - c[1]) / n)),
                             float(np.float32((w[2] - c[2]) / n))))
            c = w
        total = sum(sg[0] for sg in segs)
        k = 0
        for n, sx, sy, sa in segs:
            for _ in range(n):
                k += 1
                x += sx
                y += sy
                aj = min(max(aj + sa, ARM_MIN), ARM_MAX)
                gb = y - aj
                g = (x - hw, gb - GRIP_W / 2, x + hw, gb + GRIP_W / 2)
                zone = None
                if held and vac > 0.5 and k < total:
                    zone = (x - hw, gb - 2 * GRIP_W, x + hw, gb - GRIP_W)
                for ob in obstacles:
                    if ob[3] + EPS + BASE_R < y and ob[3] < gb - 0.25:
                        continue
                    px = min(max(x, ob[0]), ob[2])
                    py = min(max(y, ob[1]), ob[3])
                    if (px - x) ** 2 + (py - y) ** 2 < (BASE_R + EPS) ** 2:
                        self._last_hit = ob
                        return False
                    if (g[0] < ob[2] + EPS and ob[0] < g[2] + EPS and
                            g[1] < ob[3] + EPS and ob[1] < g[3] + EPS):
                        self._last_hit = ob
                        return False
                    if (x - ARM_HALF_W < ob[2] + EPS and ob[0] < x + ARM_HALF_W + EPS
                            and gb < ob[3] + EPS and ob[1] < y + EPS):
                        self._last_hit = ob
                        return False
                    for ox, oy, w, h in held_dims:
                        bx, by = x + ox, gb + oy
                        if (bx < ob[2] + HELD_EPS and ob[0] < bx + w + HELD_EPS and
                                by < ob[3] + HELD_EPS and ob[1] < by + h + HELD_EPS):
                            self._last_hit = ob
                            return False
                    if zone is not None and _aabb_overlap(zone, ob, eps=0.0):
                        self._last_hit = ob
                        return False
        return True

    def _seg_checker(self, m, vac):
        """Returns seg_ok(start, end, is_last) with memoisation; checks the
        poses visited when interpolating from start to end."""
        held = m.held
        obstacles = self._obstacles(m)
        held_dims = [(ox, oy, m.blocks[n][2], m.blocks[n][3])
                     for n, (ox, oy) in held.items()]
        hw = GRIP_H / 2
        cache = {}
        use_zone = bool(held) and vac > 0.5
        state = {"lh": None}

        def seg_ok(c, w, is_last):
            key = (c, w, is_last)
            r = cache.get(key)
            if r is not None:
                return r
            n = seg_steps(c, w)
            if n == 0:
                cache[key] = True
                return True
            sx = float(np.float32((w[0] - c[0]) / n))
            sy = float(np.float32((w[1] - c[1]) / n))
            sa = float(np.float32((w[2] - c[2]) / n))
            x, y, aj = c
            obs = obstacles
            lh = state["lh"]
            if lh is not None:
                obs = [lh] + [o for o in obstacles if o is not lh]
            ok = True
            for k in range(n):
                x += sx
                y += sy
                aj = min(max(aj + sa, ARM_MIN), ARM_MAX)
                gb = y - aj
                g0, g1, g2, g3 = x - hw, gb - GRIP_W / 2, x + hw, gb + GRIP_W / 2
                zone = use_zone and not (is_last and k == n - 1)
                for ob in obs:
                    if ob[3] + EPS + BASE_R < y and ob[3] < gb - 0.25:
                        continue
                    px = min(max(x, ob[0]), ob[2])
                    py = min(max(y, ob[1]), ob[3])
                    hit = (px - x) ** 2 + (py - y) ** 2 < (BASE_R + EPS) ** 2
                    if not hit:
                        hit = (g0 < ob[2] + EPS and ob[0] < g2 + EPS and
                               g1 < ob[3] + EPS and ob[1] < g3 + EPS)
                    if not hit:
                        hit = (x - ARM_HALF_W < ob[2] + EPS and
                               ob[0] < x + ARM_HALF_W + EPS and
                               gb < ob[3] + EPS and ob[1] < y + EPS)
                    if not hit:
                        for ox, oy, bw, bh in held_dims:
                            bx, by = x + ox, gb + oy
                            if (bx < ob[2] + HELD_EPS and ob[0] < bx + bw + HELD_EPS
                                    and by < ob[3] + HELD_EPS and
                                    ob[1] < by + bh + HELD_EPS):
                                hit = True
                                break
                    if not hit and zone:
                        hit = (g0 < ob[2] and ob[0] < g2 and
                               gb - 2 * GRIP_W < ob[3] and ob[1] < gb - GRIP_W)
                    if hit:
                        state["lh"] = ob
                        ok = False
                        break
                if not ok:
                    break
            cache[key] = ok
            return ok

        def path_ok(cur, wps):
            c = cur
            last = len(wps) - 1
            for i, w in enumerate(wps):
                if not seg_ok(c, w, i == last):
                    return False
                c = w
            return True
        return path_ok

    def _obstacles(self, m):
        return list(STATIC_BOXES) + [Model.box(b) for n, b in m.blocks.items()
                                     if n not in m.held]

    def _fast_ok(self, m, acts):
        """Quick collision check of an action sequence assuming the held set
        stays constant (and no extra block gets suctioned while vac is on)."""
        held = m.held
        obstacles = list(STATIC_BOXES) + [Model.box(b) for n, b in m.blocks.items()
                                          if n not in held]
        held_dims = [(ox, oy, m.blocks[n][2], m.blocks[n][3])
                     for n, (ox, oy) in held.items()]
        x, y, aj = m.rx, m.ry, m.aj
        hw = GRIP_H / 2
        n_acts = len(acts)
        for i, a in enumerate(acts):
            x += float(a[0])
            y += float(a[1])
            aj = min(max(aj + float(a[3]), ARM_MIN), ARM_MAX)
            g = (x - hw, y - aj - GRIP_W / 2, x + hw, y - aj + GRIP_W / 2)
            arm = (x - ARM_HALF_W, y - aj, x + ARM_HALF_W, y)
            hb = [(x + ox, y - aj + oy, x + ox + w, y - aj + oy + h)
                  for ox, oy, w, h in held_dims]
            zone = None
            if held and i < n_acts - 1 and a[4] > 0.5:
                zone = (x - hw, y - aj - 2 * GRIP_W, x + hw, y - aj - GRIP_W)
            for ob in obstacles:
                if ob[3] + EPS + BASE_R < y and ob[3] + HELD_EPS < y - aj - 0.2 - 0.03:
                    continue  # far below everything
                if _circle_aabb(x, y, BASE_R, ob):
                    return False
                if _aabb_overlap(g, ob) or _aabb_overlap(arm, ob):
                    return False
                for b in hb:
                    if _aabb_overlap(b, ob, eps=HELD_EPS):
                        return False
                if zone is not None and _aabb_overlap(zone, ob, eps=0.0):
                    return False
        return True

    def _travel(self, m, gx, gy, gaj, vac, last_vac, max_checks=100,
                bound=None):
        """Cheapest collision-free path (by step count) from the current pose
        to (gx, gy, gaj), found by enumerating 0/1/2-waypoint paths.
        The last action uses vacuum value last_vac.
        Returns (cost, acts, model, preds) or None."""
        cur = np.array([m.rx, m.ry, m.aj])
        goal = np.array([gx, gy, gaj])
        if m.collides(gx, gy, gaj, self._held_blocks_at(m, gx, gy, gaj),
                      m.held.keys()):
            return None
        lim = np.array([MAX_D, MAX_D, MAX_ARM]) * 0.999

        def steps(a, b):
            return np.ceil(np.max(np.abs(b - a) / lim, axis=-1) - 1e-9)
        direct = steps(cur, goal)
        if bound is not None and max(direct, 1) >= bound:
            return None
        lo_h = min(cur[1], gy)
        heights = np.concatenate([np.arange(lo_h, TOP_Y, 0.025), [TOP_Y]])
        fr = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        dx = gx - cur[0]
        H, A, F = np.meshgrid(heights, np.array([ARM_MIN, ARM_MAX]), fr,
                              indexing="ij")
        W = np.stack([cur[0] + F * dx, H, A], axis=-1).reshape(-1, len(fr), 3)
        c_a = steps(cur, W)  # (G, 5)
        c_b = steps(W, goal)  # (G, 5)
        G = W.shape[0]
        nf = len(fr)
        ii, jj = np.triu_indices(nf, 1)
        c1 = (c_a + c_b).reshape(-1)
        c_mid = steps(W[:, ii, :], W[:, jj, :])  # (G, P)
        c2 = (c_a[:, ii] + c_mid + c_b[:, jj]).reshape(-1)
        h1 = W[:, :, 1].reshape(-1)
        h2 = W[:, ii, 1].reshape(-1)
        P = len(ii)
        costs = np.concatenate([[direct], c1, c2])
        kinds = np.concatenate([[0], np.ones(G * nf), 2 * np.ones(G * P)])
        hs = np.concatenate([[0.0], h1, h2])
        if bound is not None:
            mask = np.maximum(costs, 1) < bound
        else:
            mask = np.ones(len(costs), dtype=bool)
        idxs = np.nonzero(mask)[0]
        order = idxs[np.lexsort((kinds[idxs], -hs[idxs], costs[idxs]))]
        checks = 0
        curt = tuple(float(v) for v in cur)
        goalt = tuple(float(v) for v in goal)

        def finish(wps, kind):
            acts = []
            c = curt
            for w in wps:
                acts += move_actions(c, w, vac)
                c = w
            if not acts:
                acts = [np.array([0, 0, 0, 0, last_vac], dtype=np.float32)]
            else:
                acts[-1] = acts[-1].copy()
                acts[-1][4] = last_vac
            if kind == -1 and bound is not None and len(acts) >= bound:
                return None
            if self._fast_ok(m, acts):
                m2, preds = self._run(m, acts)
                if m2 is not None:
                    return len(acts), acts, m2, preds
            return None

        path_ok = self._seg_checker(m, vac)
        for k in order[:max_checks]:
            if k == 0:
                wps = [goalt]
            elif k <= G * nf:
                q = k - 1
                wps = [tuple(W[q // nf, q % nf]), goalt]
            else:
                q = k - 1 - G * nf
                gi, p = q // P, q % P
                wps = [tuple(W[gi, ii[p]]), tuple(W[gi, jj[p]]), goalt]
            if not path_ok(curt, wps):
                continue
            r = finish(wps, int(kinds[k]))
            if r is not None:
                return r
        # safe fallback: straight up to the top, across, and down
        wps = [(curt[0], TOP_Y, ARM_MIN), (goalt[0], TOP_Y, ARM_MIN), goalt]
        return finish(wps, -1)

    def _grasp_xs(self, b):
        cx = b[0] + b[2] / 2
        half = b[2] / 2
        xs = [cx]
        for d in (0.35, 0.7, 1.0):
            xs += [cx - d * half, cx + d * half]
        lo_r = BASE_R + EPS + 1e-4
        hi_r = WORLD_W - BASE_R - EPS - 1e-4
        lo_b = b[0] - GRIP_H / 2 + 0.005
        hi_b = b[0] + b[2] + GRIP_H / 2 - 0.005
        lo, hi = max(lo_r, lo_b), min(hi_r, hi_b)
        if lo > hi:
            return []
        out = []
        for x in xs + [lo, hi]:
            x = min(max(x, lo), hi)
            if all(abs(x - y) > 1e-3 for y in out):
                out.append(x)
        return out

    def _picks(self, m, name, k=None):
        """Feasible grasps sorted by cost: list of (cost, acts, model, preds)."""
        key = (name, m.rx, m.ry, m.aj, m.vac,
               tuple(sorted((n, tuple(b)) for n, b in m.blocks.items())))
        cache = self.__dict__.setdefault("_pick_cache", {})
        if key not in cache:
            cache[key] = self._picks_impl(m, name)
        res = cache[key]
        return res if k is None else res[:k]

    def _picks_impl(self, m, name):
        b = m.blocks[name]
        top = b[1] + b[3]
        res = []
        for gx in self._grasp_xs(b):
            best = None
            for aj in (ARM_MAX, 0.15, ARM_MIN):
                gy = top + aj + GRASP_GAP
                if gy > TOP_Y:
                    continue
                r = self._travel(m, gx, gy, aj, 0.0, 1.0,
                                 bound=None if best is None else best[0] + 1)
                if r is None:
                    continue
                if set(r[2].held) != {name}:
                    continue
                if best is None or r[0] < best[0]:
                    best = r
            if best is not None:
                res.append(best)
        res.sort(key=lambda r: r[0])
        return res

    def _place(self, m, name, px, py, bound=None):
        """Move held block so its bottom-left is at (px, py) and release
        (release happens on the final step)."""
        ox, oy = m.held[name]
        best = None
        for aj in (ARM_MAX, 0.15, ARM_MIN):
            rx = px - ox
            ry = py - oy + aj
            if ry > TOP_Y or ry < 0:
                continue
            b = bound if best is None else (best[0] if bound is None else min(bound, best[0]))
            r = self._travel(m, rx, ry, aj, 1.0, 0.0, bound=b)
            if r is None:
                continue
            if best is None or r[0] < best[0]:
                best = r
        return best

    def _free_spots(self, m, name, forbid, mode, corridors=(), anchor=None):
        """Candidate (px, py) bottom-left placements for block `name`."""
        w, h = m.blocks[name][2], m.blocks[name][3]
        cur = m.blocks[name][0] if anchor is None else anchor - w / 2
        margin = 0.01
        pys = [TABLE_TOP + PLACE_LIFT]
        if mode == "float":
            pys += [0.13, 0.17, 0.21, 0.25, 0.3, 0.35, 0.4]
        out_all = []
        for py in pys:
            occupied = list(forbid)
            if py > TABLE_TOP + PLACE_LIFT + 1e-9:
                occupied += list(corridors)
            for n, b in m.blocks.items():
                if n == name:
                    continue
                if b[1] < py + h + margin and b[1] + b[3] > py - margin:
                    occupied.append((b[0], b[0] + b[2]))
            spots = []
            for px in np.arange(0.0, WORLD_W - w + 1e-9, 0.005):
                lo, hi = px - margin, px + w + margin
                if all(hi <= a or lo >= b for a, b in occupied):
                    spots.append(float(px))

            def clearance(px):
                lo, hi = px, px + w
                c = 1.0
                for a, b in occupied:
                    if hi <= a:
                        c = min(c, a - hi)
                    elif lo >= b:
                        c = min(c, lo - b)
                return c
            if mode == "clear":
                spots.sort(key=lambda px: (-min(round(clearance(px), 3), 0.1),
                                           abs(px - cur)))
                sep, k = 0.02, 12
            else:
                spots.sort(key=lambda px: abs(px - cur))
                sep, k = 0.015, (16 if mode == "cost" else 6)
            out = []
            for p in spots:
                if all(abs(p - q) > sep for q in out):
                    out.append(p)
                if len(out) >= k:
                    break
            out_all += [(p, py) for p in out]
        return out_all

    def _slot_xs(self, m, target, surface):
        w = m.blocks[target][2]
        sx, sw = surface[0], surface[2]
        lo, hi = sx + 0.002, sx + sw - w - 0.002
        if hi < lo:
            return [sx + (sw - w) / 2]
        return list(np.linspace(lo, hi, max(2, int((hi - lo) / 0.002) + 1)))

    def _slot_blockers(self, m, target, px, obst):
        w, h = m.blocks[target][2], m.blocks[target][3]
        res = []
        for n in obst:
            b = m.blocks[n]
            if (b[0] < px + w + 0.004 and b[0] + b[2] > px - 0.004 and
                    b[1] < TABLE_TOP + PLACE_LIFT + h + 0.004):
                res.append(n)
        return res

    def _place_xs_target(self, m, target, surface):
        obst = [n for n in m.blocks if n != target]
        sx, sw = surface[0], surface[2]
        w = m.blocks[target][2]
        center = sx + (sw - w) / 2
        free = [px for px in self._slot_xs(m, target, surface)
                if not self._slot_blockers(m, target, px, obst)]
        free.sort(key=lambda px: abs(px - center))
        out = []
        for px in free:
            if all(abs(px - q) > 0.006 for q in out):
                out.append(px)
            if len(out) >= 5:
                break
        return out

    def _reachable(self, m, n):
        return bool(self._grasp_xs(m.blocks[n]))

    def _try_target(self, m, target, surface, deadline):
        best = None
        for c1, acts1, m1, p1 in self._picks(m, target):
            for px in self._place_xs_target(m1, target, surface):
                if time.time() > deadline:
                    break
                r = self._place(m1, target, px, TABLE_TOP + PLACE_LIFT,
                                bound=None if best is None else best[0] - c1)
                if r is not None:
                    c = c1 + r[0]
                    if best is None or c < best[0]:
                        best = (c, acts1 + r[1], p1 + r[3])
        if best is None:
            return None
        return best[1], best[2]

    def _try_move(self, m, name, forbid, deadline, mode, next_x,
                  corridors=(), anchor=None):
        spots = self._free_spots(m, name, forbid, mode, corridors, anchor)
        if mode != "clear":
            b0 = m.blocks[name]
            ax = b0[0] if anchor is None else anchor - b0[2] / 2
            spots.sort(key=lambda p: max(abs(p[0] - ax), abs(p[1] - b0[1])))
        best = None
        picks = self._picks(m, name)
        for pi, (c1, acts1, m1, p1) in enumerate(picks):
            if pi >= 4 and best is not None:
                break
            for px, py in spots:
                if time.time() > deadline:
                    break
                bnd = None
                if best is not None and mode != "clear":
                    bnd = best[0] - c1
                r = self._place(m1, name, px, py, bound=bnd)
                if r is None:
                    continue
                c2, acts2, m2, p2 = r
                c = c1 + c2
                if mode != "clear":
                    c += abs(m2.rx - next_x) / MAX_D
                if best is None or c < best[0]:
                    best = (c, acts1 + acts2, m2, p1 + p2)
        if best is None:
            return None
        return best[1], best[3], best[2]

    def _blocking(self, m, obst, surface, target, exclude=()):
        """Obstructions to remove so that some slot on the surface is free.
        Chooses the slot needing the fewest (reachable) removals."""
        sx, sw = surface[0], surface[2]
        w = m.blocks[target][2]
        center = sx + (sw - w) / 2
        best = None
        for px in self._slot_xs(m, target, surface):
            bl = self._slot_blockers(m, target, px, obst)
            unreach = sum(1 for n in bl if n in exclude or not self._reachable(m, n))
            key = (unreach, len(bl), abs(px - center))
            if best is None or key < best[0]:
                best = (key, bl)
        return [] if best is None else best[1]

    def _plan_mode(self, m, target, surface, obst, deadline, mode,
                   order="tall", park_mode="float", limit=None):
        sx, sw = surface[0], surface[2]
        forbid = [(sx - 0.005, sx + sw + 0.005)]
        acts_all, preds_all = [], []
        if m.held:
            a_off = np.array([0, 0, 0, 0, 0.0], dtype=np.float32)
            m2, p = self._run(m, [a_off])
            if m2 is not None:
                m = m2
                acts_all += [a_off]
                preds_all += p
        moved = set()
        cw = BASE_R + 0.01
        if mode == "park" and self._blocking(m, obst, surface, target):
            r = self._try_move(m, target, forbid, deadline, "float",
                               sx + sw / 2, [(sx - cw, sx + sw + cw)],
                               anchor=sx + sw / 2)
            if r is None:
                return acts_all, preds_all, False
            acts, preds, m = r
            acts_all += acts
            preds_all += preds
            mode = park_mode
        tb = m.blocks[target]
        next_x = tb[0] + tb[2] / 2
        corridors = [(sx - cw, sx + sw + cw), (tb[0] - cw, tb[0] + tb[2] + cw)]
        while time.time() < deadline:
            if limit is not None and len(acts_all) >= limit:
                return acts_all, preds_all, False
            blocking = [n for n in self._blocking(m, obst, surface, target)
                        if n not in moved]
            if not blocking:
                r = self._try_target(m, target, surface, deadline)
                if r is not None:
                    return acts_all + r[0], preds_all + r[1], True
                scx = sx + sw / 2
                rest = [n for n in obst if n not in moved]
                rest.sort(key=lambda n: abs(m.blocks[n][0] + m.blocks[n][2] / 2
                                            - scx))
                blocking = rest[:2]
                if not blocking:
                    break
            if order == "tall":
                blocking.sort(key=lambda n: -m.blocks[n][3])
            elif order == "near":
                blocking.sort(key=lambda n: abs(m.blocks[n][0] + m.blocks[n][2]
                                                / 2 - m.rx))
            elif order == "short":
                blocking.sort(key=lambda n: m.blocks[n][3])
            else:
                rng = getattr(self, "_rng", None)
                if rng is None:
                    rng = self._rng = np.random.default_rng(0)
                rng.shuffle(blocking)
            progress = False
            for n in blocking:
                r = self._try_move(m, n, forbid, deadline, mode, next_x,
                                   corridors)
                if r is not None:
                    acts, preds, m = r
                    acts_all += acts
                    preds_all += preds
                    moved.add(n)
                    progress = True
                    break
            if not progress:
                break
        return acts_all, preds_all, False

    def _plan(self, state, budget=40.0):
        t_start = time.time()
        self._pick_cache = {}
        m, target, surface, obst = self._parse(state)
        best = None
        partial = None
        variants = VARIANTS
        nv = len(variants)
        for i, (mode, order, pm) in enumerate(variants):
            left = budget - (time.time() - t_start)
            if left <= 0:
                break
            frac = 1.0 / (nv - i - 0.5) if i < nv - 1 else 1.0
            acts, preds, ok = self._plan_mode(m, target, surface, obst,
                                              time.time() + left * frac, mode,
                                              order, pm)
            if ok and (best is None or len(acts) < len(best[0])):
                best = (acts, preds)
            if not ok and acts and partial is None:
                partial = (acts, preds)
        # randomised restarts while time remains
        rv = [("float", "rand", "float"), ("park", "rand", "float"),
              ("cost", "rand", "float"), ("park", "rand", "cost")]
        self._rng = np.random.default_rng(0)
        k = 0
        while (best is not None and len(obst) >= 2 and k < 16 and
               time.time() - t_start < budget * 0.6):
            mode, order, pm = rv[k % len(rv)]
            k += 1
            acts, preds, ok = self._plan_mode(
                m, target, surface, obst, t_start + budget, mode, order, pm,
                limit=len(best[0]))
            if ok and len(acts) < len(best[0]):
                best = (acts, preds)
        if best is None:
            best = partial if partial is not None else ([], [])
        return best

    # ------------------------------------------------------------------ API
    def _replan(self, state, budget):
        try:
            self.plan, self.pred = self._plan(state, budget)
        except Exception:  # pragma: no cover - safety net
            self.plan, self.pred = [], []
        self.idx = 0

    def _budget(self, want):
        left = 45.0 - (time.time() - self.t0)
        return max(0.0, min(want, left))

    def reset(self, state, info):
        self.t0 = time.time()
        self.fail_count = 0
        self._replan(state, self._budget(30.0))

    def _robot_xy(self, state):
        for o in state:
            if o.type.name == "crv_robot":
                return (float(state.get(o, "x")), float(state.get(o, "y")),
                        float(state.get(o, "arm_joint")))
        return None

    def get_action(self, state):
        if 0 < self.idx <= len(self.pred):
            exp = self.pred[self.idx - 1]
            act = self._robot_xy(state)
            if max(abs(a - b) for a, b in zip(exp, act)) > 1e-5:
                self.fail_count += 1
                if self.fail_count <= 5:
                    self._replan(state, self._budget(5.0))
        if self.idx >= len(self.plan):
            self.fail_count += 1
            if self.fail_count <= 5:
                self._replan(state, self._budget(5.0))
            else:
                self.plan, self.pred, self.idx = [], [], 0
            if not self.plan:
                return np.array([0, 0, 0, 0, 0], dtype=np.float32)
        a = self.plan[self.idx]
        self.idx += 1
        lo, hi = self.action_space.low, self.action_space.high
        return np.clip(a, lo, hi).astype(np.float32)
