"""Approach for ClutteredRetrieval2DEnv.

Model (discovered empirically)
-----------------------------
* Rect objects report their LOWER-LEFT CORNER; center = corner + (w/2)u + (h/2)v.
* Robot: base disk r=0.1 at (x,y); gripper center at
  (x+arm_joint*cos th, y+arm_joint*sin th), arm_joint in [0.1, 0.2].
  Suction (vac>0.5) grabs any object whose surface is within 0.02 in front of
  the gripper center (+-0.035 laterally); the grasp is a rigid attachment.
* World is [0,2.5]^2. Actions are exact and atomic: if the resulting pose
  collides (base, arm, gripper or held object), the whole action is rejected.
* target_region is not a collision body. Terminates when the block is fully
  inside the region.

Planning
--------
Grid search (x, y, theta) with Dijkstra. Obstructions are *soft*: crossing one
costs a large penalty, so the shortest path reveals the minimum set of
obstructions that must be dragged out of the way first.
"""
import math
import time
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra

ARENA_LO, ARENA_HI = 0.0, 2.5
BASE_R = 0.1
FOOT_R = 0.112
MARGIN = 0.004
GRIP_GAP = 0.012
WALL_R = 0.107      # base-center limit from a wall (gripper may add ~0.005)
WALL_M = 0.0        # margin for carried objects vs walls (env is exact)
PEN = 80.0
PLAN_BUDGET = 28.0
MAX_SOFT = 62


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def corner_to_center(x, y, th, w, h):
    c, s = math.cos(th), math.sin(th)
    return (x + c * w / 2 - s * h / 2, y + s * w / 2 + c * h / 2)


def halfext(w, h, th, ax):
    c, s = math.cos(th), math.sin(th)
    return abs(w / 2 * (c * ax[0] + s * ax[1])) + abs(h / 2 * (-s * ax[0] + c * ax[1]))


def dist_points_rect(P, rect):
    cx, cy, th, w, h = rect[:5]
    c, s = math.cos(th), math.sin(th)
    dx = P[:, 0] - cx
    dy = P[:, 1] - cy
    u = dx * c + dy * s
    v = -dx * s + dy * c
    du = np.maximum(np.abs(u) - w / 2, 0.0)
    dv = np.maximum(np.abs(v) - h / 2, 0.0)
    return np.hypot(du, dv)


def rects_overlap_points(P, bth, bw, bh, rect, pad=0.0):
    ox, oy, oth, ow, oh = rect[:5]
    bw2, bh2 = bw + 2 * pad, bh + 2 * pad
    diff = P - np.array([ox, oy])
    sep = np.zeros(len(P), dtype=bool)
    for th in (bth, oth):
        for ax in ((math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))):
            proj = diff[:, 0] * ax[0] + diff[:, 1] * ax[1]
            lim = halfext(bw2, bh2, bth, ax) + halfext(ow, oh, oth, ax)
            sep |= np.abs(proj) > lim
            if sep.all():
                return ~sep
    return ~sep


def rect_overlap(r1, r2, pad=0.0):
    P = np.array([[r1[0], r1[1]]])
    return bool(rects_overlap_points(P, r1[2], r1[3], r1[4], r2, pad)[0])


class Grid:
    """(x, y, theta) lattice with soft obstacle costs and Dijkstra paths.

    soft: list of rects (obstructions that could be moved away)
    hard: list of (rect, inflate) that can never be traversed
    carry: list of (rel_xy, rel_th, w, h) rigidly held rectangles
    """

    def __init__(self, ox, oy, oth, res, nth, cells, soft, hard=(),
                 carry=None, foot_r=FOOT_R, margin=MARGIN):
        self.res = res
        self.nth = nth
        self.oth = oth
        self.dth = 2 * math.pi / nth if nth > 1 else 0.0
        self.cells = cells
        r = foot_r + margin
        lo, hi = ARENA_LO + WALL_R, ARENA_HI - WALL_R
        i0 = int(math.ceil((lo - ox) / res)); i1 = int(math.floor((hi - ox) / res))
        j0 = int(math.ceil((lo - oy) / res)); j1 = int(math.floor((hi - oy) / res))
        self.xs = ox + np.arange(i0, i1 + 1) * res
        self.ys = oy + np.arange(j0, j1 + 1) * res
        self.nx = len(self.xs); self.ny = len(self.ys)
        XX, YY = np.meshgrid(self.xs, self.ys)
        self.P = np.stack([XX.ravel(), YY.ravel()], axis=1)
        npt = len(self.P)
        self.nsoft = len(soft)

        # theta-dependent carried-rect centers
        cnt = np.zeros((nth, npt), dtype=np.int16)
        mask = np.zeros((nth, npt), dtype=np.int64)
        hard_free = np.ones((nth, npt), dtype=bool)

        base_hit = []   # per soft obstacle: base-disk collision (theta independent)
        for ob in soft:
            base_hit.append(dist_points_rect(self.P, ob) <= r)
        hard_base = np.ones(npt, dtype=bool)
        for (ob, inf) in hard:
            hard_base &= dist_points_rect(self.P, ob) > (inf + margin)

        for k in range(nth):
            th = self.theta(k)
            c, s = math.cos(th), math.sin(th)
            hits = [b.copy() for b in base_hit]
            hf = hard_base.copy()
            if carry:
                for (rel, rel_th, bw, bh) in carry:
                    off = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
                    C = self.P + off
                    bth = th + rel_th
                    hx = halfext(bw, bh, bth, (1.0, 0.0)) + WALL_M
                    hy = halfext(bw, bh, bth, (0.0, 1.0)) + WALL_M
                    hf &= (C[:, 0] > ARENA_LO + hx) & (C[:, 0] < ARENA_HI - hx)
                    hf &= (C[:, 1] > ARENA_LO + hy) & (C[:, 1] < ARENA_HI - hy)
                    for j, ob in enumerate(soft):
                        hits[j] |= rects_overlap_points(C, bth, bw, bh, ob, pad=margin)
                    for (ob, inf) in hard:
                        hf &= ~rects_overlap_points(C, bth, bw, bh, ob, pad=margin)
            hard_free[k] = hf
            for j in range(self.nsoft):
                cnt[k] += hits[j]
                mask[k] |= hits[j].astype(np.int64) << j
        self.cnt = cnt.reshape(-1)
        self.mask = mask.reshape(-1)
        self.hard_free = hard_free.reshape(-1)
        self.node_cost = 1.0 + PEN * self.cnt.astype(np.float64)
        self.n = nth * npt
        self._graph = None
        self._dist = None
        self._pred = None

    def theta(self, k):
        return wrap(self.oth + k * self.dth)

    def idx(self, k, jy, ix):
        return (k * self.ny + jy) * self.nx + ix

    def snap(self, x, y, th):
        ix = int(round((x - self.xs[0]) / self.res))
        jy = int(round((y - self.ys[0]) / self.res))
        ix = min(max(ix, 0), self.nx - 1); jy = min(max(jy, 0), self.ny - 1)
        k = 0 if self.nth == 1 else int(round(wrap(th - self.oth) / self.dth)) % self.nth
        return k, jy, ix

    def nearest_node(self, x, y, th, max_cells=2, max_dk=0, need_hard_free=True):
        k, jy, ix = self.snap(x, y, th)
        best, bestd = None, 1e18
        hf = self.hard_free.reshape(self.nth, self.ny, self.nx)
        cn = self.cnt.reshape(self.nth, self.ny, self.nx)
        for dk in range(-max_dk, max_dk + 1):
            kk = (k + dk) % self.nth
            for dj in range(-max_cells, max_cells + 1):
                for di in range(-max_cells, max_cells + 1):
                    jj, ii = jy + dj, ix + di
                    if not (0 <= jj < self.ny and 0 <= ii < self.nx):
                        continue
                    if need_hard_free and not hf[kk, jj, ii]:
                        continue
                    d = ((di * self.res) ** 2 + (dj * self.res) ** 2
                         + (dk * self.dth * 0.25) ** 2 + 0.002 * cn[kk, jj, ii])
                    if d < bestd:
                        bestd, best = d, self.idx(kk, jj, ii)
        return best

    def _build(self):
        if self._graph is not None:
            return
        nth, ny, nx = self.nth, self.ny, self.nx
        hf3 = self.hard_free.reshape(nth, ny, nx)
        idx_all = np.arange(self.n).reshape(nth, ny, nx)
        cost3 = self.node_cost.reshape(nth, ny, nx)
        rows, cols, vals = [], [], []
        ks = np.arange(nth)
        dks = [-1, 0, 1] if nth > 1 else [0]
        for dk in dks:
            kd = (ks + dk) % nth
            for dj in self.cells:
                for di in self.cells:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    js0, js1 = max(0, -dj), ny - max(0, dj)
                    is0, is1 = max(0, -di), nx - max(0, di)
                    if js0 >= js1 or is0 >= is1:
                        continue
                    jd = np.arange(js0 + dj, js1 + dj)
                    idd = np.arange(is0 + di, is1 + di)
                    src = idx_all[:, js0:js1, is0:is1]
                    dst = idx_all[kd[:, None, None], jd[None, :, None], idd[None, None, :]]
                    m = hf3[:, js0:js1, is0:is1] & hf3[kd[:, None, None], jd[None, :, None], idd[None, None, :]]
                    rows.append(src[m]); cols.append(dst[m])
                    vals.append(cost3[kd[:, None, None], jd[None, :, None], idd[None, None, :]][m])
        r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
        self._graph = csr_matrix((v.astype(np.float32), (r, c)), shape=(self.n, self.n))

    def solve(self, src_node):
        self._build()
        d, p = _sp_dijkstra(self._graph, directed=True, indices=src_node,
                            return_predecessors=True)
        self._dist, self._pred = d, p

    def path_to(self, node):
        if node is None or self._dist is None or not np.isfinite(self._dist[node]):
            return None, 0
        seq = [node]
        cur = node
        while self._pred[cur] >= 0:
            cur = self._pred[cur]
            seq.append(cur)
        seq.reverse()
        m = 0
        for nd in seq:
            m |= int(self.mask[nd])
        return [self.node_pose(nd) for nd in seq], m

    def node_pose(self, node):
        k = node // (self.ny * self.nx)
        rem = node % (self.ny * self.nx)
        return (self.xs[rem % self.nx], self.ys[rem // self.nx], self.theta(k))


def bits(m):
    out = []
    j = 0
    while m:
        if m & 1:
            out.append(j)
        m >>= 1
        j += 1
    return out


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.low = np.asarray(action_space.low, dtype=np.float64)
        self.high = np.asarray(action_space.high, dtype=np.float64)

    # ------------------------------------------------------------------
    def parse(self, state):
        robot = None; block = None; region = None; obstacles = {}
        for name in sorted(state.get_object_names()):
            o = state.get_object_from_name(name)
            tn = o.type.name
            if tn == 'crv_robot':
                robot = dict(x=float(state.get(o, 'x')), y=float(state.get(o, 'y')),
                             th=float(state.get(o, 'theta')),
                             arm=float(state.get(o, 'arm_joint')),
                             vac=float(state.get(o, 'vacuum')))
                continue
            try:
                w = float(state.get(o, 'width')); h = float(state.get(o, 'height'))
            except Exception:
                continue
            th = float(state.get(o, 'theta'))
            cx, cy = corner_to_center(float(state.get(o, 'x')), float(state.get(o, 'y')), th, w, h)
            rect = (cx, cy, th, w, h)
            if tn == 'target_block':
                block = rect
            elif tn == 'target_region':
                region = rect
            else:
                obstacles[name] = rect
        return robot, block, region, obstacles

    def reset(self, state, info=None):
        try:
            self._reset(state, info)
        except Exception:
            self.task = None
            self.phase = 'nav'
            self.path = []

    def _reset(self, state, info=None):
        self.t = 0
        self.t0 = time.time()
        self.phase = 'nav'
        self.path = []
        self.pi = 0
        self.stall = 0
        self.last_pose = None
        self.task = None
        self.obj_prev = None
        self.grasp_t = 0
        self.release_t = 0
        self.fail_cnt = 0
        self.next_plan_t = 0
        self.carry_arm = 0.1
        self.banned = []
        self.carry_stuck = 0
        self.nograsp = 0
        self._carry_prev = None
        self.extra_hard = []
        self.plan(state)

    def budget(self):
        return PLAN_BUDGET - (time.time() - self.t0)

    # ---------------- grasp candidates ----------------
    def _pick_candidates(self, target):
        cx, cy, th, w, h = target
        out = []
        for k in range(4):
            phi = wrap(th + k * math.pi / 2)
            halfe = (w / 2) if k % 2 == 0 else (h / 2)
            face_half = (h / 2) if k % 2 == 0 else (w / 2)
            n = (math.cos(phi), math.sin(phi))
            tg = (-math.sin(phi), math.cos(phi))
            lats = [0.0]
            for v in (0.025, 0.045):
                if v < face_half:
                    lats += [v, -v]
            for lat in lats:
                for e in (0.10, 0.13, 0.16, 0.20):
                    D = halfe + e + GRIP_GAP
                    out.append(dict(pos=(cx + n[0] * D + tg[0] * lat,
                                         cy + n[1] * D + tg[1] * lat),
                                    th=wrap(phi + math.pi), arm=e, lat=lat,
                                    face=k, phi=phi, halfe=halfe))
        return out

    def _cand_mask(self, cand, soft, hard):
        """Return (ok, soft_bitmask) for a pre-grasp pose."""
        px, py = cand['pos']
        if not (ARENA_LO + WALL_R < px < ARENA_HI - WALL_R):
            return False, 0
        if not (ARENA_LO + WALL_R < py < ARENA_HI - WALL_R):
            return False, 0
        P = np.array([[px, py]])
        for (ob, inf) in hard:
            if dist_points_rect(P, ob)[0] <= inf + 0.003:
                return False, 0
        m = 0
        ang = cand['th']
        L = cand['arm'] + 0.005 - 0.09
        corr = None
        if L > 0.001:
            mid = (0.09 + cand['arm'] + 0.005) / 2
            corr = (px + math.cos(ang) * mid, py + math.sin(ang) * mid, ang, L, 0.076)
        for j, ob in enumerate(soft):
            hit = dist_points_rect(P, ob)[0] <= FOOT_R + 0.004
            if not hit and corr is not None:
                hit = rect_overlap(corr, ob, pad=0.002)
            if hit:
                m |= (1 << j)
        return True, m

    # ---------------- pick & place planning ----------------
    def _plan_pickplace(self, robot, target, soft_names, soft, hard,
                        goal_kind, region=None, block=None, max_cands=8,
                        max_carry=4):
        """Plan grasping `target` and moving it to a goal.

        goal_kind: 'region' (target block into target_region) or 'dump'.
        Returns a task dict with 'blockset' (names of obstructions in the way).
        """
        pick_hard = list(hard) + [(target, BASE_R + 0.003)]
        pg = Grid(robot['x'], robot['y'], robot['th'], 0.025, 1, (-2, -1, 0, 1, 2),
                  soft, pick_hard)
        src = pg.nearest_node(robot['x'], robot['y'], robot['th'], max_cells=4)
        if src is None:
            return None
        pg.solve(src)
        scored = []
        for c in self._pick_candidates(target):
            ok, cm = self._cand_mask(c, soft, pick_hard)
            if not ok:
                continue
            if any(abs(c['pos'][0] - b[0]) < 0.02 and abs(c['pos'][1] - b[1]) < 0.02
                   and abs(wrap(c['th'] - b[2])) < 0.1 for b in getattr(self, 'banned', ())):
                continue
            node = pg.nearest_node(c['pos'][0], c['pos'][1], c['th'], max_cells=2)
            if node is None:
                continue
            d = pg._dist[node]
            if not np.isfinite(d):
                continue
            np_, _ = pg.node_pose(node)[:1], None
            gx, gy, _t = pg.node_pose(node)
            slack = math.hypot(gx - c['pos'][0], gy - c['pos'][1])
            path, pm = pg.path_to(node)
            m = pm | cm
            nv = len(bits(m))
            scored.append((nv, d + slack * 30 + (c['arm'] - 0.1) * 20, c, node, m, path))
        scored.sort(key=lambda z: (z[0], z[1]))
        best = None
        tried = 0
        for (nv, sc, c, node, m, path) in scored[:max_cands]:
            if tried >= max_carry or self.budget() < 3:
                break
            if best is not None and nv >= best[0] and nv > 0:
                continue
            tried += 1
            res = self._plan_carry_for(c, target, soft, hard, goal_kind, region, block)
            if res is None:
                continue
            cpath, cm2, goal_pose = res
            tot_nv = len(bits(m | cm2))
            score = (tot_nv, sc + len(cpath))
            if best is None or score < (best[0], best[1]):
                best = (tot_nv, sc + len(cpath), c, path, cpath, goal_pose, m | cm2)
            if tot_nv == 0:
                break
        if best is None:
            return None
        tot_nv, sc, c, path, cpath, goal_pose, m = best
        blockset = [soft_names[j] for j in bits(m) if j < len(soft_names)]
        return dict(cand=c, navpath=path, carrypath=cpath, goal_pose=goal_pose,
                    blockset=blockset, nviol=tot_nv)

    def _free_base(self, x, y, obstacles):
        if not (ARENA_LO + WALL_R < x < ARENA_HI - WALL_R):
            return False
        if not (ARENA_LO + WALL_R < y < ARENA_HI - WALL_R):
            return False
        P = np.array([[x, y]])
        for ob in obstacles:
            if dist_points_rect(P, ob)[0] <= FOOT_R + 0.004:
                return False
        return True

    def _rel_from_cand(self, cand, target):
        px, py = cand['pos']
        rth = cand['th']
        c, s = math.cos(rth), math.sin(rth)
        dx, dy = target[0] - px, target[1] - py
        return (c * dx + s * dy, -s * dx + c * dy), wrap(target[2] - rth)

    def _carry_rects(self, arm, rel, rel_th, target):
        rects = [(rel, rel_th, target[3], target[4])]
        if arm > 0.105:
            L = arm + 0.005 - 0.09
            rects.append((((0.09 + arm + 0.005) / 2, 0.0), 0.0, L, 0.07))
        return rects

    def _plan_carry_for(self, cand, target, soft, hard, goal_kind, region, block):
        rel, rel_th = self._rel_from_cand(cand, target)
        carry = self._carry_rects(cand['arm'], rel, rel_th, target)
        g = Grid(cand['pos'][0], cand['pos'][1], cand['th'], 0.05, 32, (-1, 0, 1),
                 soft, hard, carry=carry)
        k, jy, ix = g.snap(cand['pos'][0], cand['pos'][1], cand['th'])
        snode = g.idx(k, jy, ix)
        sp = g.node_pose(snode)
        if (abs(sp[0] - cand['pos'][0]) > 1e-6 or abs(sp[1] - cand['pos'][1]) > 1e-6
                or not g.hard_free[snode]):
            return None
        g.solve(snode)
        return self._select_goal(g, rel, rel_th, carry, soft, hard, goal_kind,
                                 region, block, target)

    def _select_goal(self, g, rel, rel_th, carry, soft, hard, goal_kind,
                     region, block, target):
        if goal_kind == 'region':
            offs = [(0.0, 0.0)]
            for a in (0.011, 0.021):
                for u in (-1, 0, 1):
                    for v in (-1, 0, 1):
                        if u or v:
                            offs.append((u * a, v * a))
            rc, rs = math.cos(region[2]), math.sin(region[2])
            gths = [wrap(region[2] + k * math.pi / 2) for k in range(4)]
            gths.sort(key=lambda a: abs(wrap(a - target[2])))
            best = None
            for gth in gths:
                gth_r = wrap(gth - rel_th)
                cg, sg = math.cos(gth_r), math.sin(gth_r)
                for (du, dv) in offs:
                    bxc = region[0] + rc * du - rs * dv
                    byc = region[1] + rs * du + rc * dv
                    gx = bxc - (cg * rel[0] - sg * rel[1])
                    gy = byc - (sg * rel[0] + cg * rel[1])
                    okp, pm = self._pose_mask(gx, gy, gth_r, carry, soft, hard)
                    if not okp:
                        continue
                    node = g.nearest_node(gx, gy, gth_r, max_cells=1, max_dk=1)
                    if node is None:
                        continue
                    gp = g.node_pose(node)
                    if (abs(gp[0] - gx) > 0.051 or abs(gp[1] - gy) > 0.051
                            or abs(wrap(gp[2] - gth_r)) > 0.19):
                        continue
                    path, m = g.path_to(node)
                    if path is None:
                        continue
                    m |= pm
                    nv = len(bits(m))
                    sc = (nv, len(path))
                    if best is None or sc < best[0]:
                        best = (sc, path, m, (gx, gy, gth_r))
                    if nv == 0:
                        return path, m, (gx, gy, gth_r)
            if best is None:
                return None
            return best[1], best[2], best[3]
        # dump: any far-away free pose
        d = g._dist.copy()
        finite = np.isfinite(d)
        if not finite.any():
            return None
        # object center for every node
        K = np.arange(g.nth)
        ths = np.array([g.theta(k) for k in K])
        cs, sn = np.cos(ths), np.sin(ths)
        offx = cs * rel[0] - sn * rel[1]
        offy = sn * rel[0] + cs * rel[1]
        CX = (g.P[:, 0][None, :] + offx[:, None]).reshape(-1)
        CY = (g.P[:, 1][None, :] + offy[:, None]).reshape(-1)
        db = np.hypot(CX - block[0], CY - block[1])
        dr = np.hypot(CX - region[0], CY - region[1])
        dmove = np.hypot(CX - target[0], CY - target[1])
        score = d + np.where(db < 0.62, 400.0, 0.0) + np.where(dr < 0.42, 400.0, 0.0)
        score = score + np.where(dmove < 0.28, 400.0, 0.0)
        score = score - np.minimum(db, 1.0) * 3.0
        score[~finite] = np.inf
        score[g.cnt > 0] += 200.0
        order = np.argsort(score)
        for node in order[:12]:
            if not np.isfinite(score[node]) or score[node] > 300:
                break
            path, m = g.path_to(int(node))
            if path is None:
                continue
            return path, m, g.node_pose(int(node))
        return None

    def _pose_mask(self, x, y, th, rects, soft, hard, margin=0.003):
        if not (ARENA_LO + WALL_R < x < ARENA_HI - WALL_R):
            return False, 0
        if not (ARENA_LO + WALL_R < y < ARENA_HI - WALL_R):
            return False, 0
        P = np.array([[x, y]])
        for (ob, inf) in hard:
            if dist_points_rect(P, ob)[0] <= inf + margin:
                return False, 0
        m = 0
        c, s_ = math.cos(th), math.sin(th)
        Cs = []
        for (rel, rel_th, bw, bh) in rects:
            C = np.array([[x + c * rel[0] - s_ * rel[1], y + s_ * rel[0] + c * rel[1]]])
            bth = th + rel_th
            hx = halfext(bw, bh, bth, (1.0, 0.0)) + WALL_M
            hy = halfext(bw, bh, bth, (0.0, 1.0)) + WALL_M
            if not (ARENA_LO + hx < C[0, 0] < ARENA_HI - hx):
                return False, 0
            if not (ARENA_LO + hy < C[0, 1] < ARENA_HI - hy):
                return False, 0
            for (ob, inf) in hard:
                if rects_overlap_points(C, bth, bw, bh, ob, pad=margin)[0]:
                    return False, 0
            Cs.append((C, bth, bw, bh))
        for j, ob in enumerate(soft):
            if dist_points_rect(P, ob)[0] <= FOOT_R + margin:
                m |= (1 << j)
                continue
            for (C, bth, bw, bh) in Cs:
                if rects_overlap_points(C, bth, bw, bh, ob, pad=margin)[0]:
                    m |= (1 << j)
                    break
        return True, m

    # ---------------- top level planner ----------------
    def plan(self, state):
        robot, block, region, obstacles = self.parse(state)
        self.region = region
        if self.budget() < 3:
            return False
        names = sorted(obstacles.keys())
        if len(names) > MAX_SOFT:
            names.sort(key=lambda nm: (obstacles[nm][0] - block[0]) ** 2
                       + (obstacles[nm][1] - block[1]) ** 2)
            extra = names[MAX_SOFT:]
            names = names[:MAX_SOFT]
            self.extra_hard = [(obstacles[n], FOOT_R) for n in extra]
        else:
            self.extra_hard = []
        soft = [obstacles[n] for n in names]
        nobs = len(names)
        mc, mk = (8, 4) if nobs <= 30 else (5, 2)
        main = self._plan_pickplace(robot, block, names, soft, list(self.extra_hard),
                                    'region', region=region, block=block,
                                    max_cands=mc, max_carry=mk)
        if main is not None:
            main['allobs'] = soft + [block]
        if main is not None and main['nviol'] == 0:
            main['obj'] = 'target_block'
            main['release'] = False
            main['target'] = block
            self._set_task(main)
            return True
        # decide which obstruction to clear first
        cand_names = list(main['blockset']) if main else []
        if not cand_names:
            cand_names = sorted(names, key=lambda nm: (obstacles[nm][0] - block[0]) ** 2
                                + (obstacles[nm][1] - block[1]) ** 2)[:4]
        tried = set()
        frontier = list(cand_names)
        fallback = None
        for depth in range(2):
            nxt = []
            for nm in frontier:
                if nm in tried or self.budget() < 5:
                    continue
                tried.add(nm)
                sub_names = [n for n in names if n != nm]
                sub_soft = [obstacles[n] for n in sub_names]
                hard = [(block, BASE_R + 0.003)] + list(self.extra_hard)
                t = self._plan_pickplace(robot, obstacles[nm], sub_names, sub_soft,
                                         hard, 'dump', region=region, block=block,
                                         max_cands=6 if nobs <= 30 else 4, max_carry=2)
                if t is None:
                    continue
                t['obj'] = nm
                t['release'] = True
                t['target'] = obstacles[nm]
                t['allobs'] = sub_soft + [block, obstacles[nm]]
                if t['nviol'] == 0:
                    self._set_task(t)
                    return True
                if fallback is None or t['nviol'] < fallback['nviol']:
                    fallback = t
                nxt.extend(t['blockset'])
            frontier = [n for n in nxt if n not in tried]
            if not frontier:
                break
        if fallback is not None:
            self._set_task(fallback)
            return True
        if main is not None:
            main['obj'] = 'target_block'
            main['release'] = False
            main['target'] = block
            self._set_task(main)
            return True
        return False

    def _set_task(self, task):
        self.task = task
        th = task['cand']['th']
        self.path = [(p[0], p[1], th) for p in task['navpath']]
        pos = task['cand']['pos']
        phi = task['cand']['phi']
        bo = (pos[0] + math.cos(phi) * 0.05, pos[1] + math.sin(phi) * 0.05, th)
        if self._free_base(bo[0], bo[1], task.get('allobs', ())):
            self.path.append(bo)
        self.path.append((pos[0], pos[1], th))
        ded = []
        for p in self.path:
            if not ded or abs(p[0] - ded[-1][0]) > 1e-9 or abs(p[1] - ded[-1][1]) > 1e-9:
                ded.append(p)
        self.path = ded
        self.pi = 0
        self.phase = 'nav'
        self.grasp_t = 0

    # ---------------- low level ----------------
    def act(self, dx=0.0, dy=0.0, dth=0.0, darm=0.0, vac=0.0):
        a = np.array([dx, dy, dth, darm, vac], dtype=np.float64)
        return np.clip(a, self.low, self.high).astype(np.float32)

    def _drive(self, robot, target, vac, arm_target=0.1):
        dx = target[0] - robot['x']
        dy = target[1] - robot['y']
        dth = wrap(target[2] - robot['th'])
        darm = arm_target - robot['arm']
        v = self.stall
        if v == 1:
            dx *= 0.5; dy *= 0.5; dth *= 0.5
        elif v == 2:
            dy = 0.0; dth = 0.0; darm = 0.0
        elif v == 3:
            dx = 0.0; dth = 0.0; darm = 0.0
        elif v == 4:
            dx = 0.0; dy = 0.0; darm = 0.0
        elif v == 5:
            dx *= 0.25; dy *= 0.25; dth = 0.0; darm = 0.0
        return self.act(dx, dy, dth, darm, vac)

    def _reached(self, robot, target, arm_target=None, tol=2e-4, thtol=2e-3):
        if abs(robot['x'] - target[0]) > tol or abs(robot['y'] - target[1]) > tol:
            return False
        if abs(wrap(robot['th'] - target[2])) > thtol:
            return False
        if arm_target is not None and abs(robot['arm'] - arm_target) > 1e-3:
            return False
        return True

    # ---------------- main ----------------
    def get_action(self, state):
        try:
            return self._get_action(state)
        except Exception:
            # never crash: fall back to a safe no-op and try to recover next step
            self.task = None
            self.phase = 'nav'
            self.path = []
            self.next_plan_t = getattr(self, 't', 0) + 5
            return self.act()

    def _get_action(self, state):
        self.t += 1
        robot, block, region, obstacles = self.parse(state)
        self.region = region
        pose = (robot['x'], robot['y'], robot['th'], robot['arm'])
        moved = self.last_pose is None or any(abs(a - b) > 1e-7 for a, b in zip(pose, self.last_pose))
        self.stall = 0 if moved else self.stall + 1
        self.last_pose = pose

        if self.task is None:
            if self.t < self.next_plan_t or not self.plan(state):
                self.next_plan_t = self.t + 25
                return self.act()
        obj = self.task['obj']
        cur_obj = block if obj == 'target_block' else obstacles.get(obj)
        if self.phase == 'nav':
            return self._do_nav(state, robot, cur_obj)
        if self.phase == 'grasp':
            return self._do_grasp(state, robot, cur_obj)
        if self.phase == 'retract':
            return self._do_retract(state, robot, cur_obj)
        if self.phase == 'carry':
            return self._do_carry(state, robot, cur_obj)
        if self.phase == 'release':
            return self._do_release(state, robot)
        return self.act()

    def _ban_current(self):
        if self.task is not None:
            c = self.task['cand']
            self.banned.append((c['pos'][0], c['pos'][1], c['th']))

    def _abandon(self, state):
        self.fail_cnt += 1
        self.task = None
        self.stall = 0
        self.phase = 'nav'
        if self.fail_cnt > 12 or self.budget() < 3:
            return self.act()
        if not self.plan(state):
            self.next_plan_t = self.t + 25
            return self.act()
        robot, block, region, obstacles = self.parse(state)
        obj = self.task['obj']
        cur = block if obj == 'target_block' else obstacles.get(obj)
        return self._do_nav(state, robot, cur)

    def _do_nav(self, state, robot, cur_obj):
        goal = self.path[-1] if self.path else None
        if goal is not None and abs(robot['arm'] - 0.1) < 1e-3:
            if (abs(robot['x'] - goal[0]) < 0.006 and abs(robot['y'] - goal[1]) < 0.006
                    and abs(wrap(robot['th'] - goal[2])) < 0.02):
                self.pi = len(self.path)
        while self.pi < len(self.path) and self._reached(robot, self.path[self.pi], 0.1):
            self.pi += 1
        # skip intermediate waypoints we are already essentially at
        while (self.pi < len(self.path) - 1
               and abs(robot['x'] - self.path[self.pi][0]) < 0.004
               and abs(robot['y'] - self.path[self.pi][1]) < 0.004
               and abs(wrap(robot['th'] - self.path[self.pi][2])) < 0.02):
            self.pi += 1
        if self.pi >= len(self.path):
            self.phase = 'grasp'
            self.grasp_t = 0
            self.obj_prev = cur_obj
            return self._do_grasp(state, robot, cur_obj)
        if self.stall >= 6:
            if self.pi + 1 < len(self.path):
                self.pi += 1
                self.stall = 0
                return self._drive(robot, self.path[self.pi], 0.0, 0.1)
            if self.task is not None:
                c = self.task['cand']
                self.banned.append((c['pos'][0], c['pos'][1], c['th']))
            return self._abandon(state)
        return self._drive(robot, self.path[self.pi], 0.0, 0.1)

    def _do_grasp(self, state, robot, cur_obj):
        if self.obj_prev is not None and cur_obj is not None:
            if any(abs(a - b) > 1e-7 for a, b in zip(cur_obj[:3], self.obj_prev[:3])):
                self.banned = []
                self.carry_stuck = 0
                self.nograsp = 0
                self.phase = 'retract'
                self.retract_t = 0
                return self.act(0.0, 0.0, 0.0, -0.05, 1.0)
        self.obj_prev = cur_obj
        self.grasp_t += 1
        target_arm = self.task['cand']['arm']
        if cur_obj is not None and self.grasp_t > 1:
            gx = robot['x'] + robot['arm'] * math.cos(robot['th'])
            gy = robot['y'] + robot['arm'] * math.sin(robot['th'])
            gd = float(dist_points_rect(np.array([[gx, gy]]), cur_obj)[0])
            if gd <= 0.0205:
                # suction window reached: assume attached (motion may be blocked)
                self.phase = 'retract'
                self.retract_t = 0
                self.nograsp = 0
                return self.act(0.0, 0.0, 0.0, -0.05, 1.0)
        if self.grasp_t == 1:
            if robot['arm'] < target_arm - 1e-4:
                return self.act(0.0, 0.0, 0.0, min(0.02, target_arm - robot['arm']), 1.0)
            return self.act(vac=1.0)
        if self.grasp_t <= 8 and robot['arm'] < min(0.2, target_arm + 0.02):
            return self.act(0.0, 0.0, 0.0, 0.01, 1.0)
        if self.grasp_t <= 14:
            c, s = math.cos(robot['th']), math.sin(robot['th'])
            return self.act(c * 0.005, s * 0.005, 0.0, 0.0, 1.0)
        if self.task is not None:
            c = self.task['cand']
            self.banned.append((c['pos'][0], c['pos'][1], c['th']))
        return self._abandon(state)

    def _start_carry(self, state, robot, cur_obj):
        rb, block, region, obstacles = self.parse(state)
        task = self.task
        names = [n for n in sorted(obstacles.keys()) if n != task['obj']]
        if len(names) > MAX_SOFT:
            names.sort(key=lambda nm: (obstacles[nm][0] - block[0]) ** 2
                       + (obstacles[nm][1] - block[1]) ** 2)
            names = names[:MAX_SOFT]
        soft = [obstacles[n] for n in names]
        eh = list(getattr(self, 'extra_hard', []))
        hard = eh if task['obj'] == 'target_block' else ([(block, BASE_R + 0.003)] + eh)
        c, s = math.cos(rb['th']), math.sin(rb['th'])
        dx, dy = cur_obj[0] - rb['x'], cur_obj[1] - rb['y']
        rel = (c * dx + s * dy, -s * dx + c * dy)
        rel_th = wrap(cur_obj[2] - rb['th'])
        carry = self._carry_rects(rb['arm'], rel, rel_th, cur_obj)
        g = Grid(rb['x'], rb['y'], rb['th'], 0.05, 32, (-1, 0, 1), soft, hard, carry=carry)
        snode = g.nearest_node(rb['x'], rb['y'], rb['th'], max_cells=2, max_dk=1)
        if snode is None:
            self.path = []
            return
        g.solve(snode)
        kind = 'region' if task['obj'] == 'target_block' else 'dump'
        res = self._select_goal(g, rel, rel_th, carry, soft, hard, kind,
                                region, block, cur_obj)
        if res is None:
            self.path = []
            return
        path, m, goal = res
        self.path = list(path) + [goal]
        ded = []
        for p in self.path:
            if not ded or abs(p[0] - ded[-1][0]) > 1e-9 or abs(p[1] - ded[-1][1]) > 1e-9 \
                    or abs(wrap(p[2] - ded[-1][2])) > 1e-9:
                ded.append(p)
        self.path = ded
        self.carry_arm = rb['arm']
        self.pi = 0

    def _do_retract(self, state, robot, cur_obj):
        self.retract_t = getattr(self, 'retract_t', 0) + 1
        if (robot['arm'] > 0.1005 and self.retract_t <= 4 and self.stall == 0
                and cur_obj is not None):
            return self.act(0.0, 0.0, 0.0, -0.05, 1.0)
        self.phase = 'carry'
        self._carry_prev = None
        self._start_carry(state, robot, cur_obj)
        return self._do_carry(state, robot, cur_obj)

    def _do_carry(self, state, robot, cur_obj):
        arm_t = self.carry_arm
        # verify the object really is attached
        pr = getattr(self, '_carry_prev', None)
        if pr is not None and cur_obj is not None:
            rmoved = (abs(robot['x'] - pr[0]) > 1e-7 or abs(robot['y'] - pr[1]) > 1e-7
                      or abs(wrap(robot['th'] - pr[2])) > 1e-7 or abs(robot['arm'] - pr[3]) > 1e-7)
            omoved = any(abs(a - b) > 1e-7 for a, b in zip(cur_obj[:3], pr[4:7]))
            if rmoved and not omoved:
                self.nograsp = getattr(self, 'nograsp', 0) + 1
                if self.nograsp >= 2:
                    c = self.task['cand']
                    self.banned.append((c['pos'][0], c['pos'][1], c['th']))
                    self._carry_prev = None
                    return self._abandon(state)
            elif omoved:
                self.nograsp = 0
        if cur_obj is not None:
            self._carry_prev = (robot['x'], robot['y'], robot['th'], robot['arm'],
                                cur_obj[0], cur_obj[1], cur_obj[2])
        if not self.path:
            if self.task.get('release'):
                self.phase = 'release'
                self.release_t = 0
                return self.act(vac=0.0)
            return self._abandon(state)
        while self.pi < len(self.path) and self._reached(robot, self.path[self.pi], arm_t):
            self.pi += 1
        if self.pi >= len(self.path):
            if self.task.get('release'):
                self.phase = 'release'
                self.release_t = 0
                return self.act(vac=0.0)
            if self.stall >= 3:
                self.path = []
                self._start_carry(state, robot, cur_obj)
                if self.path:
                    return self._do_carry(state, robot, cur_obj)
                self._ban_current()
                self.phase = 'release'
                self.release_t = 0
                return self.act(vac=0.0)
            return self.act(vac=1.0)
        if self.stall >= 6:
            self.stall = 0
            self.fail_cnt += 1
            if self.fail_cnt > 25 or self.budget() < 3 or cur_obj is None:
                self._ban_current()
                self.phase = 'release'
                self.release_t = 0
                return self.act(vac=0.0)
            old_len = len(self.path)
            self._start_carry(state, robot, cur_obj)
            if not self.path or (self.carry_stuck >= 2):
                self._ban_current()
                self.phase = 'release'
                self.release_t = 0
                self.carry_stuck = 0
                return self.act(vac=0.0)
            self.carry_stuck = getattr(self, 'carry_stuck', 0) + 1
            return self._do_carry(state, robot, cur_obj)
        return self._drive(robot, self.path[self.pi], 1.0, arm_t)

    def _do_release(self, state, robot):
        self.release_t += 1
        if self.release_t <= 2:
            return self.act(vac=0.0)
        self.task = None
        self.phase = 'nav'
        if self.t < self.next_plan_t or not self.plan(state):
            self.next_plan_t = self.t + 25
            return self.act()
        rb, block, region, obstacles = self.parse(state)
        obj = self.task['obj']
        cur = block if obj == 'target_block' else obstacles.get(obj)
        return self._do_nav(state, rb, cur)
