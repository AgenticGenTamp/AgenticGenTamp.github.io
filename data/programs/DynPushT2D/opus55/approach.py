import heapq
import numpy as np
from tmodel import TGeom, sim_step, wrap, corners_world, R_ROBOT

AMAX = 0.0499
LO, HI = 0.1 + 1e-3, 4.9 - 1e-3
POS_TOL, ANG_TOL = 0.03, np.deg2rad(8.0)
K_ANG = 0.8
LOOKAHEAD = 16
LOCAL_FAMILY = True


def cost(bx, by, bt, goal):
    return np.hypot(bx - goal[0], by - goal[1]) + K_ANG * np.abs(wrap(bt - goal[2]))


def solved(bx, by, bt, goal, pf=0.7, af=0.7):
    return (np.abs(bx - goal[0]) < POS_TOL * pf) & (np.abs(by - goal[1]) < POS_TOL * pf) & \
        (np.abs(wrap(bt - goal[2])) < ANG_TOL * af)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.queue = []
        self.g = None
        self.lookahead = LOOKAHEAD
        self.local_family = LOCAL_FAMILY

    def reset(self, state, info):
        self.queue = []
        self.g = TGeom(state[12], state[13], state[14])
        self.goal = np.array(state[29:32], dtype=float)
        self.expect = None
        self.mode = None
        import time as _t
        self.t_start = _t.time()
        self.pts = self.g.boundary_points(0.05)
        lpx = self.pts[:, 0] + self.pts[:, 2] * (R_ROBOT + 0.045)
        lpy = self.pts[:, 1] + self.pts[:, 3] * (R_ROBOT + 0.045)
        self.pre_ok = self.g.closest(lpx, lpy)[4] > R_ROBOT + 0.03
        self.angs = np.deg2rad(np.array([-40, -25, -12, 0, 12, 25, 40]))

    # ---------------- navigation -----------------
    def _seg_free(self, pose, ax, ay, bx_, by_, m):
        """Is segment (a->b) (world, arrays) free of T inflated by m? vectorized over segments."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        lax = c * (ax - pose[0]) + s * (ay - pose[1])
        lay = -s * (ax - pose[0]) + c * (ay - pose[1])
        lbx = c * (bx_ - pose[0]) + s * (by_ - pose[1])
        lby = -s * (bx_ - pose[0]) + c * (by_ - pose[1])
        dx, dy = lbx - lax, lby - lay
        free = np.ones(np.broadcast(lax, lbx).shape, bool)
        for (x0, x1, y0, y1) in self.g.rects:
            x0, x1, y0, y1 = x0 - m, x1 + m, y0 - m, y1 + m
            t0 = np.zeros_like(free, dtype=float)
            t1 = np.ones_like(free, dtype=float)
            hit = np.ones_like(free)
            for (p, d, lo, hi) in ((lax, dx, x0, x1), (lay, dy, y0, y1)):
                p = np.broadcast_to(p, free.shape)
                d = np.broadcast_to(d, free.shape)
                small = np.abs(d) < 1e-12
                inside = (p >= lo) & (p <= hi)
                hit &= ~(small & ~inside)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ta = (lo - p) / d
                    tb = (hi - p) / d
                tmin = np.where(small, -np.inf, np.minimum(ta, tb))
                tmax = np.where(small, np.inf, np.maximum(ta, tb))
                t0 = np.maximum(t0, tmin)
                t1 = np.minimum(t1, tmax)
            hit &= t0 <= t1
            free &= ~hit
        return free

    def _nav_nodes(self, pose, m):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        pts = []
        for (x0, x1, y0, y1) in self.g.rects:
            for lx, ly in ((x0 - m, y0 - m), (x0 - m, y1 + m), (x1 + m, y0 - m), (x1 + m, y1 + m)):
                pts.append((pose[0] + c * lx - s * ly, pose[1] + s * lx + c * ly))
        pts = np.array(pts)
        ok = (pts[:, 0] > LO) & (pts[:, 0] < HI) & (pts[:, 1] > LO) & (pts[:, 1] < HI)
        pts = pts[ok]
        # remove nodes inside inflated T
        lx = c * (pts[:, 0] - pose[0]) + s * (pts[:, 1] - pose[1])
        ly = -s * (pts[:, 0] - pose[0]) + c * (pts[:, 1] - pose[1])
        _, _, _, _, d = self.g.closest(lx, ly)
        return pts[d > R_ROBOT + m * 0.6]

    def _plan_paths(self, pose, start, targets):
        """Shortest paths from start to each target avoiding T. Returns dist array, and
        function to get path waypoints."""
        mcol = 0.012
        nodes = self._nav_nodes(pose, 0.035 + R_ROBOT)
        allp = np.vstack([start[None], nodes])
        n = len(allp)
        # adjacency
        A = allp[:, None, :].repeat(n, 1)
        B = allp[None, :, :].repeat(n, 0)
        free = self._seg_free(pose, A[..., 0], A[..., 1], B[..., 0], B[..., 1], R_ROBOT + mcol)
        # start may be close to block: allow edges from start if it moves away? keep as is but
        # if start is itself inside inflation, relax with smaller margin
        free_s = self._seg_free(pose, A[0, :, 0], A[0, :, 1], B[0, :, 0], B[0, :, 1], R_ROBOT - 0.005)
        free[0, :] = free_s
        free[:, 0] = free_s
        D = np.hypot(A[..., 0] - B[..., 0], A[..., 1] - B[..., 1])
        dist = np.full(n, np.inf)
        prev = np.full(n, -1)
        dist[0] = 0
        pq = [(0.0, 0)]
        done = np.zeros(n, bool)
        while pq:
            d0, i = heapq.heappop(pq)
            if done[i]:
                continue
            done[i] = True
            for j in range(n):
                if free[i, j] and not done[j]:
                    nd = d0 + D[i, j]
                    if nd < dist[j]:
                        dist[j] = nd
                        prev[j] = i
                        heapq.heappush(pq, (nd, j))
        # targets
        T = np.asarray(targets)
        fr = self._seg_free(pose, allp[:, None, 0], allp[:, None, 1], T[None, :, 0], T[None, :, 1], R_ROBOT + mcol)
        fr[0, :] = self._seg_free(pose, allp[0, 0], allp[0, 1], T[:, 0], T[:, 1], R_ROBOT - 0.005)
        DT = np.hypot(allp[:, None, 0] - T[None, :, 0], allp[:, None, 1] - T[None, :, 1])
        tot = np.where(fr, dist[:, None] + DT, np.inf)
        best = np.argmin(tot, axis=0)
        tdist = tot[best, np.arange(len(T))]

        def path(k):
            i = best[k]
            wp = [T[k]]
            while i > 0:
                wp.append(allp[i])
                i = prev[i]
            return wp[::-1]
        return tdist, path

    # ---------------- planning -----------------
    def _eval(self, pose, robot, fine):
        """Evaluate all single push segments from (pose, robot). Returns dict or None."""
        g = self.g
        goal = self.goal
        J0 = float(cost(pose[0], pose[1], pose[2], goal))
        step = AMAX if not fine else AMAX / 3
        nsteps = 40 if not fine else 30
        pts = self.pts
        c, s = np.cos(pose[2]), np.sin(pose[2])
        lcx = pts[:, 0] + pts[:, 2] * (R_ROBOT + 0.003)
        lcy = pts[:, 1] + pts[:, 3] * (R_ROBOT + 0.003)
        wcx = pose[0] + c * lcx - s * lcy
        wcy = pose[1] + s * lcx + c * lcy
        lpx = pts[:, 0] + pts[:, 2] * (R_ROBOT + 0.045)
        lpy = pts[:, 1] + pts[:, 3] * (R_ROBOT + 0.045)
        wpx = pose[0] + c * lpx - s * lpy
        wpy = pose[1] + s * lpx + c * lpy
        inb = (wcx > LO) & (wcx < HI) & (wcy > LO) & (wcy < HI) & (wpx > LO) & (wpx < HI) & (wpy > LO) & (wpy < HI)
        inb &= self.pre_ok
        idx = np.nonzero(inb)[0]
        if len(idx) == 0:
            return None
        pre = np.stack([wpx[idx], wpy[idx]], 1)
        rlx = c * (robot[0] - pose[0]) + s * (robot[1] - pose[1])
        rly = -s * (robot[0] - pose[0]) + c * (robot[1] - pose[1])
        _, _, rnx, rny, rd = g.closest(np.array([rlx]), np.array([rly]))
        retreat = None
        if rd[0] < R_ROBOT + 0.03:
            back = R_ROBOT + 0.035 - rd[0]
            cand = robot + back * np.array([c * rnx[0] - s * rny[0], s * rnx[0] + c * rny[0]])
            if LO < cand[0] < HI and LO < cand[1] < HI:
                retreat = cand
        start = robot if retreat is None else retreat
        tdist, pathf = self._plan_paths(pose, start, pre)
        tdist = tdist + 0.045
        if retreat is not None:
            tdist = tdist + np.hypot(*(retreat - robot))
        nwx = -(c * pts[idx, 2] - s * pts[idx, 3])
        nwy = -(s * pts[idx, 2] + c * pts[idx, 3])
        angs = self.angs
        nf = 2 if self.local_family else 1
        na = len(angs)
        K = len(idx) * na * nf
        ci = np.tile(np.repeat(np.arange(len(idx)), na), nf)
        aa = np.tile(angs, len(idx) * nf)
        fam = np.repeat(np.arange(nf), len(idx) * na)
        dirx = np.cos(aa) * nwx[ci] - np.sin(aa) * nwy[ci]
        diry = np.sin(aa) * nwx[ci] + np.cos(aa) * nwy[ci]
        ux, uy = dirx * step, diry * step
        # local-frame direction (for family 1)
        lux0 = c * ux + s * uy
        luy0 = -s * ux + c * uy
        isloc = fam == 1
        bx = np.full(K, pose[0]); by = np.full(K, pose[1]); bt = np.full(K, pose[2])
        rx = wcx[idx][ci].copy(); ry = wcy[idx][ci].copy()
        valid = np.isfinite(tdist[ci])
        travel = np.ceil(tdist[ci] / AMAX) + 1.0
        cx0, cy0 = corners_world(g, pose[0], pose[1], pose[2])
        m0 = min(cx0.min(), cy0.min(), 5 - cx0.max(), 5 - cy0.max())
        wall_lim = min(0.06, m0 - 0.01)
        best_score = np.full(K, -np.inf)
        best_n = np.zeros(K, int)
        best_J = np.full(K, np.inf)
        fin = np.zeros((K, 5))
        alive = valid.copy()
        for k in range(1, nsteps + 1):
            if nf > 1:
                cb, sb = np.cos(bt), np.sin(bt)
                uxk = np.where(isloc, cb * lux0 - sb * luy0, ux)
                uyk = np.where(isloc, sb * lux0 + cb * luy0, uy)
            else:
                uxk, uyk = ux, uy
            bx, by, bt, rx, ry = sim_step(g, bx, by, bt, rx, ry, uxk, uyk)
            alive &= (rx > LO) & (rx < HI) & (ry > LO) & (ry < HI)
            cxw, cyw = corners_world(g, bx, by, bt)
            mm = np.minimum(np.minimum(cxw.min(0), cyw.min(0)), np.minimum(5 - cxw.max(0), 5 - cyw.max(0)))
            alive &= mm > wall_lim
            if not alive.any():
                break
            J = cost(bx, by, bt, goal)
            J = np.where(solved(bx, by, bt, goal), J - 0.5, J)
            steps_k = k if not fine else np.ceil(k / 3.0)
            sc = (J0 - J) / (travel + steps_k + 1.0)
            upd = alive & (sc > best_score)
            best_score = np.where(upd, sc, best_score)
            best_n = np.where(upd, k, best_n)
            best_J = np.where(upd, J, best_J)
            if upd.any():
                fin[upd] = np.stack([bx, by, bt, rx, ry], 1)[upd]
        steps_n = best_n if not fine else np.ceil(best_n / 3.0)
        return dict(J0=J0, score=best_score, n=best_n, J=best_J, fin=fin, ci=ci, idx=idx,
                    ux=ux, uy=uy, lux=lux0, luy=luy0, fam=fam, travel=travel, tot=travel + steps_n + 1.0, pathf=pathf,
                    retreat=retreat, wc=np.stack([wcx, wcy], 1), fine=fine)

    def _plan(self, obs):
        pose = np.array(obs[0:3], float)
        robot = np.array(obs[16:18], float)
        J0 = float(cost(pose[0], pose[1], pose[2], self.goal))
        fine = J0 < 0.12
        E = self._eval(pose, robot, fine)
        if E is None:
            return None
        sc = E['score']
        order = np.argsort(-sc)
        j = int(order[0])
        if not np.isfinite(sc[j]) or sc[j] <= 0:
            return None
        import time as _t
        if self.lookahead > 0 and not fine and _t.time() - self.t_start < 25.0:
            best_val, best_j = -np.inf, j
            seen = set()
            cnt = 0
            for jj in order:
                if cnt >= self.lookahead or not np.isfinite(sc[jj]) or sc[jj] <= 0:
                    break
                key = (E['ci'][jj] // 2, int(E['n'][jj] // 8))
                if key in seen:
                    continue
                seen.add(key)
                cnt += 1
                f = E['fin'][jj]
                t1 = E['tot'][jj]
                J1 = E['J'][jj]
                val = (E['J0'] - J1) / t1
                if J1 > -0.2:  # not solved
                    fine2 = J1 + 0.0 < 0.12
                    E2 = self._eval(f[0:3], f[3:5], fine2)
                    if E2 is not None:
                        with np.errstate(invalid='ignore'):
                            s2 = (E['J0'] - E2['J']) / (t1 + E2['tot'])
                        ok2 = (E2['score'] > -np.inf) & np.isfinite(E2['tot'])
                        if ok2.any():
                            val = max(val, float(np.max(s2[ok2])))
                if val > best_val:
                    best_val, best_j = val, int(jj)
            j = best_j
        return self._build(E, j, robot)

    def _build(self, E, j, robot):
        ci = E['ci']; idx = E['idx']
        n = int(E['n'][j])
        wp = E['pathf'](ci[j])
        if E['retreat'] is not None:
            wp = [E['retreat']] + wp
        acts = []
        cur = robot.copy()
        for w in wp + [E['wc'][idx][ci[j]]]:
            w = np.asarray(w, float)
            d = w - cur
            L = np.hypot(*d)
            if L < 1e-9:
                continue
            nst = int(np.ceil(L / AMAX - 1e-9))
            for _ in range(nst):
                acts.append(('nav', d / nst))
            cur = w.copy()
        a = np.array([E['ux'][j], E['uy'][j]])
        if E['fam'][j] == 1:
            la = np.array([E['lux'][j], E['luy'][j]])
            if E['fine']:
                la = la * 3
                n = int(np.ceil(n / 3.0))
            for _ in range(n):
                acts.append(('pushL', la.copy()))
        elif E['fine']:
            tot = a * n
            L = np.hypot(*tot)
            nst = max(1, int(np.ceil(L / AMAX - 1e-9)))
            for _ in range(nst):
                acts.append(('push', tot / nst))
        else:
            for _ in range(n):
                acts.append(('push', a.copy()))
        self.dbg = (E['score'][j], E['J'][j], n, E['travel'][j], E['J0'])
        return acts

    def get_action(self, state):
        try:
            a = self._get_action(state)
            a = np.clip(np.asarray(a, dtype=np.float64).reshape(2), -AMAX, AMAX)
            if not np.all(np.isfinite(a)):
                a = np.zeros(2)
            return a
        except Exception:
            self.queue = []
            return np.zeros(2)

    def _get_action(self, state):
        obs = np.asarray(state, float)
        pose = obs[0:3]
        replan = not self.queue
        if self.expect is not None and self.queue:
            ep = self.expect
            if self.queue[0][0] == 'nav':
                if np.abs(pose - ep).max() > 1e-4:
                    replan = True
        if replan:
            self.queue = self._plan(obs) or []
        if not self.queue:
            self.expect = None
            # fallback: move toward block
            d = pose[0:2] - obs[16:18]
            return np.clip(d, -AMAX, AMAX)
        kind, a = self.queue.pop(0)
        if kind == 'pushL':
            cb, sb = np.cos(pose[2]), np.sin(pose[2])
            a = np.array([cb * a[0] - sb * a[1], sb * a[0] + cb * a[1]])
        self.last_kind = kind
        self.expect = pose.copy()
        return np.clip(a, -AMAX, AMAX)
