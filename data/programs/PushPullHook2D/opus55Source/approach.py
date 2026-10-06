"""Approach for kinder/PushPullHook2D-v0.

Strategy:
  1. Grasp the hook's (long/short) side with the vacuum gripper.
  2. With the hook rigidly attached, plan (A*) in robot (x, y, theta) space to a
     pre-push pose where one face of the hook is behind the movable button
     (with outward normal pointing from button to target).
  3. Push closed-loop along the face normal until the buttons touch.
"""

import heapq
import math
import time

import numpy as np

# ---------------------------------------------------------------- constants
WORLD_X = (0.0, 3.5)
WORLD_Y = (0.0, 2.5)
TABLE_Y = 1.25
MAX_D = 0.05
MAX_DTH = np.pi / 16
MAX_ARM = 0.1
GRID = 0.05
PEN = 0.002
NTH = 32
DTH = 2 * np.pi / NTH


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def compose(a, b):
    """SE2 a * b, poses as (x, y, th)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap(-a[2]))


class Hook:
    def __init__(self, w, l1, l2):
        self.w, self.l1, self.l2 = w, l1, l2
        self.rects = [(-l1, 0.0, -w, 0.0), (-w, 0.0, -l2, 0.0)]  # x0,x1,y0,y1
        # CCW polygon
        self.poly = np.array(
            [(0, 0), (-l1, 0), (-l1, -w), (-w, -w), (-w, -l2), (0, -l2)], dtype=float
        )
        self.hull = np.array(
            [(0, 0), (-l1, 0), (-l1, -w), (-w, -l2), (0, -l2)], dtype=float
        )
        self.faces = []  # (p0, p1, normal)
        n = len(self.poly)
        for i in range(n):
            p0, p1 = self.poly[i], self.poly[(i + 1) % n]
            d = p1 - p0
            ln = np.linalg.norm(d)
            nrm = np.array([d[1], -d[0]]) / ln
            self.faces.append((p0, p1, nrm))

    def dist_local(self, px, py):
        """Distance from local points to the L (0 inside)."""
        best = None
        for x0, x1, y0, y1 in self.rects:
            dx = np.maximum(np.maximum(x0 - px, px - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - py, py - y1), 0.0)
            d = np.hypot(dx, dy)
            best = d if best is None else np.minimum(best, d)
        return best

    def dist_world(self, hp, px, py):
        """hp: hook pose (x, y, th) (scalars or arrays)."""
        hx, hy, hth = hp
        c, s = np.cos(hth), np.sin(hth)
        rx, ry = px - hx, py - hy
        lx = c * rx + s * ry
        ly = -s * rx + c * ry
        return self.dist_local(lx, ly)

    def hull_world(self, hp):
        hx, hy, hth = hp
        c, s = np.cos(hth), np.sin(hth)
        xs = hx[..., None] * 0 + hx if False else None  # noqa
        px = self.hull[:, 0]
        py = self.hull[:, 1]
        hx = np.asarray(hx)[..., None]
        hy = np.asarray(hy)[..., None]
        c = np.asarray(c)[..., None]
        s = np.asarray(s)[..., None]
        wx = hx + c * px - s * py
        wy = hy + s * px + c * py
        return wx, wy


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.lo = np.array(action_space.low, dtype=np.float32)
        self.hi = np.array(action_space.high, dtype=np.float32)

    # ------------------------------------------------------------ parsing
    def _parse(self, s):
        s = np.asarray(s, dtype=float)
        d = {}
        d["robot"] = (s[0], s[1], s[2])
        d["rad"] = s[3]
        d["arm"] = s[4]
        d["arm_max"] = s[5]
        d["vac"] = s[6]
        d["gh"] = s[7]
        d["gw"] = s[8]
        d["hook"] = (s[9], s[10], s[11])
        d["hook_shape"] = (s[17], s[18], s[19])
        d["btn"] = np.array([s[20], s[21]])
        d["br"] = s[28]
        d["tgt"] = np.array([s[29], s[30]])
        d["tr"] = s[37]
        return d

    def reset(self, state, info):
        d = self._parse(state)
        self.hook = Hook(*d["hook_shape"])
        self._init_face_pts(d["br"])
        self.goals = None
        self.mode = "plan_grasp"
        self.plan = []
        self.T = None  # base -> hook transform when grasped
        self.push = None
        self.last_robot = None
        self.stuck = 0
        self.fail_count = 0
        self.grasp_blacklist = set()
        self.carry = None
        self.no_direct = False
        self.t = 0
        self.t0 = time.time()
        self.cooldown = 0

    # ------------------------------------------------------------ actions
    def _act(self, dx=0.0, dy=0.0, dth=0.0, darm=0.0, vac=0.0):
        a = np.array([dx, dy, dth, darm, vac], dtype=np.float32)
        a[:4] = np.clip(a[:4], self.lo[:4], self.hi[:4])
        a[4] = np.clip(a[4], 0.0, 1.0)
        return a

    def _toward(self, d, wp, vac):
        x, y, th = d["robot"]
        return self._act(wp[0] - x, wp[1] - y, wrap(wp[2] - th), 0.0, vac)

    # ------------------------------------------------------------ main
    def get_action(self, state):
        self.t += 1
        d = self._parse(state)
        if time.time() - self.t0 > 50.0:
            # out of time budget: avoid any further planning
            return self._act(vac=1.0 if self.T is not None else 0.0)
        if self.cooldown > 0:
            self.cooldown -= 1
            return self._act(np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05),
                             0.0, 0.0, 1.0 if self.T is not None else 0.0)
        for _ in range(6):
            a = self._step(d)
            if a is not None:
                self.last_robot = d["robot"]
                return a
        return self._act(vac=1.0 if self.T is not None else 0.0)

    def _step(self, d):
        m = self.mode
        if m == "plan_grasp":
            ok = self._plan_grasp(d)
            if not ok:
                # wiggle randomly and try again later
                self.mode = "plan_grasp"
                if len(self.grasp_blacklist) > 0 and self.t % 2 == 0:
                    self.grasp_blacklist = set()
                self.cooldown = 4
                return self._act(np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05))
            self.mode = "retract"
            return None
        if m == "retract":
            if d["arm"] > d["rad"] + 1e-6:
                return self._act(darm=-0.1)
            self.mode = "nav_grasp"
            return None
        if m == "nav_grasp":
            if len(self.plan) == 1 and not (d["arm"] > d["rad"] + 1e-6):
                wp = self.plan[0]
                x, y, th = d["robot"]
                if (abs(wp[0] - x) <= 0.05 and abs(wp[1] - y) <= 0.05
                        and abs(wrap(wp[2] - th)) <= MAX_DTH):
                    self.mode = "verify_grasp"
                    a = self._toward(d, wp, 1.0)
                    a[3] = np.float32(self.hi[3])
                    return a
            return self._follow(d, vac=0.0, next_mode="extend")
        if m == "verify_grasp":
            wp = self.plan[0]
            x, y, th = d["robot"]
            if (abs(d["arm"] - d["arm_max"]) < 1e-4 and abs(wp[0] - x) < 1e-3
                    and abs(wp[1] - y) < 1e-3 and abs(wrap(wp[2] - th)) < 1e-3):
                self.plan = []
                self.mode = "check_grasp"
                return None
            self.grasp_blacklist.add(self.cur_grasp_key)
            self.mode = "plan_grasp"
            return self._act(vac=0.0)
        if m == "extend":
            if d["arm"] < d["arm_max"] - 1e-6:
                prev = d["arm"]
                if self.last_arm is not None and abs(prev - self.last_arm) < 1e-9 and self.ext_tries > 0:
                    # blocked
                    self.mode = "plan_grasp"
                    self.grasp_blacklist.add(self.cur_grasp_key)
                    return None
                self.last_arm = prev
                self.ext_tries += 1
                return self._act(darm=0.1)
            self.mode = "suck"
            return None
        if m == "suck":
            self.mode = "check_grasp"
            return self._act(vac=1.0)
        if m == "check_grasp" and self.carry is not None:
            self.plan = list(self.carry)
            self.mode = "carry"
            return None
        if m == "carry":
            if not self.plan:
                self.no_direct = True
                self.mode = "plan_grasp"
                return self._act(vac=0.0)
            return self._follow(d, vac=1.0, next_mode="carry_done")
        if m == "carry_done":
            self.no_direct = True
            self.mode = "plan_grasp"
            return self._act(vac=0.0)
        if m == "check_grasp":
            # compute transform
            B = d["robot"]
            H = d["hook"]
            self.T = compose(inverse(B), H)
            self.arm = d["arm"]
            self.mode = "plan_push"
            self.grasp_check_hook = H
            return None
        if m == "plan_push":
            ok = self._plan_push(d)
            if not ok:
                self.fail_count += 1
                if self.fail_count > 5:
                    # release and regrasp somewhere else
                    self.T = None
                    self.grasp_blacklist.add(self.cur_grasp_key)
                    self.mode = "plan_grasp"
                    self.fail_count = 0
                    return self._act(vac=0.0)
                self.cooldown = 2
                return self._act(np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0, 0, 1.0)
            self.mode = "nav_push"
            return None
        if m == "nav_push":
            # verify hook attached
            Hexp = compose(d["robot"], self.T)
            if np.hypot(Hexp[0] - d["hook"][0], Hexp[1] - d["hook"][1]) > 0.02:
                self.T = None
                self.mode = "plan_grasp"
                return self._act(vac=0.0)
            return self._follow(d, vac=1.0, next_mode="push")
        if m == "push":
            return self._push_step(d)
        return self._act()

    def _follow(self, d, vac, next_mode):
        x, y, th = d["robot"]
        while self.plan:
            wp = self.plan[0]
            if abs(wp[0] - x) < 1e-3 and abs(wp[1] - y) < 1e-3 and abs(wrap(wp[2] - th)) < 2e-3:
                self.plan.pop(0)
                self.stuck = 0
                continue
            break
        if not self.plan:
            self.mode = next_mode
            self.last_arm = None
            self.ext_tries = 0
            return None
        # stuck detection
        if self.last_robot is not None and np.allclose(self.last_robot, d["robot"], atol=1e-7):
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck > 2:
            self.stuck = 0
            if self.mode == "nav_grasp":
                self.grasp_blacklist.add(self.cur_grasp_key)
                self.mode = "plan_grasp"
            elif self.mode == "carry":
                self.no_direct = True
                self.mode = "carry_done"
            else:
                self.mode = "plan_push"
                self.fail_count += 1
            return None
        return self._toward(d, self.plan[0], vac)

    # ============================================================ grasp
    def _base_ok_grid(self, xs, ys, hp, clear):
        """Base circle (radius clear) clear of hook & walls & table."""
        dist = self.hook.dist_world(hp, xs, ys)
        ok = dist > clear
        ok &= (xs > clear) & (xs < WORLD_X[1] - clear)
        ok &= (ys > clear) & (ys < TABLE_Y - 0.1 - 0.003)
        return ok

    def _grasp_candidates(self, d):
        hp = d["hook"]
        hk = self.hook
        rad, amax, gw, gh = d["rad"], d["arm_max"], d["gw"], d["gh"]
        cands = []
        for fi, (p0, p1, nrm) in enumerate(hk.faces):
            L = np.linalg.norm(p1 - p0)
            margin = 0.06
            if L < 2 * margin:
                continue
            for t in np.arange(margin, L - margin + 1e-9, 0.025):
                pl = p0 + (p1 - p0) * (t / L)
                # avoid concave corner region
                pw = np.array(compose(hp, (pl[0], pl[1], 0.0))[:2])
                nw = rot(hp[2]) @ nrm
                g = pw + nw * (gw / 2 + 0.008)
                th = math.atan2(-nw[1], -nw[0])
                b = g - amax * np.array([math.cos(th), math.sin(th)])
                # base feasibility
                if not (rad + 0.01 < b[0] < WORLD_X[1] - rad - 0.01):
                    continue
                if not (rad + 0.01 < b[1] < TABLE_Y - rad - 0.005):
                    continue
                if hk.dist_world(hp, np.array([b[0]]), np.array([b[1]]))[0] < rad + 0.02:
                    continue
                # gripper rect + arm vs hook: sample points
                ok = True
                perp = np.array([-math.sin(th), math.cos(th)])
                dirv = np.array([math.cos(th), math.sin(th)])
                pts = []
                for along in np.linspace(-gw / 2, gw / 2, 3):
                    for lat in np.linspace(-gh / 2, gh / 2, 7):
                        pts.append(g + dirv * along + perp * lat)
                for k in np.linspace(0, 1, 15):
                    pts.append(b + (g - b) * k)
                pts = np.array(pts)
                dd = hk.dist_world(hp, pts[:, 0], pts[:, 1])
                if np.min(dd) < 0.003:
                    ok = False
                # suction region must overlap hook: centre of suction close
                sc = g + dirv * (gw / 2 + gw)
                sd = hk.dist_world(hp, np.array([sc[0]]), np.array([sc[1]]))[0]
                if sd > 0.004:
                    ok = False
                # gripper inside world
                if not (0.04 < g[0] < 3.46 and 0.04 < g[1] < 2.46):
                    ok = False
                if not ok:
                    continue
                cands.append(((fi, round(float(t), 3)), (float(b[0]), float(b[1]), th)))
        return cands

    def _direct_candidates(self, d):
        """Grasp the movable button directly and carry it to the target."""
        btn, tgt, r = d["btn"], d["tgt"], d["br"]
        rad, amax, gw = d["rad"], d["arm_max"], d["gw"]
        hp = d["hook"]
        hk = self.hook
        reach = amax + gw / 2 + 0.008 + r
        ymax = TABLE_Y - rad - 0.004
        out = []
        for th in np.linspace(0.25, np.pi - 0.25, 25):
            dv = np.array([math.cos(th), math.sin(th)])
            b0 = btn - reach * dv
            if not (rad + 0.01 < b0[0] < WORLD_X[1] - rad - 0.01 and rad + 0.01 < b0[1] < ymax):
                continue
            # final button position: as close to the target as possible
            goal = tgt.copy()
            if goal[1] - reach * dv[1] > ymax:
                goal[1] = ymax + reach * dv[1]
            if np.linalg.norm(goal - tgt) > 0.085:
                continue
            # move a bit past the 0.1-disk boundary only as needed
            v = goal - btn
            Dv = float(np.linalg.norm(v))
            if Dv > 0.1 - 0.02:
                # stop when within 0.08 of target along the straight line
                pass
            b1 = goal - reach * dv
            if not (rad + 0.01 < b1[0] < WORLD_X[1] - rad - 0.01 and rad + 0.01 < b1[1] < ymax):
                continue
            n = int(math.ceil(max(abs(b1[0] - b0[0]), abs(b1[1] - b0[1])) / MAX_D - 1e-9))
            n = max(n, 1)
            carry = []
            ok = True
            for k in range(1, n + 1):
                bb = b0 + (b1 - b0) * k / n
                c = btn + (b1 - b0) * k / n
                g = bb + dv * amax
                pts = np.array([bb, c, g])
                dd = hk.dist_world(hp, pts[:, 0], pts[:, 1])
                if dd[0] < rad + 0.01 or dd[1] < r + 0.01 or dd[2] < 0.05:
                    ok = False
                    break
                carry.append((float(bb[0]), float(bb[1]), float(th)))
            if not ok:
                continue
            # initial grasp pose: gripper/arm must not touch hook
            g0 = b0 + dv * amax
            if hk.dist_world(hp, np.array([g0[0]]), np.array([g0[1]]))[0] < 0.05:
                continue
            if hk.dist_world(hp, np.array([b0[0]]), np.array([b0[1]]))[0] < rad + 0.02:
                continue
            out.append(((float(b0[0]), float(b0[1]), float(th)), carry))
        return out

    def _plan_grasp(self, d):
        hp = d["hook"]
        rad = d["rad"]
        clear = rad + 0.025
        x0, y0, th0 = d["robot"]
        # grid aligned with current pose
        nx_lo = int(math.ceil((clear - x0) / GRID))
        nx_hi = int(math.floor((WORLD_X[1] - clear - x0) / GRID))
        ny_lo = int(math.ceil((clear - y0) / GRID))
        ny_hi = int(math.floor((TABLE_Y - 0.1 - 0.003 - y0) / GRID))
        ix = np.arange(nx_lo, nx_hi + 1)
        iy = np.arange(ny_lo, ny_hi + 1)
        XS = x0 + ix * GRID
        YS = y0 + iy * GRID
        GX, GY = np.meshgrid(XS, YS, indexing="ij")
        valid = self._base_ok_grid(GX, GY, hp, clear)
        sx, sy = -nx_lo, -ny_lo
        if not (0 <= sx < len(XS) and 0 <= sy < len(YS)):
            return False
        valid[sx, sy] = True
        # Dijkstra over 8-connected grid
        INF = 1e9
        dist = np.full(valid.shape, INF)
        parent = {}
        dist[sx, sy] = 0
        pq = [(0.0, sx, sy)]
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        while pq:
            dc, i, j = heapq.heappop(pq)
            if dc > dist[i, j]:
                continue
            for di, dj in nbrs:
                ni, nj = i + di, j + dj
                if 0 <= ni < valid.shape[0] and 0 <= nj < valid.shape[1] and valid[ni, nj]:
                    nd = dc + 1.0
                    if nd < dist[ni, nj]:
                        dist[ni, nj] = nd
                        parent[(ni, nj)] = (i, j)
                        heapq.heappush(pq, (nd, ni, nj))
        cands = self._grasp_candidates(d)
        best = None
        for key, bpose in cands:
            if key in self.grasp_blacklist:
                continue
            bi = int(round((bpose[0] - x0) / GRID)) + sx
            bj = int(round((bpose[1] - y0) / GRID)) + sy
            if not (0 <= bi < valid.shape[0] and 0 <= bj < valid.shape[1]):
                continue
            if dist[bi, bj] >= INF:
                continue
            # final segment from node to bpose must be clear (short)
            nxp, nyp = XS[bi], YS[bj]
            seg = np.linspace(0, 1, 5)
            px = nxp + (bpose[0] - nxp) * seg
            py = nyp + (bpose[1] - nyp) * seg
            if np.min(self.hook.dist_world(hp, px, py)) < clear - 0.01:
                continue
            # push feasibility with this grasp
            T = compose(inverse(bpose), hp)
            pcost = self._best_push_cost(d, T, bpose, d["arm_max"])
            if pcost is None:
                continue
            nr = math.ceil(abs(wrap(bpose[2] - th0)) / MAX_DTH - 1e-9)
            cost = max(dist[bi, bj] + 1, nr) + pcost
            if best is None or cost < best[0]:
                best = (cost, key, bpose, (bi, bj), None)
        if not self.no_direct:
            for bpose, carry in self._direct_candidates(d):
                bi = int(round((bpose[0] - x0) / GRID)) + sx
                bj = int(round((bpose[1] - y0) / GRID)) + sy
                if not (0 <= bi < valid.shape[0] and 0 <= bj < valid.shape[1]):
                    continue
                if dist[bi, bj] >= INF:
                    continue
                nxp, nyp = XS[bi], YS[bj]
                seg = np.linspace(0, 1, 5)
                px = nxp + (bpose[0] - nxp) * seg
                py = nyp + (bpose[1] - nyp) * seg
                if np.min(self.hook.dist_world(hp, px, py)) < clear - 0.01:
                    continue
                nr = math.ceil(abs(wrap(bpose[2] - th0)) / MAX_DTH - 1e-9)
                cost = max(dist[bi, bj] + 1, nr) + len(carry) + 1
                if best is None or cost < best[0]:
                    best = (cost, ("direct",), bpose, (bi, bj), carry)
        if best is None:
            return False
        _, key, bpose, (bi, bj), carry = best
        self.carry = carry
        path = []
        node = (bi, bj)
        while node != (sx, sy):
            path.append(node)
            node = parent[node]
        path.reverse()
        pts = [(float(XS[i]), float(YS[j])) for i, j in path]
        pts.append((bpose[0], bpose[1]))
        tgt = bpose[2]
        nrot = int(math.ceil(abs(wrap(tgt - th0)) / MAX_DTH - 1e-9))
        while len(pts) < nrot:
            pts.insert(len(pts) - 1, pts[-2] if len(pts) >= 2 else (x0, y0))
        plan = []
        cur = th0
        for k, p in enumerate(pts):
            remaining = len(pts) - k
            step = wrap(tgt - cur) / remaining if remaining > 0 else 0.0
            # rotate as early as possible (bounded)
            step = float(np.clip(wrap(tgt - cur), -MAX_DTH, MAX_DTH))
            cur = wrap(cur + step)
            plan.append((p[0], p[1], cur))
        plan[-1] = (bpose[0], bpose[1], tgt)
        self.plan = plan
        self.cur_grasp_key = key
        return True

    # ============================================================ push
    def _push_options(self, d, T, arm, btn, goal):
        """Enumerate push options: list of (face, t, base_contact_pose, u, D, hth)."""
        r = d["br"]
        v = goal - btn
        D = float(np.linalg.norm(v))
        u = v / max(D, 1e-9)
        ang_u = math.atan2(u[1], u[0])
        hk = self.hook
        opts = []
        Tinv = inverse(T)
        for fi, (p0, p1, nrm) in enumerate(hk.faces):
            hth = wrap(ang_u - math.atan2(nrm[1], nrm[0]))
            R = rot(hth)
            for pl in self._face_pts[fi]:
                pw = btn - u * (r + 0.005)
                hpos = pw - R @ pl
                H = (float(hpos[0]), float(hpos[1]), hth)
                B = compose(H, Tinv)
                opts.append((fi, pl, B, u, D, hth))
        return opts

    def _init_face_pts(self, r):
        hk = self.hook
        self._face_pts = []
        for fi, (p0, p1, nrm) in enumerate(hk.faces):
            L = float(np.linalg.norm(p1 - p0))
            margin = 0.02
            if L < 2 * margin:
                ts = [L / 2]
            else:
                ts = np.linspace(margin, L - margin, max(2, int((L - 2 * margin) / 0.05) + 1))
            pts = []
            for t in ts:
                pl = p0 + (p1 - p0) * (t / L)
                c_local = pl + nrm * (r + 0.005)
                okf = True
                for fj, (q0, q1, _) in enumerate(hk.faces):
                    if fj != fi and seg_dist(c_local, q0, q1) < r + 0.012:
                        okf = False
                        break
                if not okf:
                    continue
                if hk.dist_local(np.array([c_local[0]]), np.array([c_local[1]]))[0] < r:
                    continue
                pts.append(pl)
            self._face_pts.append(pts)

    def _pose_valid(self, d, B, T, arm, btn, margin, check_btn=True):
        x, y, th = B
        rad = d["rad"]
        if not (rad + 0.005 < x < WORLD_X[1] - rad - 0.005):
            return False
        if not (rad + 0.005 < y < TABLE_Y - rad - 0.003):
            return False
        g = (x + arm * math.cos(th), y + arm * math.sin(th))
        if not (0.04 < g[0] < 3.46 and 0.04 < g[1] < 2.46):
            return False
        H = compose(B, T)
        wx, wy = self.hook.hull_world((np.array(H[0]), np.array(H[1]), np.array(H[2])))
        if np.min(wx) < 0.004 or np.max(wx) > WORLD_X[1] - 0.004:
            return False
        if np.min(wy) < 0.004 or np.max(wy) > WORLD_Y[1] - 0.004:
            return False
        if check_btn:
            r = d["br"]
            if math.hypot(g[0] - btn[0], g[1] - btn[1]) < r + 0.045:
                return False
            if self.hook.dist_world(H, np.array([btn[0]]), np.array([btn[1]]))[0] < r + margin:
                return False
        return True

    def _option_feasible(self, d, opt, T, arm, btn):
        fi, pl, B, u, D, hth = opt
        r = d["br"]
        # quick base checks first
        rad = d["rad"]
        ymax = TABLE_Y - rad - 0.003
        if B[1] > ymax or B[1] + u[1] * D > ymax:
            return None
        pre = (B[0] - u[0] * 0.06, B[1] - u[1] * 0.06, B[2])
        end = (B[0] + u[0] * D, B[1] + u[1] * D, B[2])
        if not self._pose_valid(d, B, T, arm, btn, 0, check_btn=False):
            return None
        if not self._pose_valid(d, end, T, arm, btn, 0, check_btn=False):
            return None
        if not self._pose_valid(d, pre, T, arm, btn, 0.04):
            return None
        # gripper must avoid button along the push
        for k in np.linspace(0, 1, 6):
            bb = (B[0] + u[0] * D * k, B[1] + u[1] * D * k, B[2])
            g = np.array([bb[0] + arm * math.cos(bb[2]), bb[1] + arm * math.sin(bb[2])])
            c = btn + u * D * k
            if np.hypot(*(g - c)) < r + 0.045:
                return None
        return pre, end

    def _leg_options(self, d, T, arm, btn, goal, B0, limit=None):
        out = []
        for opt in self._push_options(d, T, arm, btn, goal):
            res = self._option_feasible(d, opt, T, arm, btn)
            if res is None:
                continue
            pre, end = res
            D = opt[4]
            c = max(abs(pre[0] - B0[0]) / GRID, abs(pre[1] - B0[1]) / GRID,
                    abs(wrap(pre[2] - B0[2])) / DTH) + 1.0 + D / 0.045
            out.append((c, opt, pre, end))
            if limit is not None and len(out) >= limit:
                break
        out.sort(key=lambda z: z[0])
        return out

    def _final_goals(self, d):
        tgt = d["tgt"]
        goals = [tgt]
        for k in range(8):
            a = k * np.pi / 4
            goals.append(tgt + 0.055 * np.array([math.cos(a), math.sin(a)]))
        return goals

    def _intermediates(self, d, btn):
        tgt = d["tgt"]
        out = []
        for rho in (0.12, 0.22, 0.32):
            for k in range(12):
                a = k * np.pi / 6
                w = btn + rho * np.array([math.cos(a), math.sin(a)])
                if not (0.15 < w[0] < 3.35 and TABLE_Y + 0.06 < w[1] < 2.4):
                    continue
                if np.linalg.norm(w - tgt) < 0.1:
                    continue
                out.append(w)
        out.sort(key=lambda w: np.linalg.norm(w - btn) + np.linalg.norm(tgt - w))
        return out

    def _plan_sequence(self, d, T, arm, B0, quick=False):
        """Return (cost, goals list, first-leg options) or None."""
        btn = d["btn"]
        best = None
        for goal in self._final_goals(d):
            legs = self._leg_options(d, T, arm, btn, goal, B0)
            if legs:
                c = legs[0][0] + np.linalg.norm(goal - d["tgt"]) * 20
                if best is None or c < best[0]:
                    best = (c, [goal], legs)
        if best is not None:
            return best
        for w in self._intermediates(d, btn):
            legs1 = self._leg_options(d, T, arm, btn, w, B0)
            if not legs1:
                continue
            for goal in self._final_goals(d)[:1]:
                end = legs1[0][3]
                legs2 = self._leg_options(d, T, arm, w, goal, end, limit=1)
                if legs2:
                    c = legs1[0][0] + 30 + legs2[0][0]
                    return (c, [w, goal], legs1)
        return None

    # ============================================================ BFS push planning
    def _poses_valid(self, d, X, Y, TH, T, arm, btn=None, margin=0.0):
        rad = d["rad"]
        ok = (X > rad + 0.005) & (X < WORLD_X[1] - rad - 0.005)
        ok &= (Y > rad + 0.005) & (Y < TABLE_Y - rad - 0.003)
        c, s = np.cos(TH), np.sin(TH)
        tx, ty, tth = T
        HX = X + c * tx - s * ty
        HY = Y + s * tx + c * ty
        HT = TH + tth
        wx, wy = self.hook.hull_world((HX, HY, HT))
        ok &= (wx.min(-1) > 0.004) & (wx.max(-1) < WORLD_X[1] - 0.004)
        ok &= (wy.min(-1) > 0.004) & (wy.max(-1) < WORLD_Y[1] - 0.004)
        GXg = X + arm * c
        GYg = Y + arm * s
        ok &= (GXg > 0.04) & (GXg < 3.46) & (GYg > 0.04) & (GYg < 2.46)
        if btn is not None:
            r = d["br"]
            ok &= self.hook.dist_world((HX, HY, HT), btn[0], btn[1]) > r + margin
            ok &= np.hypot(GXg - btn[0], GYg - btn[1]) > r + 0.045
        return ok

    def _bfs_grid(self, d, B0, T, arm, btn, margin=0.012):
        rad = d["rad"]
        x0, y0, th0 = B0
        xlo, xhi = rad + 0.005, WORLD_X[1] - rad - 0.005
        ylo, yhi = rad + 0.005, TABLE_Y - rad - 0.003
        ix = np.arange(int(math.ceil((xlo - x0) / GRID)), int(math.floor((xhi - x0) / GRID)) + 1)
        iy = np.arange(int(math.ceil((ylo - y0) / GRID)), int(math.floor((yhi - y0) / GRID)) + 1)
        if len(ix) == 0 or len(iy) == 0 or ix[0] > 0 or iy[0] > 0 or ix[-1] < 0 or iy[-1] < 0:
            ix = np.arange(min(ix[0] if len(ix) else 0, 0), max(ix[-1] if len(ix) else 0, 0) + 1)
            iy = np.arange(min(iy[0] if len(iy) else 0, 0), max(iy[-1] if len(iy) else 0, 0) + 1)
        XS = x0 + ix * GRID
        YS = y0 + iy * GRID
        THS = th0 + np.arange(NTH) * DTH
        GX, GY, GT = np.meshgrid(XS, YS, THS, indexing="ij")
        valid = self._poses_valid(d, GX, GY, GT, T, arm, btn, margin)
        si, sj = -ix[0], -iy[0]
        valid[si, sj, 0] = True
        dist = np.full(valid.shape, -1, dtype=np.int32)
        dist[si, sj, 0] = 0
        frontier = np.zeros(valid.shape, dtype=bool)
        frontier[si, sj, 0] = True
        visited = frontier.copy()
        lvl = 0
        while frontier.any() and lvl < 400:
            lvl += 1
            f = frontier | np.roll(frontier, 1, 2) | np.roll(frontier, -1, 2)
            g = f.copy()
            g[1:] |= f[:-1]
            g[:-1] |= f[1:]
            h = g.copy()
            h[:, 1:] |= g[:, :-1]
            h[:, :-1] |= g[:, 1:]
            new = h & valid & ~visited
            dist[new] = lvl
            visited |= new
            frontier = new
        return dict(XS=XS, YS=YS, x0=x0, y0=y0, th0=th0, ix0=ix[0], iy0=iy[0],
                    dist=dist, valid=valid, start=(si, sj, 0))

    def _hop_nodes(self, G, P):
        """Grid nodes from which pose P is reachable in one step."""
        fx = (P[0] - G["x0"]) / GRID - G["ix0"]
        fy = (P[1] - G["y0"]) / GRID - G["iy0"]
        fk = wrap(P[2] - G["th0"]) / DTH
        nX, nY, _ = G["dist"].shape
        out = []
        for i in {int(math.floor(fx)), int(math.ceil(fx))}:
            for j in {int(math.floor(fy)), int(math.ceil(fy))}:
                for k in {int(math.floor(fk)), int(math.ceil(fk))}:
                    if 0 <= i < nX and 0 <= j < nY:
                        out.append((i, j, k % NTH))
        return out

    def _contact_options(self, d, T, arm, btn, goals):
        """Vectorized contact poses for all faces/points/goals.
        Returns list of tuples (fi, pl, Bc, u, Dn, goal)."""
        r = d["br"]
        tgt = d["tgt"]
        Tinv = inverse(T)
        out = []
        for goal in goals:
            v = goal - btn
            D = float(np.linalg.norm(v))
            if D < 1e-6:
                continue
            u = v / D
            Dn = max(0.0, D - (0.1 - float(np.linalg.norm(goal - tgt))) + 0.006) if len(goals) and np.linalg.norm(goal - tgt) < 0.1 else D
            ang_u = math.atan2(u[1], u[0])
            for fi, (p0, p1, nrm) in enumerate(self.hook.faces):
                pts = self._face_pts[fi]
                if not pts:
                    continue
                hth = wrap(ang_u - math.atan2(nrm[1], nrm[0]))
                R = rot(hth)
                pw = btn - u * (r - PEN)
                for pl in pts:
                    hpos = pw - R @ pl
                    Bc = compose((float(hpos[0]), float(hpos[1]), hth), Tinv)
                    out.append((fi, pl, Bc, u, Dn, goal, D))
        if not out:
            return []
        X = np.array([o[2][0] for o in out])
        Y = np.array([o[2][1] for o in out])
        TH = np.array([o[2][2] for o in out])
        ok = self._poses_valid(d, X, Y, TH, T, arm)
        U = np.array([o[3] for o in out])
        DD = np.array([o[6] for o in out])
        ok &= self._poses_valid(d, X + U[:, 0] * DD, Y + U[:, 1] * DD, TH, T, arm)
        # gripper must avoid the button along the push
        GXg = X + arm * np.cos(TH)
        GYg = Y + arm * np.sin(TH)
        for k in np.linspace(0, 1, 6):
            cx = btn[0] + U[:, 0] * DD * k
            cy = btn[1] + U[:, 1] * DD * k
            ok &= np.hypot(GXg + U[:, 0] * DD * k - cx, GYg + U[:, 1] * DD * k - cy) > r + 0.045
        return [o for o, f in zip(out, ok) if f]

    def _eval_push(self, d, T, arm, B0, btn, goals):
        """Return best (cost, option, node, G) using BFS distances."""
        G = self._bfs_grid(d, B0, T, arm, btn)
        dist = G["dist"]
        best = None
        for o in self._contact_options(d, T, arm, btn, goals):
            fi, pl, Bc, u, Dn, goal, D = o
            push_steps = max(1, int(math.ceil(Dn / (0.05 - PEN))))
            for n in self._hop_nodes(G, Bc):
                dn = dist[n]
                if dn < 0:
                    continue
                c = dn + push_steps
                if best is None or c < best[0]:
                    best = (c, o, n, G)
        return best

    def _extract(self, G, node):
        dist = G["dist"]
        nX, nY, _ = dist.shape
        path = [node]
        n = node
        while dist[n] > 0:
            i, j, k = n
            found = None
            for dk in (0, -1, 1):
                for di in (0, -1, 1):
                    for dj in (0, -1, 1):
                        ni, nj, nk = i + di, j + dj, (k + dk) % NTH
                        if 0 <= ni < nX and 0 <= nj < nY and dist[ni, nj, nk] == dist[n] - 1:
                            found = (ni, nj, nk)
                            break
                    if found:
                        break
                if found:
                    break
            if found is None:
                return None
            n = found
            path.append(n)
        path.reverse()
        out = []
        for i, j, k in path[1:]:
            out.append((float(G["XS"][i]), float(G["YS"][j]), float(wrap(G["th0"] + k * DTH))))
        return out

    def _best_push_cost(self, d, T, B0, arm):
        res = self._eval_push(d, T, arm, B0, d["btn"], self._final_goals(d))
        if res is not None:
            return res[0]
        res = self._plan_sequence(d, T, arm, B0)
        return None if res is None else res[0] + 10

    def _plan_push(self, d):
        if self.fail_count > 4:
            return False
        T = self.T
        arm = self.arm
        B0 = d["robot"]
        goals = self.goals if (self.goals is not None and len(self.goals) > 1) else None
        if goals is None:
            res = self._eval_push(d, T, arm, B0, d["btn"], self._final_goals(d))
            if res is not None:
                c, o, node, G = res
                path = self._extract(G, node)
                if path is not None:
                    self.plan = path
                    self.push = (o[0], o[1], o[2], o[3], o[6], None)
                    self.goals = [o[5]]
                    return True
            seq = self._plan_sequence(d, T, arm, B0)
            if seq is None:
                return False
            goals = seq[1]
        # multi-leg: plan first leg towards goals[0]
        res = self._eval_push(d, T, arm, B0, d["btn"], [goals[0]])
        if res is None:
            self.goals = None
            return False
        c, o, node, G = res
        path = self._extract(G, node)
        if path is None:
            return False
        self.plan = path
        self.push = (o[0], o[1], o[2], o[3], o[6], None)
        self.goals = list(goals)
        return True

    def _astar(self, d, goal, T, arm, margin=0.015):
        """A* over (x, y, theta) with 26-connected unit-cost moves.

        The env only checks collisions at the resulting pose of each step, so an
        edge is valid iff its end node is valid.
        """
        rad = d["rad"]
        btn = d["btn"]
        r = d["br"]
        gx, gy, gth = goal
        x0, y0, th0 = d["robot"]
        xlo, xhi = rad + 0.005, WORLD_X[1] - rad - 0.005
        ylo, yhi = rad + 0.005, TABLE_Y - rad - 0.003
        ix = np.arange(int(math.ceil((xlo - x0) / GRID)), int(math.floor((xhi - x0) / GRID)) + 1)
        iy = np.arange(int(math.ceil((ylo - y0) / GRID)), int(math.floor((yhi - y0) / GRID)) + 1)
        if len(ix) == 0 or len(iy) == 0 or ix[0] > 0 or iy[0] > 0 or ix[-1] < 0 or iy[-1] < 0:
            return None
        XS = x0 + ix * GRID
        YS = y0 + iy * GRID
        THS = gth + np.arange(NTH) * DTH
        GX, GY, GT = np.meshgrid(XS, YS, THS, indexing="ij")
        tx, ty, tth = T
        c, s = np.cos(GT), np.sin(GT)
        HX = GX + c * tx - s * ty
        HY = GY + s * tx + c * ty
        HT = GT + tth
        hk = self.hook
        wx, wy = hk.hull_world((HX, HY, HT))
        ok = (wx.min(-1) > 0.004) & (wx.max(-1) < WORLD_X[1] - 0.004)
        ok &= (wy.min(-1) > 0.004) & (wy.max(-1) < WORLD_Y[1] - 0.004)
        ok &= hk.dist_world((HX, HY, HT), btn[0], btn[1]) > r + margin
        GXg = GX + arm * c
        GYg = GY + arm * s
        ok &= (GXg > 0.04) & (GXg < 3.46) & (GYg > 0.04) & (GYg < 2.46)
        ok &= np.hypot(GXg - btn[0], GYg - btn[1]) > r + 0.045
        node_ok = ok
        nX, nY, _ = node_ok.shape
        si, sj = -ix[0], -iy[0]
        gi = int(round((gx - x0) / GRID)) - ix[0]
        gj = int(round((gy - y0) / GRID)) - iy[0]
        if not (0 <= gi < nX and 0 <= gj < nY) or not node_ok[gi, gj, 0]:
            return None
        # final hop from goal node to the exact goal must be one step
        goaln = (gi, gj, 0)
        kf = wrap(th0 - gth) / DTH
        start_ks = [k for k in range(int(math.floor(kf)) - 1, int(math.ceil(kf)) + 2)
                    if abs(k - kf) <= 1.0 + 1e-9]
        start_ks = sorted(set(k % NTH for k in start_ks))

        def h(i, j, k):
            dk = k % NTH
            dk = min(dk, NTH - dk)
            return max(abs(i - gi), abs(j - gj), dk)

        START = ("S",)
        g = {}
        parent = {}
        pq = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = si + di, sj + dj
                if not (0 <= ni < nX and 0 <= nj < nY):
                    continue
                for k in start_ks:
                    if node_ok[ni, nj, k]:
                        n = (ni, nj, k)
                        g[n] = 1
                        parent[n] = START
                        heapq.heappush(pq, (1 + h(*n), 1, n))
        found = False
        expansions = 0
        while pq:
            f, gc, n = heapq.heappop(pq)
            if gc > g.get(n, 1e18):
                continue
            if n == goaln:
                found = True
                break
            expansions += 1
            if expansions > 150000:
                break
            i, j, k = n
            for dk in (-1, 0, 1):
                nk = (k + dk) % NTH
                for di in (-1, 0, 1):
                    ni = i + di
                    if ni < 0 or ni >= nX:
                        continue
                    for dj in (-1, 0, 1):
                        nj = j + dj
                        if nj < 0 or nj >= nY or (di == 0 and dj == 0 and dk == 0):
                            continue
                        if not node_ok[ni, nj, nk]:
                            continue
                        sn = (ni, nj, nk)
                        ng = gc + 1
                        if ng < g.get(sn, 1e18):
                            g[sn] = ng
                            parent[sn] = n
                            heapq.heappush(pq, (ng + h(ni, nj, nk), ng, sn))
        if not found:
            return None
        path = []
        n = goaln
        while n is not START:
            path.append(n)
            n = parent[n]
        path.reverse()
        out = []
        for i, j, k in path:
            out.append((float(XS[i]), float(YS[j]), float(wrap(gth + k * DTH))))
        if abs(out[-1][0] - gx) > 1e-6 or abs(out[-1][1] - gy) > 1e-6:
            out.append((gx, gy, gth))
        return out

    def _push_step(self, d):
        fi, pl, B, u0, D, hth0 = self.push
        T = self.T
        btn, r = d["btn"], d["br"]
        # verify hook attached
        Hexp = compose(d["robot"], T)
        if np.hypot(Hexp[0] - d["hook"][0], Hexp[1] - d["hook"][1]) > 0.02:
            self.T = None
            self.mode = "plan_grasp"
            return self._act(vac=0.0)
        goal = self.goals[0]
        v = goal - btn
        dist = float(np.linalg.norm(v))
        if len(self.goals) > 1 and (dist < 0.025 or np.dot(v, u0) < 0.005):
            # reached intermediate subgoal -> replan from here
            self.goals = self.goals[1:]
            self.mode = "plan_push"
            self.fail_count = 0
            return None
        if dist < 1e-6:
            v = u0
            dist = 1.0
        u = v / dist
        if np.dot(u, u0) < math.cos(0.35):
            self.mode = "plan_push"
            return None
        hk = self.hook
        p0, p1, nrm = hk.faces[fi]
        hth = wrap(math.atan2(u[1], u[0]) - math.atan2(nrm[1], nrm[0]))
        R = rot(hth)
        pw = btn - u * (r - PEN)
        hpos = pw - R @ pl
        Bt = compose((float(hpos[0]), float(hpos[1]), hth), inverse(T))
        x, y, th = d["robot"]
        # stuck detection
        if self.last_robot is not None and np.allclose(self.last_robot, d["robot"], atol=1e-7):
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck > 2 or not self._pose_valid(d, Bt, T, self.arm, btn, 0, check_btn=False):
            self.stuck = 0
            self.mode = "plan_push"
            self.fail_count += 1
            return None
        dth = wrap(Bt[2] - th)
        dth = float(np.clip(dth, -0.1, 0.1))
        dx = float(np.clip(Bt[0] - x, -0.05, 0.05))
        dy = float(np.clip(Bt[1] - y, -0.05, 0.05))
        for (ax, ay, at) in ((dx, dy, dth), (dx, dy, 0.0), (dx, dy, dth * 0.5),
                             (0.0, 0.0, dth), (dx * 0.5, dy * 0.5, dth)):
            P = (x + ax, y + ay, wrap(th + at))
            if self._pose_valid(d, P, T, self.arm, btn, 0, check_btn=False):
                return self._act(ax, ay, at, 0.0, 1.0)
        self.stuck = 0
        self.mode = "plan_push"
        self.fail_count += 1
        return None
