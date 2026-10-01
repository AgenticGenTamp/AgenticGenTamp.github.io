"""Approach for ClutteredStorage2DEnv: pick up blocks and stow them in the shelf.

Geometry facts established by probing the black box:

* rectangles are given by their ORIGIN CORNER (x,y); the body covers
  corner + u*width*(cos t, sin t) + v*height*(-sin t, cos t), u,v in [0,1].
* the robot collision body is the base circle (r = base_radius) plus the
  vacuum pad, a rectangle CENTRED at base + arm_joint*(cos th, sin th) with
  half extent gripper_width/2 along the arm and gripper_height/2 across it.
  The arm shaft itself does not collide.
* a held block is rigidly attached in the gripper frame and is part of the
  collision body.
* vacuum attaches any block whose rectangle reaches within 0.03 in FRONT of
  the pad (measured along the arm axis) and within the pad's lateral span.
* a step whose resulting configuration is in collision is refused entirely.
* termination: every block rectangle fully inside the shelf opening
  (x1..x1+width1) x (y1..y1+height1).
"""
import heapq
import math

import numpy as np

MAX_DXY = 0.05
MAX_DTH = 0.19634954631328583
MAX_DARM = 0.1

ROOM_X0, ROOM_X1 = 0.0, 5.0
ROOM_Y0 = 0.0
WALL_Y0 = 2.625
WALL_Y1 = 3.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clip(v, m):
    return max(-m, min(m, v))


def rect_extent(x, y, th, w, h):
    c, s = math.cos(th), math.sin(th)
    xs = (x, x + c * w, x + c * w - s * h, x - s * h)
    ys = (y, y + s * w, y + s * w + c * h, y + c * h)
    return min(xs), max(xs), min(ys), max(ys)


def rect_center(x, y, th, w, h):
    c, s = math.cos(th), math.sin(th)
    return (x + 0.5 * (c * w - s * h), y + 0.5 * (s * w + c * h))


def corner_from_center(cx, cy, th, w, h):
    c, s = math.cos(th), math.sin(th)
    return (cx - 0.5 * (c * w - s * h), cy - 0.5 * (s * w + c * h))


def dist_pts_to_rect(px, py, rect):
    x, y, th, w, h = rect
    c, s = math.cos(th), math.sin(th)
    dx = px - x
    dy = py - y
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    qx = np.clip(lx, 0.0, w)
    qy = np.clip(ly, 0.0, h)
    return np.hypot(lx - qx, ly - qy)


def dist_pt_rect(px, py, rect):
    x, y, th, w, h = rect
    c, s = math.cos(th), math.sin(th)
    dx = px - x
    dy = py - y
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    qx = min(max(lx, 0.0), w)
    qy = min(max(ly, 0.0), h)
    return math.hypot(lx - qx, ly - qy)


def rect_pts(rect):
    x, y, th, w, h = rect
    c, s = math.cos(th), math.sin(th)
    ux, uy = c * w, s * w
    vx, vy = -s * h, c * h
    return ((x, y), (x + ux, y + uy), (x + ux + vx, y + uy + vy), (x + vx, y + vy))


def rect_overlap(r1, r2, eps=1e-9):
    p1 = rect_pts(r1)
    p2 = rect_pts(r2)
    for th in (r1[2], r2[2]):
        c, s = math.cos(th), math.sin(th)
        for ax, ay in ((c, s), (-s, c)):
            a0 = min(px * ax + py * ay for px, py in p1)
            a1 = max(px * ax + py * ay for px, py in p1)
            b0 = min(px * ax + py * ay for px, py in p2)
            b1 = max(px * ax + py * ay for px, py in p2)
            if a1 <= b0 + eps or b1 <= a0 + eps:
                return False
    return True


class GeneratedApproach:
    # ------------------------------------------------------------------
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------------------------
    def _parse(self, state):
        robot = shelf = None
        blocks = []
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            tn = o.type.name
            if tn == "crv_robot":
                robot = o
            elif tn == "shelf":
                shelf = o
            else:
                blocks.append((name, o))
        blocks.sort(key=lambda kv: (len(kv[0]), kv[0]))
        g = state.get
        d = {
            "rx": float(g(robot, "x")),
            "ry": float(g(robot, "y")),
            "rth": float(g(robot, "theta")),
            "arm": float(g(robot, "arm_joint")),
            "vac": float(g(robot, "vacuum")),
            "brad": float(g(robot, "base_radius")),
            "gw": float(g(robot, "gripper_width")),
            "gh": float(g(robot, "gripper_height")),
            "armmax": float(g(robot, "arm_length")),
            "sx": float(g(shelf, "x1")),
            "sy": float(g(shelf, "y1")),
            "sw": float(g(shelf, "width1")),
            "sh": float(g(shelf, "height1")),
            "blocks": [
                (float(g(o, "x")), float(g(o, "y")), float(g(o, "theta")),
                 float(g(o, "width")), float(g(o, "height")))
                for _, o in blocks
            ],
        }
        return d

    # ------------------------------------------------------------------
    def _walls(self, d):
        sx, sw = d["sx"], d["sw"]
        big = 3.0
        return [
            (-big, WALL_Y0, 0.0, big + sx, WALL_Y1 - WALL_Y0),
            (sx + sw, WALL_Y0, 0.0, ROOM_X1 - sx - sw + big, WALL_Y1 - WALL_Y0),
            (-big, WALL_Y1, 0.0, ROOM_X1 + 2 * big, big),
            (-big, -big, 0.0, ROOM_X1 + 2 * big, big),
            (-big, -big, 0.0, big, WALL_Y1 + 2 * big),
            (ROOM_X1, -big, 0.0, big, WALL_Y1 + 2 * big),
        ]

    # ------------------------------------------------------------------
    def _inside(self, d, b, margin=0.0):
        x0, x1, y0, y1 = rect_extent(*b)
        return (x0 >= d["sx"] - margin and x1 <= d["sx"] + d["sw"] + margin
                and y0 >= d["sy"] - margin and y1 <= d["sy"] + d["sh"] + margin)

    # ------------------------------------------------------------------
    # collision
    # ------------------------------------------------------------------
    def _pad_rect(self, d, x, y, th, arm):
        gw, gh = d["gw"], d["gh"]
        cx = x + arm * math.cos(th)
        cy = y + arm * math.sin(th)
        return corner_from_center(cx, cy, th, gw, gh) + (th, gw, gh)

    def _held_rect(self, d, x, y, th, arm):
        if self.grasp is None:
            return None
        A, B, C, w, h = self.grasp
        c, s = math.cos(th), math.sin(th)
        lx = arm + A
        cx = x + lx * c - B * s
        cy = y + lx * s + B * c
        bth = th + C
        return corner_from_center(cx, cy, bth, w, h) + (bth, w, h)

    def _collides(self, d, x, y, th, arm, obst, holding):
        brad = d["brad"]
        for r in obst:
            if dist_pt_rect(x, y, r) < brad - 1e-9:
                return True
        pad = self._pad_rect(d, x, y, th, arm)
        for r in obst:
            if rect_overlap(pad, r):
                return True
        lw = 0.014
        shaft = corner_from_center(x + 0.5 * arm * math.cos(th),
                                   y + 0.5 * arm * math.sin(th),
                                   th, arm, lw) + (th, arm, lw)
        for r in obst:
            if rect_overlap(shaft, r):
                return True
        if holding:
            hb = self._held_rect(d, x, y, th, arm)
            if hb is not None:
                for r in obst:
                    if rect_overlap(hb, r):
                        return True
        return False

    def _obstacles(self, d, skip, walls=True):
        obs = self._walls(d) if walls else []
        for i, b in enumerate(d["blocks"]):
            if i == skip:
                continue
            obs.append(b)
        return obs

    # ------------------------------------------------------------------
    # slot layout
    # ------------------------------------------------------------------
    def _make_slots(self, d):
        bw = d["blocks"][0][3] if d["blocks"] else 0.28
        bh = d["blocks"][0][4] if d["blocks"] else 0.04
        k = max(1, int(d["sw"] / (bw + 0.012)))
        colw = d["sw"] / k
        self.k = k
        self.cols = [d["sx"] + colw * (i + 0.5) for i in range(k)]
        pitch = bh + 0.016
        top = d["sy"] + d["sh"] - bh / 2 - 0.006
        nrows = max(1, int((d["sh"] - bh - 0.012) / pitch) + 1)
        self.rows = [top - j * pitch for j in range(nrows)]
        slots = []
        for j, ry in enumerate(self.rows):
            for i, cx in enumerate(self.cols):
                slots.append((j, i, cx, ry))
        return slots

    # ------------------------------------------------------------------
    def reset(self, state, info):
        self.step_i = 0
        self.phase = "nav_grasp"
        self.op = None
        self.op_i = 0
        self.plan_path = None
        self.grasp = None
        self.assign = None
        self.prev = None
        self.prev_a = None
        self.blocked = 0
        self.osc = 0
        self.hist = []
        self.phase_steps = 0
        self.bad_grasp = set()
        self.fails = {}
        self.gp = None
        self.place = None
        self.detour = None
        self.retry = 0
        self.op_steps = 0

    # ------------------------------------------------------------------
    def _assign(self, d):
        slots = self._make_slots(d)
        nb = len(d["blocks"])
        centers = [rect_center(*b) for b in d["blocks"]]
        inside = [i for i in range(nb) if self._inside(d, d["blocks"][i], 0.03)]
        used = [False] * len(slots)
        assign = {}
        for i in sorted(inside, key=lambda i: -centers[i][1]):
            col = min(range(self.k), key=lambda c: abs(self.cols[c] - centers[i][0]))
            best = None
            for si, s in enumerate(slots):
                if not used[si] and s[1] == col:
                    best = si
                    break
            if best is None:
                for si in range(len(slots)):
                    if not used[si]:
                        best = si
                        break
            if best is None:
                continue
            used[best] = True
            assign[i] = slots[best]
        rest = [i for i in range(nb) if i not in assign]
        for si in range(len(slots)):
            if not rest:
                break
            if used[si]:
                continue
            s = slots[si]
            j = min(rest, key=lambda i: (self.fails.get(i, 0),
                                         abs(centers[i][0] - s[2])))
            rest.remove(j)
            used[si] = True
            assign[j] = s
        order = sorted(assign.keys(), key=lambda i: (assign[i][0], assign[i][1]))
        return [(i, assign[i]) for i in order]

    # ------------------------------------------------------------------
    # coarse grid A*
    # ------------------------------------------------------------------
    def _build_grid(self, d, skip, radius):
        res = 0.05
        reach_r = math.hypot(0.2 + d["gw"] / 2, d["gh"] / 2) + 0.005
        x0, x1 = 0.21, ROOM_X1 - 0.21
        y0, y1 = 0.21, WALL_Y0 - reach_r
        nx = int((x1 - x0) / res) + 1
        ny = int((y1 - y0) / res) + 1
        gx = x0 + res * np.arange(nx)
        gy = y0 + res * np.arange(ny)
        PX, PY = np.meshgrid(gx, gy, indexing="ij")
        free = np.ones((nx, ny), dtype=bool)
        for i, b in enumerate(d["blocks"]):
            if i == skip:
                continue
            free &= dist_pts_to_rect(PX, PY, b) > radius
        return free, x0, y0, res, nx, ny

    def _astar(self, grid, start, goal):
        free, x0, y0, res, nx, ny = grid
        si = (min(max(int(round((start[0] - x0) / res)), 0), nx - 1),
              min(max(int(round((start[1] - y0) / res)), 0), ny - 1))
        gi = (min(max(int(round((goal[0] - x0) / res)), 0), nx - 1),
              min(max(int(round((goal[1] - y0) / res)), 0), ny - 1))
        if not free[gi]:
            return None
        if si == gi:
            return [goal]
        nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4143), (-1, 1, 1.4143), (1, -1, 1.4143), (1, 1, 1.4143))
        INF = float("inf")
        dist = {si: 0.0}
        came = {}

        def h(p):
            return math.hypot(p[0] - gi[0], p[1] - gi[1])

        pq = [(h(si), si)]
        found = False
        while pq:
            f, cur = heapq.heappop(pq)
            if cur == gi:
                found = True
                break
            dc = dist.get(cur, INF)
            if f - h(cur) > dc + 1e-9:
                continue
            for dx, dy, w in nbrs:
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                    continue
                if not free[nb]:
                    continue
                nd = dc + w
                if nd < dist.get(nb, INF) - 1e-9:
                    dist[nb] = nd
                    came[nb] = cur
                    heapq.heappush(pq, (nd + h(nb), nb))
        if not found:
            return None
        path = [gi]
        while path[-1] != si:
            path.append(came[path[-1]])
        path.reverse()
        pts = [(x0 + p[0] * res, y0 + p[1] * res) for p in path[1:]]
        pts.append(goal)
        out = []
        for i, p in enumerate(pts):
            if i == 0 or i == len(pts) - 1:
                out.append(p)
                continue
            a = pts[i - 1]
            b = pts[i + 1]
            if abs((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) > 1e-9:
                out.append(p)
        return out

    # ------------------------------------------------------------------
    # motion with exact collision checking
    # ------------------------------------------------------------------
    def _nominal(self, d, tx, ty, tth, tarm):
        return (_clip(tx - d["rx"], MAX_DXY), _clip(ty - d["ry"], MAX_DXY),
                _clip(wrap(tth - d["rth"]), MAX_DTH), _clip(tarm - d["arm"], MAX_DARM))

    def _safe_move(self, d, tx, ty, tth, tarm, vac, skip, holding):
        obst = self._obstacles(d, skip)
        dx, dy, dth, da = self._nominal(d, tx, ty, tth, tarm)
        x, y, th, arm = d["rx"], d["ry"], d["rth"], d["arm"]

        def ok(a):
            return not self._collides(d, x + a[0], y + a[1], th + a[2], arm + a[3],
                                      obst, holding)

        cands = [(dx, dy, dth, da)]
        if abs(dth) > 1e-9 or abs(da) > 1e-9:
            cands.append((dx, dy, 0.0, da))
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            cands.append((0.0, 0.0, dth, da))
            cands.append((0.0, 0.0, 0.0, da))
        if abs(dth) > 1e-9 or abs(da) > 1e-9:
            cands.append((dx, dy, dth, 0.0))
            cands.append((dx, dy, 0.0, 0.0))
        want = math.hypot(dx, dy)
        base_ang = math.atan2(dy, dx) if want > 1e-9 else th
        for k in range(1, 9):
            for sgn in (1, -1):
                ang = base_ang + sgn * k * math.pi / 8
                ax = MAX_DXY * math.cos(ang)
                ay = MAX_DXY * math.sin(ang)
                cands.append((ax, ay, 0.0, 0.0))
                cands.append((ax, ay, dth, da))
        cands.extend([(0, 0, dth, 0), (0, 0, -dth, 0), (0, 0, 0, da), (0, 0, 0, -da),
                      (0, 0, MAX_DTH, 0), (0, 0, -MAX_DTH, 0), (0, 0, 0, -MAX_DARM)])
        good = []
        for a in cands:
            if max(abs(v) for v in a) < 1e-9:
                continue
            if ok(a):
                good.append(a)
                if len(good) > self.blocked + self.osc:
                    break
        out = np.zeros(5, dtype=np.float32)
        if good:
            out[:4] = good[min(self.blocked + self.osc, len(good) - 1)]
        out[4] = vac
        return out

    @staticmethod
    def _at(d, tx, ty, tth, tarm, tol=1e-4):
        return (abs(d["rx"] - tx) < tol and abs(d["ry"] - ty) < tol
                and abs(wrap(d["rth"] - tth)) < tol and abs(d["arm"] - tarm) < tol)

    # ------------------------------------------------------------------
    def get_action(self, state):
        d = self._parse(state)
        self.step_i += 1
        cur = (d["rx"], d["ry"], d["rth"], d["arm"])
        if self.prev is not None and self.prev_a is not None:
            moved = max(abs(cur[i] - self.prev[i]) for i in range(4)) > 1e-7
            want = float(np.max(np.abs(self.prev_a[:4]))) > 1e-7
            self.blocked = self.blocked + 1 if (want and not moved) else 0
        self.prev = cur
        self.hist.append(cur)
        if len(self.hist) > 9:
            self.hist.pop(0)
        rep = 0
        for j in range(len(self.hist) - 3, -1, -1):
            if max(abs(self.hist[j][i] - cur[i]) for i in range(4)) < 1e-6:
                rep += 1
        self.osc = rep
        try:
            a = self._policy(d)
        except Exception:
            a = np.zeros(5, dtype=np.float32)
            a[4] = 1.0 if self.grasp is not None else 0.0
        self.prev_a = a
        return a

    # ------------------------------------------------------------------
    def _face_options(self, d, bi):
        x, y, th, w, h = d["blocks"][bi]
        c, s = math.cos(th), math.sin(th)
        gap = 0.025
        out = []
        for face in (0, 1):
            if face == 0:
                nx_, ny_ = s, -c
                bxm, bym = x, y
            else:
                nx_, ny_ = -s, c
                bxm, bym = x - s * h, y + c * h
            thr = math.atan2(-ny_, -nx_)
            for off in (0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12):
                mx = bxm + (0.5 * w + off) * c
                my = bym + (0.5 * w + off) * s
                for reach in (0.35, 0.45, 0.28, 0.55, 0.65):
                    bx = mx + nx_ * (reach + gap)
                    by = my + ny_ * (reach + gap)
                    if not (0.21 <= bx <= ROOM_X1 - 0.21
                            and 0.21 <= by <= WALL_Y0 - 0.21):
                        continue
                    out.append((thr, bx, by, reach, face, off))
        cx, cy = d["rx"], d["ry"]
        out.sort(key=lambda o: (math.hypot(o[1] - cx, o[2] - cy) / 0.0707
                                + abs(wrap(o[0] - d["rth"])) / MAX_DTH
                                + (o[3] - 0.2) / MAX_DARM
                                + 8.0 * abs(o[5])))
        return out

    # ------------------------------------------------------------------
    def _choose_grasp(self, d, bi, slot=None):
        opts = self._face_options(d, bi)
        grid = None
        obst = self._obstacles(d, None)
        obst_t = self._obstacles(d, bi)
        for o in opts:
            thr, bx, by, reach, face, off = o
            if (bi, face, round(off, 2)) in self.bad_grasp:
                continue
            if slot is not None and not self._place_feasible(slot, face, off):
                continue
            if self._collides(d, bx, by, thr, 0.2, obst, False):
                continue
            bad = False
            a = 0.2
            while a < reach - 1e-9:
                a = min(a + 0.05, reach)
                if self._collides(d, bx, by, thr, a, obst_t, False):
                    bad = True
                    break
            if bad or self._collides(d, bx, by, thr, reach, obst, False):
                continue
            if self._multi_grasp(d, bi, bx, by, thr, reach):
                continue
            if grid is None:
                grid = self._build_grid(d, None, 0.235)
            path = self._astar(grid, (d["rx"], d["ry"]), (bx, by))
            if path is not None:
                return (thr, bx, by, reach, face, off, path)
        for o in opts:
            thr, bx, by, reach, face, off = o
            if (bi, face, round(off, 2)) in self.bad_grasp:
                continue
            if slot is not None and not self._place_feasible(slot, face, off):
                continue
            if self._collides(d, bx, by, thr, 0.2, obst, False):
                continue
            return (thr, bx, by, reach, face, off, [(bx, by)])
        return None

    def _multi_grasp(self, d, bi, x, y, th, arm):
        """True if a block other than bi would also be sucked up."""
        gh = d["gh"]
        ln = d["gw"] + 0.03
        cx = x + (arm + 0.015) * math.cos(th)
        cy = y + (arm + 0.015) * math.sin(th)
        reg = corner_from_center(cx, cy, th, ln, gh) + (th, ln, gh)
        for i, b in enumerate(d["blocks"]):
            if i == bi:
                continue
            if rect_overlap(reg, b):
                return True
        return False

    @staticmethod
    def _place_feasible(slot, face, off):
        bpred = off if face == 0 else -off
        bx = slot[2] + bpred
        return 0.215 <= bx <= ROOM_X1 - 0.215

    # ------------------------------------------------------------------
    def _policy(self, d):
        nb = len(d["blocks"])
        if self.assign is None:
            self.assign = self._assign(d)
            self.op_i = 0
            self.op = None

        while self.op is None:
            if self.op_i >= len(self.assign):
                bad = [i for i in range(nb) if not self._inside(d, d["blocks"][i])]
                if not bad:
                    return np.zeros(5, dtype=np.float32)
                self.assign = self._assign(d)
                self.op_i = 0
                self.bad_grasp = set()
                if not self.assign:
                    return np.zeros(5, dtype=np.float32)
            bi, slot = self.assign[self.op_i]
            cx, cy = rect_center(*d["blocks"][bi])
            done = (self._inside(d, d["blocks"][bi], -0.003)
                    and abs(cy - slot[3]) < 0.03 and abs(cx - slot[2]) < 0.06)
            if done:
                self.op_i += 1
                continue
            self.op = (bi, slot)
            self.op_steps = 0
            self.phase = "nav_grasp"
            self.phase_steps = 0
            self.plan_path = None
            self.grasp = None
            self.gp = None
            break

        bi, slot = self.op
        self.phase_steps += 1
        self.op_steps += 1
        if self.op_steps > 140:
            self.op_steps = 0
            self.op = None
            self.op_i += 1
            self.grasp = None
            self.gp = None
            self.fails[bi] = self.fails.get(bi, 0) + 1
            return np.zeros(5, dtype=np.float32)
        return self._run_op(d, bi, slot)

    # ------------------------------------------------------------------
    def _run_op(self, d, bi, slot):
        if self.phase == "nav_grasp":
            if self.gp is None:
                self.gp = self._choose_grasp(d, bi, slot)
                if self.gp is None:
                    self.op = None
                    self.op_i += 1
                    return np.zeros(5, dtype=np.float32)
                self.plan_path = list(self.gp[6])
            thr, bx, by, reach, face, off, _ = self.gp
            if self.phase_steps > 70:
                self.bad_grasp.add((bi, face, round(off, 2)))
                self.gp = None
                self.phase_steps = 0
                return np.zeros(5, dtype=np.float32)
            while self.plan_path and math.hypot(self.plan_path[0][0] - d["rx"],
                                                self.plan_path[0][1] - d["ry"]) < 0.02:
                self.plan_path.pop(0)
            if self.plan_path:
                wx, wy = self.plan_path[0]
                return self._safe_move(d, wx, wy, thr, 0.2, 0.0, None, False)
            if not self._at(d, bx, by, thr, 0.2, 1e-4):
                return self._safe_move(d, bx, by, thr, 0.2, 0.0, None, False)
            self.phase = "reach"
            self.phase_steps = 0

        if self.phase == "reach":
            thr, bx, by, reach, face, off, _ = self.gp
            if d["arm"] >= reach - 1e-4 or self.blocked >= 1 or self.phase_steps > 8:
                self.phase = "grasp"
                self.phase_steps = 0
                a = np.zeros(5, dtype=np.float32)
                a[4] = 1.0
                return a
            a = np.zeros(5, dtype=np.float32)
            a[3] = _clip(reach - d["arm"], MAX_DARM)
            return a

        if self.phase == "grasp":
            self.grasp = self._measure(d, bi)
            if not self._grasp_valid(d, bi):
                self.bad_grasp.add((bi, self.gp[4], round(self.gp[5], 2)))
                self.grasp = None
                self.gp = None
                self.phase = "nav_grasp"
                self.phase_steps = 0
                return np.zeros(5, dtype=np.float32)
            self.phase = "nav_place"
            self.plan_path = None
            self.phase_steps = 0

        if self.phase == "nav_place":
            if not self._holding_ok(d, bi):
                self.bad_grasp.add((bi, self.gp[4], round(self.gp[5], 2)))
                self.grasp = None
                self.gp = None
                self.phase = "nav_grasp"
                self.phase_steps = 0
                return np.zeros(5, dtype=np.float32)
            A, B, C, w, h = self.grasp
            X, Y = slot[2], slot[3]
            # robot heading that makes the held block exactly horizontal
            cands_t = [wrap(-C), wrap(math.pi - C)]
            thr = min(cands_t, key=lambda t: abs(wrap(t - math.pi / 2)))
            ct, st = math.cos(thr), math.sin(thr)
            lim_y = min(WALL_Y0 - 0.2214,
                        WALL_Y0 - 0.2 - abs(A) - h / 2 - 0.006)
            af = 0.78
            for trial in (0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78):
                byt = Y - ((trial + A) * st + B * ct)
                if 0.21 <= byt <= lim_y:
                    af = trial
                    break
            by = Y - ((af + A) * st + B * ct)
            bx = X - ((af + A) * ct - B * st)
            if not (0.212 <= bx <= ROOM_X1 - 0.212 and 0.21 <= by <= lim_y):
                self.bad_grasp.add((bi, self.gp[4], round(self.gp[5], 2)))
                self.grasp = None
                self.gp = None
                self.phase = "nav_grasp"
                self.phase_steps = 0
                return np.zeros(5, dtype=np.float32)
            self.place = (bx, by, thr, af)
            if self.phase_steps > 0 and self.phase_steps % 45 == 0:
                self.plan_path = None
                if self.phase_steps % 90 == 0:
                    self.detour = (min(max(d["rx"], 0.4), ROOM_X1 - 0.4), 0.55)
            if self.plan_path is None:
                grid = self._build_grid(d, bi, 0.34)
                path = self._astar(grid, (d["rx"], d["ry"]), (bx, by))
                if path is None:
                    grid = self._build_grid(d, bi, 0.235)
                    path = self._astar(grid, (d["rx"], d["ry"]), (bx, by))
                if path is None:
                    path = [(bx, by)]
                if self.detour is not None:
                    path = [self.detour] + path
                    self.detour = None
                self.plan_path = path
            while self.plan_path and math.hypot(self.plan_path[0][0] - d["rx"],
                                                self.plan_path[0][1] - d["ry"]) < 0.02:
                self.plan_path.pop(0)
            if self.plan_path:
                wx, wy = self.plan_path[0]
                return self._safe_move(d, wx, wy, thr, 0.2, 1.0, bi, True)
            if not self._at(d, bx, by, thr, 0.2, 1e-4):
                return self._safe_move(d, bx, by, thr, 0.2, 1.0, bi, True)
            self.phase = "insert"
            self.phase_steps = 0

        if self.phase == "insert":
            bx, by, thr, af = self.place
            hb = self._held_rect(d, d["rx"], d["ry"], d["rth"], d["arm"])
            seated = False
            if hb is not None and self._inside(d, hb, -0.0015):
                hcy = rect_center(*hb)[1]
                seated = hcy >= slot[3] - 0.022
            if seated:
                self.phase = "release"
                self.phase_steps = 0
            elif d["arm"] < af - 1e-4 and self.blocked < 2 and self.phase_steps <= 16:
                a = np.zeros(5, dtype=np.float32)
                a[3] = _clip(af - d["arm"], MAX_DARM)
                a[4] = 1.0
                return a
            else:
                self.retry = getattr(self, "retry", 0) + 1
                if self.retry > 3:
                    self.retry = 0
                    self.bad_grasp.add((bi, self.gp[4], round(self.gp[5], 2)))
                    self.grasp = None
                    self.gp = None
                    self.phase = "nav_grasp"
                    self.phase_steps = 0
                    return np.zeros(5, dtype=np.float32)
                self.phase = "nav_place"
                self.plan_path = None
                if self.retry >= 2:
                    self.detour = (min(max(d["rx"], 0.4), ROOM_X1 - 0.4), 1.6)
                self.phase_steps = 0
                a = np.zeros(5, dtype=np.float32)
                a[3] = -MAX_DARM
                a[4] = 1.0
                return a

        if self.phase == "release":
            self.grasp = None
            self.phase = "done"
            self.phase_steps = 0
            self.op = None
            self.op_i += 1
            return np.zeros(5, dtype=np.float32)

        return np.zeros(5, dtype=np.float32)

    # ------------------------------------------------------------------
    def _measure(self, d, bi):
        bx, by, bth, w, h = d["blocks"][bi]
        cx, cy = rect_center(bx, by, bth, w, h)
        dx, dy = cx - d["rx"], cy - d["ry"]
        c, s = math.cos(d["rth"]), math.sin(d["rth"])
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        return (lx - d["arm"], ly, wrap(bth - d["rth"]), w, h)

    def _holding_ok(self, d, bi):
        if self.grasp is None:
            return False
        pred = self._held_rect(d, d["rx"], d["ry"], d["rth"], d["arm"])
        b = d["blocks"][bi]
        return (abs(pred[0] - b[0]) < 6e-3 and abs(pred[1] - b[1]) < 6e-3
                and abs(wrap(pred[2] - b[2])) < 2e-2)

    def _grasp_valid(self, d, bi):
        A, B, C, w, h = self.grasp
        if abs(A) > h / 2 + 0.05 or abs(B) > d["gh"] / 2 + w / 2:
            return False
        return True
