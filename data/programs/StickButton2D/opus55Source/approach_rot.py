import time
import numpy as np

BASE_R = 0.1
BTN_R = 0.05
STICK_W = 0.05
STICK_H = 1.25
XMIN, XMAX = 0.0, 3.5
YMIN, YMAX = 0.0, 2.5
TABLE_Y = 1.25
STEP = 0.05
MAXTH = np.pi / 16
GRASP_D = 0.113  # robot center -> stick bottom when grasping from below
# bounds for robot center while holding the stick
HX_LO, HX_HI = BASE_R + 0.004, XMAX - BASE_R - 0.004
HY_LO, HY_HI = BASE_R + 0.004, TABLE_Y - BASE_R - 0.004
# bounds before grasp (arbitrary orientation; keep gripper off walls)
PX_LO, PX_HI = 0.118, XMAX - 0.118
PY_LO, PY_HI = 0.118, TABLE_Y - BASE_R - 0.004
BASE_PRESS = BASE_R + BTN_R - 0.012
STICK_M = 0.012
CLEAR = 0.122  # clearance between robot center and (free) stick rectangle
WALL_EPS = 0.004
TH_GRID = np.pi / 2 + np.arange(-16, 17) * (np.pi / 32)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rect_dist(px, py, x0, y0, x1, y1):
    dx = np.maximum(np.maximum(x0 - px, 0.0), px - x1)
    dy = np.maximum(np.maximum(y0 - py, 0.0), py - y1)
    return np.hypot(dx, dy)


def nsteps(p, q):
    d = max(abs(q[0] - p[0]), abs(q[1] - p[1]))
    return int(np.ceil(d / STEP - 1e-9))


def path_points(p, q):
    n = nsteps(p, q)
    if n == 0:
        return np.zeros((0, 2))
    k = np.arange(1, n + 1) * STEP
    dx, dy = q[0] - p[0], q[1] - p[1]
    xs = p[0] + np.sign(dx) * np.minimum(k, abs(dx))
    ys = p[1] + np.sign(dy) * np.minimum(k, abs(dy))
    return np.stack([xs, ys], 1)


def cfg_steps(c0, c1):
    dth = abs(wrap(c1[2] - c0[2]))
    return int(np.ceil(max(abs(c1[0] - c0[0]) / STEP, abs(c1[1] - c0[1]) / STEP,
                           dth / MAXTH) - 1e-9))


def cfg_path(c0, c1):
    n = cfg_steps(c0, c1)
    if n == 0:
        return np.zeros((0, 3))
    k = np.arange(1, n + 1)
    out = np.empty((n, 3))
    for j, mx in ((0, STEP), (1, STEP)):
        d = c1[j] - c0[j]
        out[:, j] = c0[j] + np.sign(d) * np.minimum(k * mx, abs(d))
    d = wrap(c1[2] - c0[2])
    out[:, 2] = c0[2] + np.sign(d) * np.minimum(k * MAXTH, abs(d))
    return out


class World:
    """Simple geometric model used for planning."""

    def __init__(self, bx, by, stick_xy):
        self.bx = np.asarray(bx, dtype=float)
        self.by = np.asarray(by, dtype=float)
        self.sx, self.sy = stick_xy  # stick bottom-left corner (free stick)

    # ================= before grasp (translation only) =================
    def base_target(self, i, p):
        bx, by = self.bx[i], self.by[i]
        xlo, xhi, ylo, yhi = PX_LO, PX_HI, PY_LO, PY_HI
        cands = []
        vx, vy = p[0] - bx, p[1] - by
        nrm = np.hypot(vx, vy)
        if nrm > 1e-9:
            cands.append((bx + vx / nrm * BASE_PRESS, by + vy / nrm * BASE_PRESS))
        r2 = BASE_PRESS / np.sqrt(2)
        cands.append((bx + np.sign(vx) * r2, by + np.sign(vy) * r2))
        cands.append((bx, by))
        if nrm <= BASE_PRESS:
            cands.append((p[0], p[1]))
        best = None
        for (cx, cy) in cands:
            cx = min(max(cx, xlo), xhi)
            cy = min(max(cy, ylo), yhi)
            if np.hypot(cx - bx, cy - by) <= BASE_PRESS + 1e-9:
                d = max(abs(cx - p[0]), abs(cy - p[1]))
                if best is None or d < best[0]:
                    best = (d, (cx, cy))
        return None if best is None else best[1]

    def presses_free(self, pts):
        if len(pts) == 0 or len(self.bx) == 0:
            return np.zeros(len(self.bx), dtype=bool)
        d = np.hypot(pts[:, 0:1] - self.bx[None], pts[:, 1:2] - self.by[None])
        return (d < BASE_R + BTN_R - 0.003).any(0)

    def collides(self, pts):
        if len(pts) == 0:
            return False
        d = rect_dist(pts[:, 0], pts[:, 1], self.sx, self.sy,
                      self.sx + STICK_W, self.sy + STICK_H)
        return bool((d < CLEAR).any())

    def grasp_waypoints(self, p):
        """Waypoints (excluding ones equal to p) ending at the grasp pose."""
        scx = self.sx + STICK_W / 2
        sb = self.sy
        W = (scx, sb - 0.14)
        G = (scx, sb - GRASP_D)
        if abs(p[0] - G[0]) < 1e-7 and G[1] - 0.03 <= p[1] <= G[1] + 1e-7:
            return [G] if abs(p[1] - G[1]) > 1e-7 else []
        low = max(PY_LO, sb - 0.26)
        options = [[W], [(p[0], low), W]]
        for side in (-1, 1):
            ex = (self.sx - 0.135) if side < 0 else (self.sx + STICK_W + 0.135)
            if PX_LO <= ex <= PX_HI:
                options.append([(ex, p[1]), (ex, low), W])
        for opt in options:
            q = p
            ok = True
            for w in opt:
                if self.collides(path_points(q, w)):
                    ok = False
                    break
                q = w
            if ok:
                out = list(opt) + [G]
                while out and abs(out[0][0] - p[0]) < 1e-7 and abs(out[0][1] - p[1]) < 1e-7:
                    out.pop(0)
                return out
        return [W, G]

    # ================= holding the stick (x, y, theta) =================
    @staticmethod
    def _stick_frames(C, h):
        """C: (K,3). h=(relx, rely, phi). Returns B (K,2), v (K,2), n (K,2)."""
        relx, rely, phi = h
        c, s = np.cos(C[:, 2]), np.sin(C[:, 2])
        Bx = C[:, 0] + relx * c - rely * s
        By = C[:, 1] + relx * s + rely * c
        a = C[:, 2] + phi
        v = np.stack([np.cos(a), np.sin(a)], 1)
        n = np.stack([-v[:, 1], v[:, 0]], 1)
        return np.stack([Bx, By], 1), v, n

    def hold_feasible(self, C, h):
        if len(C) == 0:
            return np.zeros(0, dtype=bool)
        ok = (C[:, 0] >= HX_LO) & (C[:, 0] <= HX_HI) & (C[:, 1] >= HY_LO) & (C[:, 1] <= HY_HI)
        B, v, n = self._stick_frames(C, h)
        hw = STICK_W / 2
        u = np.stack([np.cos(C[:, 2]), np.sin(C[:, 2])], 1)
        un = np.stack([-u[:, 1], u[:, 0]], 1)
        P = C[:, :2]
        corners = [B + hw * n, B - hw * n, B + STICK_H * v + hw * n, B + STICK_H * v - hw * n,
                   P + 0.111 * u + 0.036 * un, P + 0.111 * u - 0.036 * un]
        for q in corners:
            ok &= (q[:, 0] > WALL_EPS) & (q[:, 0] < XMAX - WALL_EPS)
            ok &= (q[:, 1] > WALL_EPS) & (q[:, 1] < YMAX - WALL_EPS)
        return ok

    def hold_press_matrix(self, C, h, margin=0.003):
        """(K, N) bool: config k presses button j."""
        if len(C) == 0 or len(self.bx) == 0:
            return np.zeros((len(C), len(self.bx)), dtype=bool)
        dx = self.bx[None] - C[:, 0:1]
        dy = self.by[None] - C[:, 1:2]
        hit = np.hypot(dx, dy) < BASE_R + BTN_R - margin
        B, v, n = self._stick_frames(C, h)
        rx = self.bx[None] - B[:, 0:1]
        ry = self.by[None] - B[:, 1:2]
        s = rx * v[:, 0:1] + ry * v[:, 1:2]
        t = rx * n[:, 0:1] + ry * n[:, 1:2]
        ds = np.maximum(np.maximum(-s, 0.0), s - STICK_H)
        dt = np.maximum(np.abs(t) - STICK_W / 2, 0.0)
        hit |= np.hypot(ds, dt) < BTN_R - margin
        return hit

    def hold_target(self, i, c, h, rotate=True):
        bx, by = self.bx[i], self.by[i]
        cands = []
        # base contact, keep theta
        vx, vy = c[0] - bx, c[1] - by
        nrm = np.hypot(vx, vy)
        pts = []
        if nrm > 1e-9:
            pts.append((bx + vx / nrm * BASE_PRESS, by + vy / nrm * BASE_PRESS))
        r2 = BASE_PRESS / np.sqrt(2)
        pts.append((bx + np.sign(vx) * r2, by + np.sign(vy) * r2))
        pts.append((bx, by))
        for (px, py) in pts:
            cands.append((px, py, c[2]))
            cands.append((min(max(px, HX_LO), HX_HI), min(max(py, HY_LO), HY_HI), c[2]))
        # stick contact
        T = np.concatenate([[c[2]], TH_GRID]) if rotate else np.array([c[2]])
        relx, rely, phi = h
        cs, sn = np.cos(T), np.sin(T)
        offx = relx * cs - rely * sn
        offy = relx * sn + rely * cs
        a = T + phi
        v = np.stack([np.cos(a), np.sin(a)], 1)
        n = np.stack([-v[:, 1], v[:, 0]], 1)
        qx = bx - offx
        qy = by - offy  # robot pos if button at stick bottom-center
        # coordinates of p relative: q = qstar - s v - t n
        dx, dy = qx - c[0], qy - c[1]
        sp = dx * v[:, 0] + dy * v[:, 1]
        tp = dx * n[:, 0] + dy * n[:, 1]
        allc = [np.array(cands)]
        for (slo, shi, tw) in ((-(BTN_R - STICK_M), STICK_H + BTN_R - STICK_M, STICK_W / 2 - 0.004),
                               (0.004, STICK_H - 0.004, STICK_W / 2 + BTN_R - STICK_M)):
            ss = np.clip(sp, slo, shi)
            tt = np.clip(tp, -tw, tw)
            X = qx - ss * v[:, 0] - tt * n[:, 0]
            Y = qy - ss * v[:, 1] - tt * n[:, 1]
            allc.append(np.stack([X, Y, T], 1))
            allc.append(np.stack([np.clip(X, HX_LO, HX_HI), np.clip(Y, HY_LO, HY_HI), T], 1))
        C = np.concatenate(allc, 0)
        ok = self.hold_feasible(C, h)
        if not ok.any():
            return None
        C = C[ok]
        pm = self.hold_press_matrix(C, h, margin=0.008)[:, i]
        if not pm.any():
            return None
        C = C[pm]
        dth = np.abs(wrap(C[:, 2] - c[2])) / MAXTH
        cost = np.maximum(np.maximum(np.abs(C[:, 0] - c[0]), np.abs(C[:, 1] - c[1])) / STEP, dth)
        cost = np.ceil(cost - 1e-9)
        order = np.argsort(cost, kind="stable")
        for k in order[:6]:
            tgt = tuple(C[k])
            path = cfg_path(c, tgt)
            if self.hold_feasible(path, h).all():
                return tgt
        return None


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    # ---------- state helpers ----------
    def _parse(self, state):
        robot = None
        stick = None
        buttons = []
        for n in sorted(state.get_object_names()):
            o = state.get_object_from_name(n)
            t = o.type.name
            if t == "crv_robot":
                robot = o
            elif t == "rectangle":
                stick = o
            elif t == "circle":
                buttons.append(o)
        g = state.get
        r = dict(x=g(robot, "x"), y=g(robot, "y"), th=g(robot, "theta"),
                 arm=g(robot, "arm_joint"), vac=g(robot, "vacuum"))
        s = dict(x=g(stick, "x"), y=g(stick, "y"), th=g(stick, "theta"))
        btns = []
        for b in buttons:
            pressed = g(b, "color_g") > 0.5 and g(b, "color_r") < 0.5
            btns.append(dict(name=b.name, x=g(b, "x"), y=g(b, "y"), pressed=pressed))
        return r, s, btns

    def reset(self, state, info):
        self.holding = False
        self.grasp_cmd = False
        self.prev = None
        self.prev_action = None
        self.plan = None
        self.replans = 0
        self.h = (GRASP_D, 0.0, 0.0)
        self.cur_item = None
        self.cur_tgt = None
        self.rng = np.random.default_rng(0)
        r, s, btns = self._parse(state)
        scx = s["x"] + STICK_W / 2
        self.phase = "normal"
        if scx < HX_LO + 0.002 or scx > HX_HI - 0.002:
            self.phase = "side_approach"
            self.side_d = 1.0 if scx < XMAX / 2 else -1.0
            world = World([b["x"] for b in btns], [b["y"] for b in btns], (s["x"], s["y"]))
            p = (r["x"], r["y"])
            if all(world.base_target(i, p) is not None for i in range(len(btns))):
                self.phase = "normal"

    def _act(self, dx, dy, dth, vac):
        a = np.array([np.clip(dx, -STEP, STEP), np.clip(dy, -STEP, STEP),
                      np.clip(dth, -MAXTH, MAXTH), -0.1, vac], dtype=np.float32)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        self.prev_action = a
        return a

    # ---------- planning ----------
    def _sim(self, world, seq, c0, pressed0, holding0, h, t_rot0, greedy_items=None):
        """Simulate a sequence (optionally completing greedily).
        Returns (cost, full_sequence)."""
        c = c0
        pressed = pressed0.copy()
        holding = holding0
        t = 0
        out = []

        def move_hold(c, tgt):
            path = cfg_path(c, tgt)
            if len(path):
                pressed[:] |= world.hold_press_matrix(path, h).any(0)
            return len(path)

        for item in seq:
            if item == "G":
                if holding:
                    continue
                p = (c[0], c[1])
                wps = world.grasp_waypoints(p)
                for k, w in enumerate(wps):
                    pts = path_points(p, w)
                    if k < len(wps) - 1 and world.collides(pts):
                        return np.inf, None
                    pressed |= world.presses_free(pts)
                    t += len(pts)
                    if k == len(wps) - 2:
                        t = max(t, t_rot0)
                    p = w
                t += 1
                holding = True
                h = (GRASP_D, 0.0, 0.0)
                c = (p[0], p[1], np.pi / 2)
                out.append("G")
                continue
            if pressed[item]:
                continue
            if holding:
                tgt = world.hold_target(item, c, h)
                if tgt is None:
                    return np.inf, None
                t += move_hold(c, tgt)
                pressed[item] = True
                c = tgt
            else:
                p = (c[0], c[1])
                tgt = world.base_target(item, p)
                if tgt is None:
                    return np.inf, None
                pts = path_points(p, tgt)
                if world.collides(pts):
                    return np.inf, None
                pressed |= world.presses_free(pts)
                pressed[item] = True
                t += len(pts)
                c = (tgt[0], tgt[1], c[2])
            out.append(item)
        if greedy_items is not None:
            while True:
                rem = [i for i in greedy_items if not pressed[i]]
                if not rem:
                    break
                best = None
                for i in rem:
                    if holding:
                        tgt = world.hold_target(i, c, h)
                        if tgt is None:
                            continue
                        d = cfg_steps(c, tgt)
                    else:
                        p = (c[0], c[1])
                        tgt = world.base_target(i, p)
                        if tgt is None or world.collides(path_points(p, tgt)):
                            continue
                        d = nsteps(p, tgt)
                        tgt = (tgt[0], tgt[1], c[2])
                    if best is None or d < best[0]:
                        best = (d, i, tgt)
                if best is None:
                    return np.inf, None
                _, i, tgt = best
                if holding:
                    t += move_hold(c, tgt)
                else:
                    pts = path_points((c[0], c[1]), (tgt[0], tgt[1]))
                    pressed |= world.presses_free(pts)
                    t += len(pts)
                pressed[i] = True
                c = tgt
                out.append(i)
        if not pressed.all():
            return np.inf, None
        return t, out

    def _make_plan(self, r, s, btns, budget=1.5):
        t_start = time.time()
        world = World([b["x"] for b in btns], [b["y"] for b in btns], (s["x"], s["y"]))
        pressed0 = np.array([b["pressed"] for b in btns], dtype=bool)
        items = [i for i in range(len(btns)) if not pressed0[i]]
        c0 = (r["x"], r["y"], r["th"])
        holding = self.holding
        h = self.h
        t_rot = int(np.ceil(abs(wrap(np.pi / 2 - r["th"])) / MAXTH - 1e-9))
        rng = self.rng

        def ev(seq):
            return self._sim(world, seq, c0, pressed0, holding, h, t_rot)[0]

        best_seq, best_c = None, np.inf
        prefixes = [[]] if holding else ([["G"]] + [[i, "G"] for i in items] + [[]])
        for prefix in prefixes:
            if time.time() - t_start > budget * 0.6 and best_seq is not None:
                break
            c, seq = self._sim(world, prefix, c0, pressed0, holding, h, t_rot,
                               greedy_items=items)
            if seq is None:
                continue
            if c < best_c:
                best_seq, best_c = list(seq), c
        if best_seq is None:
            best_seq = (["G"] if not holding else []) + items
            best_c = ev(best_seq)
        cur = list(best_seq) + [i for i in items if i not in best_seq]
        if not holding and "G" not in cur:
            cur = cur + ["G"]
        cur_c = ev(cur)
        n = len(cur)
        it = 0
        while n >= 2 and time.time() - t_start < budget and it < 4000:
            it += 1
            new = list(cur)
            mv = rng.integers(3)
            i, j = sorted(rng.choice(n, 2, replace=False))
            if mv == 0:
                new[i], new[j] = new[j], new[i]
            elif mv == 1:
                x = new.pop(i)
                new.insert(j, x)
            else:
                new[i:j + 1] = new[i:j + 1][::-1]
            v = ev(new)
            if v <= cur_c and v < np.inf:
                cur, cur_c = new, v
        if cur_c <= best_c:
            best_seq, best_c = cur, cur_c
        self.world = world
        self.plan = list(best_seq)
        self.plan_cost = best_c
        self.cur_item = None

    # ---------- main ----------
    def get_action(self, state):
        r, s, btns = self._parse(state)
        rejected = False
        if self.prev is not None and self.prev_action is not None:
            pr = self.prev
            moved = abs(pr["x"] - r["x"]) + abs(pr["y"] - r["y"]) + abs(wrap(pr["th"] - r["th"])) > 1e-9
            wanted = np.abs(self.prev_action[:3]).sum() > 1e-9
            rejected = wanted and not moved
        self.prev = r

        if self.grasp_cmd and r["vac"] > 0.5:
            self.holding = True
            self.grasp_cmd = False
            # stick bottom-center and axis in robot frame
            st = s["th"]
            bcx = s["x"] + np.cos(st) * STICK_W / 2
            bcy = s["y"] + np.sin(st) * STICK_W / 2
            dx, dy = bcx - r["x"], bcy - r["y"]
            c, sn = np.cos(r["th"]), np.sin(r["th"])
            self.h = (c * dx + sn * dy, -sn * dx + c * dy, wrap(st + np.pi / 2 - r["th"]))
            self.cur_item = None

        if self.phase != "normal":
            a = self._side_step(r, s)
            if a is not None:
                return a

        if self.plan is None:
            self._make_plan(r, s, btns)
        return self._follow(r, s, btns, rejected)

    def _follow(self, r, s, btns, rejected, depth=0):
        p = (r["x"], r["y"])
        c = (r["x"], r["y"], r["th"])
        vac = 1.0 if self.holding else 0.0
        while self.plan:
            item = self.plan[0]
            if item == "G":
                if self.holding or all(b["pressed"] for b in btns):
                    self.plan.pop(0)
                    continue
                return self._grasp_step(r, s, rejected)
            if btns[item]["pressed"]:
                self.plan.pop(0)
                self.cur_item = None
                continue
            if self.holding:
                if self.cur_item != item or self.cur_tgt is None or rejected:
                    self.cur_item = item
                    self.cur_tgt = self.world.hold_target(item, c, self.h)
                tgt = self.cur_tgt
                if tgt is None or cfg_steps(c, tgt) == 0:
                    self.plan.pop(0)
                    self.cur_item = None
                    continue
                if rejected:
                    dx, dy = self.rng.uniform(-STEP, STEP, size=2)
                    return self._act(dx, dy, 0.0, vac)
                return self._act(tgt[0] - c[0], tgt[1] - c[1], wrap(tgt[2] - c[2]), vac)
            tgt = self.world.base_target(item, p)
            if tgt is None or max(abs(tgt[0] - p[0]), abs(tgt[1] - p[1])) < 1e-7:
                self.plan.pop(0)
                continue
            dth = 0.0 if rejected else wrap(np.pi / 2 - r["th"])
            dx, dy = tgt[0] - p[0], tgt[1] - p[1]
            if rejected:
                dx, dy = self.rng.uniform(-STEP, STEP, size=2)
            return self._act(dx, dy, dth, vac)
        if all(b["pressed"] for b in btns):
            return self._act(0, 0, 0, vac)
        self.replans += 1
        if self.replans > 30 or depth > 0:
            dx, dy = self.rng.uniform(-STEP, STEP, size=2)
            return self._act(dx, dy, 0.0, vac)
        self.world = World([b["x"] for b in btns], [b["y"] for b in btns], (s["x"], s["y"]))
        self._make_plan(r, s, btns, budget=0.3)
        if not self.holding and "G" not in self.plan:
            self.plan.append("G")
        return self._follow(r, s, btns, rejected, depth + 1)

    # ---------- relocation of a stick stuck at a side wall ----------
    def _side_step(self, r, s):
        d = self.side_d
        sb = s["y"]
        if d > 0:
            tx = s["x"] + STICK_W + 0.113
            th = np.pi
        else:
            tx = s["x"] - 0.113
            th = 0.0
        if self.phase == "side_approach":
            if not hasattr(self, "side_ty"):
                self.side_ty = float(min(PY_HI, max(sb + 0.04, r["y"])))
            ty = self.side_ty
            dth = wrap(th - r["th"])
            if abs(dth) > 1e-6 or abs(r["y"] - ty) > 1e-6:
                px = tx + d * 0.15
                return self._act(px - r["x"], ty - r["y"], dth, 0.0)
            if abs(r["x"] - tx) > 1e-6:
                return self._act(tx - r["x"], 0.0, 0.0, 0.0)
            if r["vac"] > 0.5:
                self.phase = "side_carry"
                self.carry_x = tx + d * 0.3
            else:
                return self._act(0.0, 0.0, 0.0, 1.0)
        if self.phase == "side_carry":
            if abs(r["x"] - self.carry_x) > 1e-6:
                return self._act(self.carry_x - r["x"], 0.0, 0.0, 1.0)
            self.phase = "normal"
            self.plan = None
            return self._act(0.0, 0.0, 0.0, 0.0)
        return None

    # ---------- grasping from below ----------
    def _grasp_step(self, r, s, rejected):
        if self.grasp_cmd:
            return self._act(0.0, 0.0, 0.0, 1.0)
        p = (r["x"], r["y"])
        world = World([], [], (s["x"], s["y"]))
        dth = wrap(np.pi / 2 - r["th"])
        wps = world.grasp_waypoints(p)
        if not wps:
            if abs(dth) < 1e-7:
                self.grasp_cmd = True
                return self._act(0.0, 0.0, 0.0, 1.0)
            return self._act(0.0, -STEP, 0.0, 0.0)
        tgt = wps[0]
        if len(wps) == 1 and abs(dth) > 1e-7:
            if rejected:
                return self._act(0.0, -STEP, 0.0, 0.0)
            return self._act(0.0, 0.0, dth, 0.0)
        dth_cmd = 0.0 if rejected else dth
        dx, dy = tgt[0] - p[0], tgt[1] - p[1]
        if rejected and abs(self.prev_action[2]) < 1e-9:
            dx, dy = self.rng.uniform(-STEP, STEP, size=2)
        return self._act(dx, dy, dth_cmd, 0.0)
