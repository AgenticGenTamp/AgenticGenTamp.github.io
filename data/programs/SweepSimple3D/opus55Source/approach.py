import numpy as np

# Arm IK table: pinch site at base-frame (HOLD_R, 0, z), gripper pointing down,
# fingers closing along base lateral axis (computed offline with MuJoCo DLS IK).
HOLD_R = 0.5
ARM_TABLE_Z = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
ARM_TABLE_Q = np.array([
    [0.00293, 1.71363, 3.14235, -1.54392, 0.00647, 0.11597, 1.56719],
    [0.00291, 1.69003, 3.14237, -1.57169, 0.0064, 0.12013, 1.56726],
    [0.0029, 1.66644, 3.14238, -1.59869, 0.00634, 0.12354, 1.56733],
    [0.0029, 1.64281, 3.14238, -1.62498, 0.00627, 0.1262, 1.56741],
    [0.0029, 1.61921, 3.14239, -1.6505, 0.00621, 0.12812, 1.56749],
    [0.0029, 1.59554, 3.14239, -1.67534, 0.00615, 0.12929, 1.56758],
    [0.00291, 1.57181, 3.14238, -1.69949, 0.00609, 0.1297, 1.56766],
    [0.00292, 1.54801, 3.14237, -1.72294, 0.00603, 0.12936, 1.56775],
    [0.00293, 1.52412, 3.14236, -1.74571, 0.00598, 0.12824, 1.56783],
])

Z_GRASP = 0.035
G_OPEN = 0.55
V_CARRY = 0.1
Z_APPROACH = 0.07
Z_HIGH = 0.10
FAR_D = 0.06
Z_CARRY = 0.08
THETA = -np.pi / 2

# goal region (padded check region is +-0.05)
GX0, GX1, GY0, GY1 = 0.125, 0.875, 1.0, 1.375
CHK = (0.085, 0.915, 0.96, 1.415)


FK_CHAIN = [([0.1199, 0.0, 0.3948], [1.0, 0.0, 0.0, 0.0], None),
            ([0.0, 0.0, 0.15643], [0.0, 1.0, 0.0, 0.0], 0),
            ([0.0, 0.005375, -0.12838], [0.707107, 0.707107, 0.0, 0.0], 1),
            ([0.0, -0.21038, -0.006375], [0.707107, -0.707107, 0.0, 0.0], 2),
            ([0.0, 0.006375, -0.21038], [0.707107, 0.707107, 0.0, 0.0], 3),
            ([0.0, -0.20843, -0.006375], [0.707107, -0.707107, 0.0, 0.0], 4),
            ([0.0, 0.000175, -0.10593], [0.707107, 0.707107, 0.0, 0.0], 5),
            ([0.0, -0.10593, -0.000175], [0.707107, -0.707107, 0.0, 0.0], 6)]
FK_SITE = np.array([0.0, 0.0, -0.181525])


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


_FK = [(np.array(p), _quat_mat(q), j) for p, q, j in FK_CHAIN]


def fk_pinch(q):
    """Pinch-site position in the base frame for arm joints q."""
    R = np.eye(3)
    p = np.zeros(3)
    for pos, Rb, j in _FK:
        p = p + R @ pos
        R = R @ Rb
        if j is not None:
            c, s = np.cos(q[j]), np.sin(q[j])
            R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return p + R @ FK_SITE


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def arm_q(z):
    z = float(np.clip(z, ARM_TABLE_Z[0], ARM_TABLE_Z[-1]))
    return np.array([np.interp(z, ARM_TABLE_Z, ARM_TABLE_Q[:, i]) for i in range(7)])


def in_goal(c):
    return CHK[0] <= c[0] <= CHK[1] and CHK[2] <= c[1] <= CHK[3]


def make_slots():
    slots = []
    for x in np.arange(0.88, 0.14, -0.04):
        for y in np.arange(1.05, 1.39, 0.04):
            slots.append(np.array([x, y]))
    return slots


class Target:
    def __init__(self, base=None, arm=None, grip=None, vbase=0.1, vrot=0.1,
                 varm=0.1, tol=0.01, tol_arm=0.03, max_steps=80, min_steps=0, name=""):
        self.base = None if base is None else np.array(base, float)
        self.arm = None if arm is None else np.array(arm, float)
        self.grip = grip
        self.vbase, self.vrot, self.varm = vbase, vrot, varm
        self.tol, self.tol_arm = tol, tol_arm
        self.max_steps, self.min_steps = max_steps, min_steps
        self.steps = 0
        self.name = name
        self.lower_z = None
        self.pinch = None
        self.track = None
        self.lowering = False
        self.far_z = None
        self.track_off = np.zeros(2)

    def eff_arm(self, base):
        if self.arm is None:
            return None
        if self.far_z is not None and self.base is not None and \
                np.hypot(*(self.base[:2] - base[:2])) > FAR_D:
            return arm_q(self.far_z)
        return self.arm


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------------------------ state
    def _read(self, state):
        robot = None
        cubes = []
        self.wiper = None
        for obj in state:
            n = obj.name
            if n == "robot":
                robot = obj
            elif n.startswith("cube"):
                qw, qz = state.get(obj, "qw"), state.get(obj, "qz")
                qx, qy = state.get(obj, "qx"), state.get(obj, "qy")
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                cubes.append(np.array([state.get(obj, "x"), state.get(obj, "y"), state.get(obj, "z"), yaw]))
            elif n.startswith("wiper"):
                qw, qz = state.get(obj, "qw"), state.get(obj, "qz")
                qx, qy = state.get(obj, "qx"), state.get(obj, "qy")
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                self.wiper = np.array([state.get(obj, "x"), state.get(obj, "y"), state.get(obj, "z"), yaw])
                self.wiper_R = np.array([
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        base = np.array([state.get(robot, "pos_base_x"), state.get(robot, "pos_base_y"),
                         state.get(robot, "pos_base_rot")])
        arm = np.array([state.get(robot, "pos_arm_joint%d" % i) for i in range(1, 8)])
        grip = state.get(robot, "pos_gripper")
        return base, arm, grip, cubes, None

    def reset(self, state, info):
        self.wmoves = 0
        self.queue = []
        self.log = []
        self.stage = "pick"
        self.slots = make_slots()
        self.used = set()
        self.target_idx = None
        self.fails = {}
        self.tried = {}
        self.wiper_R = np.eye(3)

    # ---------------------------------------------------------------- planning
    def _base_for(self, p, th=THETA):
        return np.array([p[0] - HOLD_R * np.cos(th), p[1] - HOLD_R * np.sin(th), th])

    @staticmethod
    def _base_ok(b, m=0.03):
        x, y, th = b
        e = 0.29 * (abs(np.cos(th)) + abs(np.sin(th))) + m
        if x + e > 2.225 or y + e > 2.19:
            return False
        return GeneratedApproach._base_free(x, y, th, m)

    @staticmethod
    def _base_free(x, y, th, m=0.02):
        """True if the rotated square base footprint is clear of the island quadrant."""
        X, Y = 0.935 + m, 1.0 + m
        c, s_ = np.cos(th), np.sin(th)
        h = 0.29
        for u in (-1, 0, 1):
            for v in (-1, 0, 1):
                if u == 0 and v == 0:
                    continue
                px = x + h * (u * c - v * s_)
                py = y + h * (u * s_ + v * c)
                if px < X and py < Y:
                    return False
        dx, dy = X - x, Y - y
        if abs(dx * c + dy * s_) < h and abs(-dx * s_ + dy * c) < h:
            return False
        return True

    @staticmethod
    def _arm_hits(xy, th):
        d = np.array([np.cos(th), np.sin(th)])
        for s in (0.1, 0.25, 0.4, 0.5, 0.6):
            p = xy + s * d
            if 0.06 < p[0] < 0.945 and p[1] < 1.015:
                return True
            if p[0] > 2.20 or p[1] > 2.17:
                return True
        return False

    @staticmethod
    def _bad(xy, th):
        return GeneratedApproach._arm_hits(xy, th) or \
            not GeneratedApproach._base_free(xy[0], xy[1], th, 0.01)

    def _grasp_theta(self, i, cubes):
        c = cubes[i]
        obst = [cc[:2] for j, cc in enumerate(cubes) if j != i and np.hypot(*(cc[:2] - c[:2])) < 0.12]
        if self.wiper is not None:
            o, R = self.wiper[:3], self.wiper_R
            pts = [o + R @ np.array([s, 0, 0.02]) for s in np.linspace(-0.15, 0.15, 31)]
            pts += [o + R @ np.array([0, 0, s]) for s in np.linspace(0.04, 0.34, 31)]
            for p in pts:
                if p[2] < 0.25 and np.hypot(*(p[:2] - c[:2])) < 0.12:
                    obst.append(p[:2])
        wpts = None
        if self.wiper is not None:
            o, R = self.wiper[:3], self.wiper_R
            wpts = np.array([(o + R @ np.array([s, 0, 0.02]))[:2] for s in np.linspace(-0.15, 0.15, 7)])
        best, bestc, best_off = THETA, 1e9, np.zeros(2)
        cands = [wrap(c[3] + k * np.pi / 2 + np.pi / 2) for k in range(4)]
        cands += list(THETA + np.linspace(-1.2, 1.2, 13))
        for th in cands:
            dev = abs(wrap(th - THETA))
            if dev > 1.3:
                continue
            mis = abs(wrap(4 * (th - c[3])) / 4)
            u = np.array([-np.sin(th), np.cos(th)])
            f = np.array([np.cos(th), np.sin(th)])
            cost = 0.3 * dev + 3.0 * mis
            ob = 0.0
            side = set()
            for o in obst:
                r = o - c[:2]
                a, b = abs(r @ u), r @ f
                if a < 0.06:
                    if abs(b) < 0.013:
                        ob += 5.0
                    elif abs(b) < 0.035:
                        side.add(1 if b > 0 else -1)
                        ob += 0.5
            if len(side) == 2:
                ob += 5.0
            off = np.zeros(2)
            if len(side) == 1:
                off = -0.006 * next(iter(side)) * f
            cost += min(ob, 15.0)
            bb = self._base_for(c, th)
            if not self._base_ok(bb) or self._arm_hits(bb[:2], th):
                cost += 100.0
            if wpts is not None:
                rel = wpts - bb[:2]
                lx = np.abs(rel @ f)
                ly = np.abs(rel @ u)
                if np.any((lx < 0.34) & (ly < 0.34)):
                    cost += 8.0
            for pth in self.tried.get(i, []):
                if abs(wrap(th - pth)) < 0.2:
                    cost += 1.5
            if cost < bestc:
                best, bestc, best_off = th, cost, off
        self.tried.setdefault(i, []).append(best)
        self.grasp_off = best_off
        return best

    def _pick_slot(self, cubes, pinch, th):
        """Closest free drop slot reachable with base heading th (falls back to THETA)."""
        for heading in (th, THETA):
            best, bc = None, 1e9
            for i, s in enumerate(self.slots):
                if i in self.used:
                    continue
                if any(np.hypot(c[0] - s[0], c[1] - s[1]) < 0.04 for c in cubes):
                    continue
                bb = self._base_for(s, heading)
                if not self._base_ok(bb) or self._arm_hits(bb[:2], heading):
                    continue
                d = np.hypot(*(s - pinch))
                if d < bc:
                    best, bc = i, d
            if best is not None:
                return best, heading
        return None, THETA
    def _free_slot(self, cubes):
        for i, s in enumerate(self.slots):
            if i in self.used:
                continue
            if any(np.hypot(c[0] - s[0], c[1] - s[1]) < 0.035 for c in cubes):
                continue
            return i
        return None

    def _seg_ok(self, p0, p1, th):
        n = max(2, int(np.hypot(*(p1 - p0)) / 0.03))
        for t in np.linspace(0, 1, n):
            p = p0 + t * (p1 - p0)
            if self._bad(self._base_for(p, th)[:2], th):
                return False
        return True

    def _route(self, pinch, dest_pinch, th, **kw):
        """Targets moving the pinch to dest_pinch, routing around the island corner."""
        tg = []
        pinch = np.asarray(pinch, float)
        dest_pinch = np.asarray(dest_pinch, float)
        if not self._seg_ok(pinch, dest_pinch, th):
            cands = [np.array([1.0, dest_pinch[1]]), np.array([1.0, pinch[1]]),
                     np.array([max(pinch[0], 1.0), max(dest_pinch[1], 1.06)]),
                     np.array([1.03, 1.18])]
            best, bl = cands[-1], 1e9
            for w in cands:
                if self._seg_ok(pinch, w, th) and self._seg_ok(w, dest_pinch, th):
                    L = np.hypot(*(w - pinch)) + np.hypot(*(dest_pinch - w))
                    if L < bl:
                        best, bl = w, L
            kw2 = dict(kw)
            kw2["tol"] = 0.05
            kw2["tol_arm"] = 1.0
            kw2["name"] = kw.get("name", "") + "_wp"
            tg.append(Target(base=self._base_for(best, th), **kw2))
        tg.append(Target(base=self._base_for(dest_pinch, th), **kw))
        tg[-1].pinch = dest_pinch.copy()
        return tg

    def _wiper_plan(self, i, cubes, pinch):
        """Grab the wiper (handle or blade) and drag it away from cube i."""
        if self.wiper is None:
            return None
        o, R = self.wiper[:3], self.wiper_R
        c = cubes[i]
        allp = [o + R @ np.array([0, 0, s]) for s in np.linspace(0.04, 0.34, 16)]
        allp += [o + R @ np.array([s, 0, 0]) for s in np.linspace(-0.15, 0.15, 16)]
        if min(np.hypot(*(p[:2] - c[:2])) for p in allp) > 0.07:
            return None
        feats = []
        ax = R[:, 2]
        if abs(ax[2]) < 0.3:
            a2 = ax[:2] / (np.linalg.norm(ax[:2]) + 1e-9)
            for s in np.linspace(0.08, 0.30, 12):
                feats.append(((o + R @ np.array([0, 0, s]))[:2], a2, abs(s - 0.19) * 2, G_OPEN))
        bx = R[:, 0]
        if abs(bx[2]) < 0.3 and (abs(R[2, 1]) > 0.8 or abs(R[2, 2]) > 0.8):
            b2 = bx[:2] / (np.linalg.norm(bx[:2]) + 1e-9)
            for s in (-0.13, -0.11, -0.09, -0.07, 0.07, 0.09, 0.11, 0.13):
                feats.append(((o + R @ np.array([s, 0, 0]))[:2], b2, 0.3, 0.4))
        best = None
        for p, a2, pen, gopen in feats:
            dc = min(np.hypot(*(cc[:2] - p)) for cc in cubes)
            if dc < 0.05:
                continue
            for th in (np.arctan2(a2[1], a2[0]), np.arctan2(-a2[1], -a2[0])):
                dev = abs(wrap(th - THETA))
                bb = self._base_for(p, th)
                if dev > 1.6 or not self._base_ok(bb) or self._arm_hits(bb[:2], th):
                    continue
                for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    dv = np.array([np.cos(ang), np.sin(ang)])
                    for dist in (0.25, 0.15):
                        dest = p + dist * dv
                        bd = self._base_for(dest, th)
                        if not self._base_ok(bd) or self._arm_hits(bd[:2], th):
                            continue
                        gain = np.hypot(*(dest - c[:2])) - np.hypot(*(p - c[:2]))
                        if gain < 0.08:
                            continue
                        sc = dev + pen - gain * 4
                        if best is None or sc < best[0]:
                            best = (sc, p, th, dest, gopen)
                        break
        if best is None:
            return None
        _, p, th, dest, gopen = best
        q = self._route(pinch, p, th, arm=arm_q(Z_APPROACH), grip=gopen, tol=0.004, tol_arm=0.04,
                        max_steps=120, name="wapproach")
        q[-1].lower_z = Z_GRASP
        for tq in q:
            tq.far_z = Z_HIGH
        b = q[-1].base
        low = Target(base=b, arm=arm_q(Z_GRASP), grip=gopen, tol=0.004, tol_arm=0.03, max_steps=4, name="wlower")
        q += [low,
              Target(arm=arm_q(Z_GRASP), grip=1.0, min_steps=3, max_steps=3, name="wclose"),
              Target(arm=arm_q(Z_CARRY), grip=1.0, min_steps=2, max_steps=2, name="wlift"),
              Target(base=self._base_for(dest, th), arm=arm_q(Z_CARRY), grip=1.0, tol=0.02, tol_arm=0.5,
                     vbase=0.05, max_steps=30, name="wmove"),
              Target(arm=arm_q(Z_CARRY), grip=0.0, min_steps=2, max_steps=2, name="wdrop"),
              Target(arm=arm_q(Z_HIGH), grip=0.0, min_steps=3, max_steps=3, name="wup")]
        return q

    def _replan(self, base, arm, grip, cubes):
        off = fk_pinch(arm)[:2]
        c_, s_ = np.cos(base[2]), np.sin(base[2])
        pinch = base[:2] + np.array([c_ * off[0] - s_ * off[1], s_ * off[0] + c_ * off[1]])
        if self.stage == "carry":
            # did we lift a cube?
            held = [i for i, c in enumerate(cubes)
                    if c[2] > 0.016 and np.hypot(*(c[:2] - pinch)) < 0.05]
            if held:
                si, hd = self._pick_slot(cubes, pinch, base[2])
                if si is None:
                    si = len(self.used) % len(self.slots)
                self.used.add(si)
                s = self.slots[si]
                self.queue = self._route(pinch, s, hd, arm=arm_q(Z_CARRY), grip=1.0, tol=0.015,
                                         tol_arm=0.5, vbase=V_CARRY, max_steps=60, name="carry")
                self.queue.append(Target(arm=arm_q(Z_CARRY), grip=0.0,
                                         min_steps=2, max_steps=2, name="drop"))
                self.stage = "pick"
                return
            self.stage = "pick"
        # pick stage: nearest cube outside the goal
        todo = [i for i, c in enumerate(cubes) if not in_goal(c)]
        if not todo:
            self.queue = [Target(base=base, arm=arm_q(Z_CARRY), grip=0.0, max_steps=5, min_steps=5)]
            return
        def cost(i):
            c = cubes[i]
            return np.hypot(*(c[:2] - pinch)) + 0.3 * self.fails.get(i, 0)
        i = min(todo, key=cost)
        c = cubes[i]
        self.fails[i] = self.fails.get(i, 0) + 1
        if self.fails[i] >= 3 and self.wmoves < 4:
            wq = self._wiper_plan(i, cubes, pinch)
            if wq is not None:
                self.wmoves += 1
                self.queue = wq
                self.stage = "pick"
                return
        th = self._grasp_theta(i, cubes)
        goff = self.grasp_off
        q = self._route(pinch, c[:2] + goff, th, arm=arm_q(Z_APPROACH), grip=G_OPEN, tol=0.004, tol_arm=0.04,
                        max_steps=120, name="approach")
        q[-1].lower_z = Z_GRASP
        q[-1].track = i
        q[-1].track_off = goff
        for tq in q:
            tq.far_z = Z_HIGH
        b = q[-1].base
        low = Target(base=b, arm=arm_q(Z_GRASP), grip=G_OPEN, tol=0.004, tol_arm=0.03, max_steps=4, name="lower")
        low.pinch, low.track, low.track_off = c[:2] + goff, i, goff
        q += [
            low,
            Target(arm=arm_q(Z_GRASP), grip=1.0, min_steps=3, max_steps=3, name="close"),
            Target(arm=arm_q(Z_CARRY), grip=1.0, min_steps=2, max_steps=2, name="lift"),
        ]
        self.queue = q
        self.stage = "carry"

    # ------------------------------------------------------------------ action
    def _update_target(self, t, arm, cubes):
        if t.pinch is None or t.base is None:
            return
        if t.track is not None and t.track < len(cubes):
            t.pinch = cubes[t.track][:2] + t.track_off
        tarm = t.arm if t.lower_z is None else arm_q(t.lower_z)
        off = np.array([HOLD_R, 0.0])
        if tarm is not None and np.max(np.abs(wrap(tarm - arm))) < 0.12:
            off = fk_pinch(arm)[:2]
        c, s = np.cos(t.base[2]), np.sin(t.base[2])
        t.base[:2] = t.pinch - np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])

    def get_action(self, state):
        try:
            return self._get_action(state)
        except Exception:
            self.queue = []
            a = np.zeros(self.action_space.shape, dtype=np.float32)
            return a

    def _get_action(self, state):
        base, arm, grip, cubes, _ = self._read(state)
        for _ in range(4):
            if not self.queue:
                self._replan(base, arm, grip, cubes)
            t = self.queue[0]
            self._update_target(t, arm, cubes)
            done = t.steps >= t.max_steps
            if not done and t.steps >= t.min_steps:
                ok = True
                if t.base is not None:
                    ok &= np.hypot(*(t.base[:2] - base[:2])) < t.tol
                    ok &= abs(wrap(t.base[2] - base[2])) < 0.03
                if t.arm is not None:
                    tz = t.eff_arm(base) if t.lower_z is None else arm_q(t.lower_z)
                    ok &= np.max(np.abs(wrap(tz - arm))) < t.tol_arm
                done = ok
            if done:
                self.log.append((t.name, t.steps))
                self.queue.pop(0)
                continue
            break
        t = self.queue[0]
        t.steps += 1
        self._update_target(t, arm, cubes)
        act = np.zeros(11)
        if t.base is not None:
            d = t.base[:2] - base[:2]
            act[0:2] = np.clip(d, -t.vbase, t.vbase)
            act[2] = np.clip(wrap(t.base[2] - base[2]), -t.vrot, t.vrot)
            if not self._bad(base[:2], base[2]):
                dx, dy, dr = act[0], act[1], act[2]
                if self._bad(base[:2] + act[0:2], base[2] + dr):
                    best, bv = (0.0, 0.0, 0.0), 0.0
                    for alt in ((dx, 0.0, dr), (0.0, dy, dr), (dx, dy, 0.0), (dx, 0.0, 0.0),
                                (0.0, dy, 0.0), (0.0, 0.0, dr), (dx / 2, dy / 2, dr / 2),
                                (dx, dy, dr * 0.2), (0.0, 0.0, dr * 0.5), (0.0, 0.0, dr * 0.2)):
                        if self._bad(base[:2] + np.array(alt[:2]), base[2] + alt[2]):
                            continue
                        v = abs(alt[0]) + abs(alt[1]) + 1.0 * abs(alt[2])
                        if v > bv:
                            best, bv = alt, v
                    if bv < 0.3 * (abs(dx) + abs(dy) + abs(dr)):
                        for esc in ((0.0, 0.1, 0.0), (0.1, 0.0, 0.0), (0.07, 0.07, 0.0)):
                            if not self._bad(base[:2] + np.array(esc[:2]), base[2]):
                                best = esc
                                break
                    act[0:3] = best
        if t.arm is not None:
            tarm = t.eff_arm(base)
            if t.lower_z is not None and t.base is not None and \
                    np.hypot(*(t.base[:2] - base[:2])) < 0.006 and abs(wrap(t.base[2] - base[2])) < 0.02:
                t.lowering = True
            if t.lower_z is not None and t.lowering:
                tarm = arm_q(t.lower_z)
            d = wrap(tarm - arm)
            mx = np.max(np.abs(d))
            if mx > t.varm:
                d = d * (t.varm / mx)
            act[3:10] = d
        act[10] = grip if t.grip is None else t.grip
        lo, hi = self.action_space.low, self.action_space.high
        return np.clip(act, lo, hi).astype(np.float32)
