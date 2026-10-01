"""SweepIntoDrawer3D approach: open island drawer s1c1 by hooking its top edge,
then pick-and-place each cube into the open drawer."""
import numpy as np
from kin import fk_world, ik, R_down

Q_HOME = np.array([0, -0.349, 3.142, -2.548, 0, -0.873, 1.571])
# sweep pose (tilted pusher), base-relative with base heading pi; tool at (-0.65, 0.68, 0.455)
Q_SW = np.array([0.66444, 1.15437, 3.55502, -1.21463, 0.59576, 0.13987, 1.59426])
DRAWER_IDX = 107          # kitchen_island_drawer_s1c1
GAP_X = 0.917             # gap between counter edge and drawer front
HOOK_Z = 0.285
TABLE_Z = 0.45


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    def reset(self, state, info):
        self.obs = np.asarray(state, dtype=float)
        self.gen = self._plan()
        self.grip = 0.0
        self.last = np.zeros(11)
        self.fails = {}
        self.log = []
        self.t = 0
        self.mode = getattr(self, 'mode', 'sweep')
        self.pgrip = getattr(self, 'pgrip', 1.0)
        self.syaw = getattr(self, 'syaw', np.pi / 2)
        self.sz = getattr(self, 'sz', 0.455)
        self.v = getattr(self, 'v', 0.06)
        self.off = getattr(self, 'off', (-0.65, 0.68))
        self.kq = getattr(self, 'kq', 2.0)
        self.liftz = getattr(self, 'liftz', 0.40)
        self.srot = getattr(self, 'srot', None)
        self.early = getattr(self, 'early', True)
        self.holdvia = getattr(self, 'holdvia', False)
        self.dds = getattr(self, 'dds', 0.04)
        self.dtol = getattr(self, 'dtol', 0.04)
        self.xend = getattr(self, 'xend', 0.99)
        self.BP = np.array([1.25, -0.70, np.pi])

    # ------------------------------------------------------------------ utils
    @property
    def base(self):
        return self.obs[125:128].copy()

    @property
    def q(self):
        return self.obs[128:135].copy()

    def cube(self, i):
        return self.obs[16 * i:16 * i + 3].copy()

    def cube_yaw(self, i):
        qw, qz = self.obs[16 * i + 3], self.obs[16 * i + 6]
        return 2 * np.arctan2(qz, qw)

    def _act(self, q_des=None, base_des=None, grip=None):
        a = np.zeros(11)
        if q_des is not None:
            a[3:10] = np.clip(self.kq * (q_des - self.q), -0.1, 0.1)
        if base_des is not None:
            d = np.array(base_des, float) - self.base
            d[2] = wrap(d[2])
            a[0:3] = np.clip(d, -0.1, 0.1)
        if grip is not None:
            self.grip = grip
        a[10] = self.grip
        return a

    def arm_to(self, q_des, base_des=None, tol=0.01, maxn=80, grip=None):
        for _ in range(maxn):
            yield self._act(q_des, base_des, grip)
            ok = np.max(np.abs(q_des - self.q)) < tol
            if base_des is not None:
                d = np.array(base_des) - self.base
                d[2] = wrap(d[2])
                ok = ok and np.max(np.abs(d[:2])) < 0.02 and abs(d[2]) < 0.05
            if ok:
                return

    def hold(self, n, grip=None, base_des=None):
        q = self.q
        for _ in range(n):
            yield self._act(q, base_des, grip)

    def solve(self, base, p, Rd, q0=None):
        q, e = ik(base, self.q if q0 is None else q0, np.asarray(p, float), Rd, iters=150)
        return q

    def line(self, base, p0, p1, Rd, ds=0.02, tol_mid=0.03, tol_end=0.01, maxn=30):
        """Move tool along straight line (world) with base fixed at `base`."""
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / ds)))
        qc = self.q
        for k in range(1, n + 1):
            p = p0 + (p1 - p0) * k / n
            qc = self.solve(base, p, Rd, qc)
            yield from self.arm_to(qc, base, tol=tol_end if k == n else tol_mid, maxn=maxn)

    def tool(self):
        return fk_world(self.base, self.q)[:3, 3]

    def sweep(self, B):
        Rw = R_down(np.pi / 2)
        w = self.obs[147:150].copy()
        g = np.array([w[0] - 0.01, w[1], 0.0])
        self.grip = 0.0
        yield from self.line(B, self.tool(), [g[0], g[1], 0.54], Rw, ds=0.05, tol_mid=0.05,
                             tol_end=0.02)
        yield from self.line(B, [g[0], g[1], 0.54], [g[0], g[1], 0.462], Rw, ds=0.04,
                             tol_end=0.01, maxn=20)
        yield from self.hold(5, grip=1.0)
        C = np.array([self.cube(i) for i in range(5)])
        on = C[:, 2] > 0.44
        C = C[on]
        bx = C[:, 0].min() - 0.055
        cy = 0.5 * (C[:, 1].min() + C[:, 1].max())
        off = 0.09  # blade is ~0.09 in -x of the grasp point
        yield from self.line(B, self.tool(), [g[0], g[1], 0.50], Rw, ds=0.04, tol_end=0.03)
        yield from self.line(B, self.tool(), [bx + off, cy, 0.50], Rw, ds=0.04, tol_end=0.02)
        yield from self.line(B, self.tool(), [bx + off, cy, 0.466], Rw, ds=0.04, tol_end=0.01,
                             maxn=15)
        yield from self.line(B, self.tool(), [0.97 + off, cy, 0.466], Rw, ds=0.03,
                             tol_mid=0.03, tol_end=0.02)
        yield from self.hold(3)

    def remaining(self):
        return [i for i in range(5) if self.cube(i)[2] > 0.44 and self.cube(i)[0] < 0.92]

    def choose(self, cands):
        """Pick cube + grasp yaw with the most finger clearance (prefer near tool)."""
        best = None
        allc = [self.cube(j) for j in range(5)]
        tp = self.tool()
        for i in cands:
            c = allc[i]
            yaw = self.cube_yaw(i)
            for k in range(2):
                a = yaw + k * np.pi / 2
                d = np.array([np.cos(a), np.sin(a)])
                clear = 1.0
                for j in range(5):
                    if j == i or allc[j][2] < 0.4:
                        continue
                    for sgn in (1, -1):
                        f = c[:2] + sgn * 0.055 * d
                        clear = min(clear, np.linalg.norm(allc[j][:2] - f))
                gyy = min([a, a + np.pi, a - np.pi], key=lambda t: abs(wrap(t - np.pi)))
                _, e = ik(self.BP, Q_HOME, np.array([c[0], c[1], 0.462]), R_down(gyy), iters=150)
                if e > 0.005:
                    continue
                score = min(clear, 0.06) - 0.05 * np.linalg.norm(c[:2] - tp[:2])
                score -= 0.03 * self.fails.get((i, k), 0)
                if best is None or score > best[0]:
                    best = (score, i, a, k)
        if best is None:
            i = cands[0]
            return i, self.cube_yaw(i), 0
        return best[1], best[2], best[3]

    def push(self, BP, i, yaw=np.pi / 2, back=0.045, z=0.48, xend=0.99):
        c = self.cube(i)
        R = R_down(yaw)
        self.grip = 1.0
        s = [c[0] - back, c[1], 0.53]
        yield from self.line(BP, self.tool(), s, R, ds=0.06, tol_mid=0.06, tol_end=0.02)
        yield from self.line(BP, s, [s[0], s[1], z], R, ds=0.04, tol_end=0.01, maxn=15)
        yield from self.line(BP, self.tool(), [xend, s[1], z], R, ds=0.03, tol_mid=0.03,
                             tol_end=0.02, maxn=15)
        p = self.tool()
        yield from self.line(BP, p, [p[0] - 0.03, p[1], 0.53], R, ds=0.05, tol_end=0.03)
        return True

    def lane(self, cands, hw=0.05):
        C = np.array([self.cube(j) for j in cands])
        best = None
        for i in range(len(C)):
            yl = C[i, 1]
            m = np.abs(C[:, 1] - yl) < hw
            xs = C[m, 0].min() - 0.045
            score = m.sum() + 0.1 * C[i, 0]
            if best is None or score > best[0]:
                best = (score, yl, xs)
        return best[1], best[2]

    def push_lane(self, BP, yl, xs, yaw=np.pi / 2, z=0.472, xend=0.99, v=0.03):
        R = R_down(yaw)
        self.grip = self.pgrip
        s = [xs, yl, 0.53]
        yield from self.line(BP, self.tool(), s, R, ds=0.06, tol_mid=0.06, tol_end=0.02)
        yield from self.line(BP, s, [xs, yl, z], R, ds=0.04, tol_end=0.01, maxn=15)
        qp = self.solve(BP, [xs, yl, z], R, self.q)
        b = BP.copy()
        n = int(np.ceil((xend - xs) / v))
        for k in range(n):
            b[0] = BP[0] + min((k + 1) * v, xend - xs)
            yield self._act(qp, b)
        yield from self.arm_to(qp, b, tol=0.02, maxn=5)
        p = self.tool()
        qu = self.solve(b, [p[0], p[1], 0.53], R, qp)
        yield from self.arm_to(qu, BP, tol=0.05, maxn=20)
        return True

    def pick_place(self, BP, i, gy):
        c = self.cube(i)
        # grasp yaw: equivalent modulo pi, choose closest to pi (arm heading) for wrist
        gy = min([gy, gy + np.pi, gy - np.pi], key=lambda a: abs(wrap(a - BP[2])))
        Rg = R_down(gy)
        self.grip = 0.0
        yield from self.line(BP, self.tool(), [c[0], c[1], 0.53], Rg, ds=0.06, tol_mid=0.06,
                             tol_end=0.02)
        c = self.cube(i)
        yield from self.line(BP, self.tool(), [c[0], c[1], 0.462], Rg, ds=0.04, tol_end=0.01,
                             maxn=15)
        yield from self.hold(4, grip=1.0)
        yield from self.line(BP, self.tool(), [c[0], c[1], 0.52], Rg, ds=0.04, tol_mid=0.04,
                             tol_end=0.02, maxn=15)
        if self.cube(i)[2] < 0.485:      # failed grasp
            self.grip = 0.0
            return False
        # transport with base
        shift = np.clip(1.10 - c[0], 0.25, 0.45)
        BD = BP + np.array([shift, 0, 0])
        qh = self.q
        yield from self.arm_to(qh, BD, tol=0.02, maxn=20)
        yield from self.hold(3, grip=0.0, base_des=BD)
        yield from self.arm_to(qh, BP, tol=0.02, maxn=20)
        return True

    def base_to(self, q, b, tol=0.02, maxn=30):
        b = np.array(b, float)
        for _ in range(maxn):
            yield self._act(q, b)
            d = b - self.base
            if np.max(np.abs(d[:2])) < tol and abs(wrap(d[2])) < 0.05:
                return

    def lower_to(self, qlo, b, zok=0.483, maxn=15):
        b = np.array(b, float)
        for _ in range(maxn):
            yield self._act(qlo, b)
            d = b - self.base
            if self.tool()[2] < zok and np.max(np.abs(d[:2])) < 0.015:
                return

    def sweep_all(self, B1):
        OFF = np.array(self.off)
        th = np.array([0, 0, np.pi])
        R = fk_world(th, Q_SW)[:3, :3] if self.srot is None else self.srot
        qlo = self.solve(th, [OFF[0], OFF[1], self.sz], R, Q_SW)
        qhi = self.solve(th, [OFF[0], OFF[1], 0.53], R, qlo)
        self.grip = 1.0
        # go to via point beside drawer, arm to hi pose
        if self.holdvia:
            yield from self.base_to(self.q, [max(B1[0], 1.75), -0.75, np.pi], tol=0.05, maxn=20)
        else:
            yield from self.arm_to(qhi, [max(B1[0], 1.75), -0.75, np.pi], tol=0.05, maxn=40)
        cands = self.remaining()
        if not cands:
            return
        yl, xs = self.lane(cands)
        yield from self.arm_to(qhi, [xs - OFF[0], yl - OFF[1], np.pi], tol=0.03, maxn=40)
        yield from self.lower_to(qlo, [xs - OFF[0], yl - OFF[1], np.pi])
        for it in range(25):
            # push along lane
            b = self.base.copy(); b[2] = np.pi
            x0 = b[0]
            xe = self.xend - OFF[0]
            lane_ids = [j for j in self.remaining() if abs(self.cube(j)[1] - yl) < 0.05]
            while b[0] < xe - 1e-6:
                b[0] = min(b[0] + self.v, xe)
                yield self._act(qlo, b)
                if self.early:
                    px = self.base[0] + OFF[0]
                    ahead = [j for j in lane_ids if self.cube(j)[2] > 0.44 and
                             self.cube(j)[0] < 0.92 and self.cube(j)[0] > px - 0.03]
                    if not ahead:
                        break
            if not self.early:
                yield from self.base_to(qlo, b, maxn=5)
            self.log.append((self.t, round(yl, 3), round(xs, 3),
                             np.array([self.cube(j) for j in range(5)])[:, :2].round(3).tolist()))
            cands = self.remaining()
            if not cands:
                break
            yl, xs = self.lane(cands)
            xb = min(self.cube(j)[0] for j in cands) - 0.05
            xs = min(xs, xb)
            if self.early:
                for _ in range(10):
                    if self.tool()[2] > 0.515:
                        break
                    yield self._act(qhi, self.base)
            else:
                yield from self.arm_to(qhi, None, tol=0.03, maxn=10)
            yield from self.arm_to(qhi, [xs - OFF[0], yl - OFF[1], np.pi], tol=0.03, maxn=20)
            yield from self.lower_to(qlo, [xs - OFF[0], yl - OFF[1], np.pi])

    # ------------------------------------------------------------------ plan
    def _plan(self):
        # ---------------- open drawer s1c1 by hooking behind its top edge
        yh = 0.0
        B0 = np.array([1.45, yh, np.pi])
        Rh = R_down(np.pi / 2)
        q_pre = self.solve(B0, [GAP_X, yh, 0.52], Rh, Q_HOME)
        yield from self.arm_to(q_pre, B0, tol=0.02, grip=1.0)
        yield from self.line(B0, self.tool(), [GAP_X, yh, 0.52], Rh, ds=0.03)
        zh = []
        for _ in self.line(B0, [GAP_X, yh, 0.52], [GAP_X, yh, HOOK_Z], Rh, ds=self.dds,
                           tol_mid=self.dtol, maxn=10):
            yield _
            zh.append(self.tool()[2])
            if len(zh) > 4 and zh[-1] < 0.34 and zh[-4] - zh[-1] < 0.003:
                break
        # pull with base, arm fixed
        qh = self.q
        b = B0.copy()
        for _ in range(40):
            if self.obs[DRAWER_IDX] > 0.33:
                break
            b[0] = min(b[0] + 0.02, B0[0] + 0.42)
            yield self._act(qh, b)
        B1 = self.base.copy(); B1[2] = np.pi
        # lift out of the gap
        p = self.tool()
        for _ in self.line(B1, p, [p[0], p[1], self.liftz + 0.03], Rh, ds=0.05, tol_mid=0.04):
            yield _
            if self.tool()[2] > self.liftz:
                break
        if self.mode == 'sweep':
            yield from self.sweep_all(B1)
            while True:
                yield from self.hold(10)
        BP = self.BP
        yield from self.arm_to(self.q, [max(B1[0], 1.75), BP[1], np.pi], tol=0.05, maxn=30)
        yield from self.arm_to(self.q, BP, tol=0.05, maxn=30)
        # ---------------- pick & place cubes (base does the transport)
        for it in range(12):
            cands = self.remaining()
            if not cands:
                break
            i, gy, k = self.choose(cands)
            if self.mode == 'push':
                yl, xs = self.lane(cands)
                ok = yield from self.push_lane(BP, yl, xs)
            else:
                ok = yield from self.pick_place(BP, i, gy)
            if not ok:
                self.fails[(i, k)] = self.fails.get((i, k), 0) + 1
        while True:
            yield from self.hold(10)

    def get_action(self, state):
        self.obs = np.asarray(state, dtype=float)
        self.t += 1
        try:
            a = next(self.gen)
        except StopIteration:
            a = self._act(self.q)
        a = np.clip(np.asarray(a, dtype=np.float32), self.action_space.low, self.action_space.high)
        return a
