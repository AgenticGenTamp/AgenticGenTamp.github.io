"""Simulation-based push-primitive MPC for DynPushT2D.

At each decision point we enumerate push primitives (contact point on the T
boundary + push direction), simulate each in a private copy of the env, and
pick the one with best progress per step (including navigation steps).
The robot navigates to the pre-contact point via a visibility-graph path that
avoids the (inflated) T-block.
"""
import heapq
import math
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

STEP = 0.049
R_ROBOT = 0.1
WALL_LO, WALL_HI = 0.12, 4.88
POS_TOL = 0.03
ANG_TOL = np.deg2rad(8.0)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self._sim = None
        self.plan = []
        self.expected_block = None
        import os
        self.alpha = float(os.environ.get("ALPHA", 0.5))
        self.denom = float(os.environ.get("DENOM", 4))
        self.angs = (-0.5, 0.0, 0.5) if os.environ.get("ANGS", "5") == "3" else (-0.6, -0.3, 0.0, 0.3, 0.6)
        self.t0 = time.time()
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ sim
    def _get_sim(self):
        if self._sim is None:
            from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
            self._sim = KinderGeom2DEnv("kinder/DynPushT2D-t1-v0")
            self._sim.reset(seed=0)
        return self._sim

    # ------------------------------------------------------------ geometry
    def _rects(self, state):
        """Return list of (center, theta, half_extents) for crossbar & stem."""
        x, y, th = state[0], state[1], state[2]
        w, lh, lv = state[12], state[13], state[14]
        R = rot(th)
        p = np.array([x, y])
        c1 = p + R @ np.array([0.0, -w / 2])
        c2 = p + R @ np.array([0.0, -w - lv / 2])
        return [(c1, th, np.array([lh / 2, w / 2])), (c2, th, np.array([w / 2, lv / 2]))]

    @staticmethod
    def _inside(pt, rects, infl):
        for c, th, he in rects:
            q = rot(-th) @ (pt - c)
            if abs(q[0]) < he[0] + infl and abs(q[1]) < he[1] + infl:
                return True
        return False

    @staticmethod
    def _seg_hits(a, b, rects, infl):
        for c, th, he in rects:
            Rt = rot(-th)
            p0 = Rt @ (a - c)
            d = Rt @ (b - a)
            t0, t1 = 0.0, 1.0
            ok = True
            for i in range(2):
                lo, hi = -he[i] - infl, he[i] + infl
                if abs(d[i]) < 1e-12:
                    if p0[i] <= lo or p0[i] >= hi:
                        ok = False
                        break
                else:
                    ta = (lo - p0[i]) / d[i]
                    tb = (hi - p0[i]) / d[i]
                    if ta > tb:
                        ta, tb = tb, ta
                    t0 = max(t0, ta)
                    t1 = min(t1, tb)
                    if t0 >= t1:
                        ok = False
                        break
            if ok and t1 - t0 > 1e-9:
                return True
        return False

    def _escape(self, pt, rects, infl):
        """Move pt out of inflated rects (push along outward direction)."""
        p = pt.copy()
        for _ in range(4):
            moved = False
            for c, th, he in rects:
                R = rot(th)
                q = R.T @ (p - c)
                if abs(q[0]) < he[0] + infl and abs(q[1]) < he[1] + infl:
                    # nearest exit along axis
                    dx = he[0] + infl - abs(q[0])
                    dy = he[1] + infl - abs(q[1])
                    if dx < dy:
                        q[0] = math.copysign(he[0] + infl + 0.01, q[0] if q[0] != 0 else 1)
                    else:
                        q[1] = math.copysign(he[1] + infl + 0.01, q[1] if q[1] != 0 else 1)
                    p = c + R @ q
                    moved = True
            if not moved:
                break
        return p

    def _plan_path(self, start, goal, rects):
        """Visibility graph shortest path from start to goal avoiding rects."""
        infl = R_ROBOT + 0.015
        pts = [start]
        s = start
        if self._inside(start, rects, infl):
            s = self._escape(start, rects, infl + 0.005)
            s = np.clip(s, WALL_LO, WALL_HI)
            pts.append(s)
        if not self._seg_hits(s, goal, rects, infl):
            return pts + [goal]
        nodes = [s, goal]
        vi = infl + 0.04
        for c, th, he in rects:
            R = rot(th)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    v = c + R @ np.array([sx * (he[0] + vi), sy * (he[1] + vi)])
                    if WALL_LO <= v[0] <= WALL_HI and WALL_LO <= v[1] <= WALL_HI \
                            and not self._inside(v, rects, infl):
                        nodes.append(v)
        n = len(nodes)
        dist = [math.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, i = heapq.heappop(pq)
            if d > dist[i]:
                continue
            if i == 1:
                break
            for j in range(n):
                if j == i:
                    continue
                nd = d + float(np.linalg.norm(nodes[j] - nodes[i]))
                if nd < dist[j] and not self._seg_hits(nodes[i], nodes[j], rects, infl):
                    dist[j] = nd
                    prev[j] = i
                    heapq.heappush(pq, (nd, j))
        if dist[1] == math.inf:
            return None
        path = []
        k = 1
        while k != -1:
            path.append(nodes[k])
            k = prev[k]
        path.reverse()
        return pts[:-1] + path if len(pts) > 1 else path

    @staticmethod
    def _path_actions(path):
        acts = []
        cur = path[0].copy()
        for wp in path[1:]:
            while True:
                d = wp - cur
                n = float(np.linalg.norm(d))
                if n < 1e-6:
                    break
                if n <= STEP:
                    acts.append(d.copy())
                    cur = wp.copy()
                    break
                a = d / n * STEP
                acts.append(a)
                cur = cur + a
        return acts

    # ------------------------------------------------------------- goal/cost
    def _goal_angle(self, state):
        g = state[31]
        lo, hi = g - ANG_TOL, g + ANG_TOL
        lo, hi = max(lo, -np.pi), min(hi, np.pi)
        return 0.5 * (lo + hi)

    def _cost(self, x, y, th, state):
        gx, gy = state[29], state[30]
        gth = self._goal_angle(state)
        dp = math.hypot(x - gx, y - gy)
        da = abs(wrap(th - gth))
        return dp + self.alpha * da

    def _success(self, x, y, th, state):
        return (abs(x - state[29]) < POS_TOL * 0.8 and abs(y - state[30]) < POS_TOL * 0.8
                and abs(th - state[31]) < ANG_TOL * 0.8)

    # ------------------------------------------------------------ primitives
    def _contacts(self, state):
        w, lh, lv = state[12], state[13], state[14]
        out = []  # (local point, local outward normal)

        def edge(p0, p1, nrm, k):
            for t in np.linspace(0, 1, k + 2)[1:-1]:
                out.append((p0 + t * (p1 - p0), np.array(nrm)))

        edge(np.array([-lh / 2, 0]), np.array([lh / 2, 0]), (0, 1), 7)  # top
        edge(np.array([-lh / 2, -w]), np.array([-w / 2, -w]), (0, -1), 3)  # under left
        edge(np.array([w / 2, -w]), np.array([lh / 2, -w]), (0, -1), 3)  # under right
        edge(np.array([-lh / 2, -w]), np.array([-lh / 2, 0]), (-1, 0), 1)
        edge(np.array([lh / 2, -w]), np.array([lh / 2, 0]), (1, 0), 1)
        edge(np.array([-w / 2, -w - lv]), np.array([-w / 2, -w]), (-1, 0), 5)
        edge(np.array([w / 2, -w - lv]), np.array([w / 2, -w]), (1, 0), 5)
        edge(np.array([-w / 2, -w - lv]), np.array([w / 2, -w - lv]), (0, -1), 1)
        return out

    def _decide(self, state):
        sim = self._get_sim()
        rects = self._rects(state)
        x, y, th = state[0], state[1], state[2]
        R = rot(th)
        p = np.array([x, y])
        robot = np.array([state[16], state[17]])
        c0 = self._cost(x, y, th, state)
        near = c0 < 0.25
        step_sizes = [STEP] if c0 > 0.4 else ([STEP, 0.02] if c0 > 0.12 else [0.02, 0.01])
        kmax = 14 if not near else 8
        best = None
        cands = []
        for lp, ln in self._contacts(state):
            cp = p + R @ lp
            nw = R @ ln
            pre = cp + nw * (R_ROBOT + 0.03)
            if not (WALL_LO <= pre[0] <= WALL_HI and WALL_LO <= pre[1] <= WALL_HI):
                continue
            if self._inside(pre, rects, R_ROBOT + 0.01):
                continue
            path = self._plan_path(robot, pre, rects)
            if path is None:
                continue
            plen = sum(float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1))
            nav_steps = math.ceil(plen / STEP - 1e-6)
            for ang in self.angs:
                d = rot(ang) @ (-nw)
                cands.append((pre, path, nav_steps, d))
        elapsed = time.time() - self.t0
        budget = max(0.3, min(5.0, (45.0 - elapsed) / 4.0))
        tstart = time.time()
        order = self.rng.permutation(len(cands))
        for ci in order:
            if time.time() - tstart > budget and best is not None:
                break
            pre, path, nav_steps, d = cands[ci]
            for ss in step_sizes:
                s = state.copy()
                s[16], s[17] = pre[0], pre[1]
                s[19:22] = 0
                sim.set_state(s)
                a = (d * ss).astype(np.float64)
                moved = False
                for k in range(1, kmax + 1):
                    o, _, _, _, _ = sim.step(a)
                    bx, by, bth = o[0], o[1], o[2]
                    if abs(bx - x) + abs(by - y) + abs(wrap(bth - th)) > 1e-4:
                        moved = True
                    if not moved:
                        continue
                    if o[16] < WALL_LO - 0.02 or o[16] > WALL_HI + 0.02 or \
                            o[17] < WALL_LO - 0.02 or o[17] > WALL_HI + 0.02:
                        break
                    c = self._cost(bx, by, bth, state)
                    if c > c0 + 0.05 and k >= 4:
                        break
                    total = nav_steps + k + 1
                    if self._success(bx, by, bth, state):
                        score = 1e6 - total
                    else:
                        score = (c0 - c) / (total + self.denom)
                    if best is None or score > best[0]:
                        best = (score, path, d * ss, k, (bx, by, bth))
                    if self._success(bx, by, bth, state):
                        break
        return best

    # ------------------------------------------------------------- interface
    def reset(self, state, info=None):
        self.plan = []
        self.expected_block = None
        self.t0 = time.time()
        self.rng = np.random.default_rng(0)

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float64)
        if self.plan and self.expected_block is not None and self.plan[0][1] == "nav":
            eb = self.expected_block
            if abs(state[0] - eb[0]) + abs(state[1] - eb[1]) + abs(wrap(state[2] - eb[2])) > 0.02:
                self.plan = []
        if not self.plan:
            best = self._decide(state)
            if best is None or best[0] <= 0:
                # fallback: tiny random jiggle
                a = np.random.uniform(-0.03, 0.03, size=2)
                return a.astype(np.float32)
            _, path, push, k, _ = best
            robot = np.array([state[16], state[17]])
            path = [robot] + list(path[1:])
            nav = self._path_actions(path)
            self.plan = [(a, "nav") for a in nav] + [(push.copy(), "push") for _ in range(k)]
            self.expected_block = (state[0], state[1], state[2])
        a, _ = self.plan.pop(0)
        a = np.clip(a, -0.0499, 0.0499)
        return a.astype(np.float32)
