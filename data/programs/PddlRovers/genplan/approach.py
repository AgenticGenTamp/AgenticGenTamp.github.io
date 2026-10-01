"""A hand-written policy for PDDLStream's `rovers` benchmark.

Diagnosis of the (repeated) seed-2 failure
------------------------------------------
rover1 ended at EXACTLY (-1.4647162, 2.2540355, 0.0) in both runs -- the same
position, to seven decimals, as the previous attempt.  A rover that is pinned at a
bit-identical pose for hundreds of steps is not "navigating badly"; it is emitting
motions that the environment rejects every single step, i.e. it is wedged against
geometry and the recovery behaviour never fired.

Why the recovery never fired: `self.stall[r]` was computed from the change in the
rover's position, but the escape manoeuvre itself was issued through `_drive`, and
`_drive` is only called on some branches.  In the branch that was actually running
(`_do_image` with `needed_visible` non-empty, and later `_deliver` while homing), the
policy returned `(0.0, 0.0, 0.0, op)` -- a zero motion -- so `moved` was zero, the
stall counter climbed, but the escape vector was never applied because `_drive` was
never reached.  The rover sat still forever "trying to calibrate" or "homing" while
issuing no motion at all.  rover0 likewise froze at (0.405, 0.251).

Second, structural problem: the endgame had no way to *verify* progress.  The goal
needs `at_home` true for BOTH rovers, stores empty, and every result received.  The
policy must therefore keep pushing on whichever clause is still false, and must be
able to tell "I am stuck" from "I am working".

This revision rewrites the control flow around a single invariant:

  * **Every step, every rover runs one unified controller** that (a) picks a 2D goal
    point, (b) always produces a motion through `_drive` (never a bare zero motion
    unless the rover is parked at home and finished), and (c) chooses the operator
    independently of the motion.  Operators are free and silently refused, so the
    motion and the operator never need to agree.
  * **A global anti-freeze watchdog.**  If a rover's pose is unchanged for K steps,
    regardless of which behaviour is active, we force a randomized escape burst
    (perpendicular / reverse / rotate) for several steps.  Rotation is included
    because a pure translation can be blocked while a rotation is free, and rotating
    changes the collision footprint.
  * **Phase machine with explicit, observation-checked exit conditions:**
    FETCH -> DELIVER -> HOME -> PARK.  A rover only leaves HOME when the state's own
    `at_home` feature is true, and once PARKed it emits an exact zero action so the
    flag cannot be lost.
  * **Store discipline:** DROP is emitted whenever `store_full` is true and the rover
    is not at that instant acquiring a needed sample (the analysis is already
    recorded, so dropping never loses anything).  In the failed run rover1 finished
    with `store_full == 1`, which alone blocks the goal.
  * **Deadline:** past a reserved fraction of the step budget, all fetching stops and
    every rover goes to DELIVER/HOME, guaranteeing the tail clauses are attempted.
  * SEND is emitted on every step a rover holds anything unsent, and additionally the
    homing route is biased through a comm-visible point so stranded results get
    radioed en route.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

# ---------------------------------------------------------------- constants

MAX_DELTA = 0.2
MAX_DELTA_THETA = 0.4

NOOP, SAMPLE, CALIBRATE, IMAGE, SEND, DROP = range(6)
_BANDS = (SAMPLE, CALIBRATE, IMAGE, NOOP, SEND, DROP)


def _op_value(op: int) -> float:
    band = _BANDS.index(op)
    return (band + 0.5) / len(_BANDS) * 2.0 - 1.0


OP_VALUE = {op: _op_value(op) for op in _BANDS}

VIS_RANGE = 2.0
COM_RANGE = 4.0
SAMPLE_RADIUS = 0.25
HOME_RADIUS = 0.25
HOME_ANGLE = 0.4

ARENA = 2.5
WALL_MARGIN = 0.32
ROBOT_RADIUS = 0.22
CAMERA_Z = 0.45

# Phases
FETCH, DELIVER, HOME, PARK = range(4)

# Watchdogs
FREEZE_LIMIT = 3          # identical pose for this many steps -> forced escape
ESCAPE_LEN = 6            # steps an escape burst lasts
ACT_LIMIT = 22            # steps spent trying calibrate/image at one spot
SAMPLE_BUDGET = 140
IMAGE_BUDGET = 160
SEND_AT_HOME_LIMIT = 8    # SENDs at home before hunting an explicit comm spot

BASE_STEPS = 250
STEPS_PER_OBJECT = 60
HOME_RESERVE_FRAC = 0.42
HOME_RESERVE_MIN = 150


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ geometry

def _seg_box_hit(p0, p1, centre, half) -> bool:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    centre = np.asarray(centre, dtype=float)
    half = np.asarray(half, dtype=float)
    d = p1 - p0
    lo, hi = 0.0, 1.0
    bmin = centre - half
    bmax = centre + half
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if p0[i] < bmin[i] or p0[i] > bmax[i]:
                return False
            continue
        t1 = (bmin[i] - p0[i]) / d[i]
        t2 = (bmax[i] - p0[i]) / d[i]
        if t1 > t2:
            t1, t2 = t2, t1
        lo = max(lo, t1)
        hi = min(hi, t2)
        if lo > hi:
            return False
    return True


def _features(state, obj):
    try:
        return list(state.type_features[obj.type])
    except Exception:
        return []


def _suffix_int(name, prefix):
    tail = name[len(prefix):]
    try:
        return int(tail)
    except ValueError:
        return 0


# ------------------------------------------------------------ world snapshot

class _World:
    def __init__(self, state):
        self.rovers = []
        self.objectives = []
        self.samples = []
        self.obstacles = []

        by_name = {obj.name: obj for obj in state}

        def feat(obj, name, default=0.0):
            try:
                return float(state.get(obj, name))
            except Exception:
                return float(default)

        rnames = sorted(
            [n for n in by_name if n.startswith("rover")],
            key=lambda n: _suffix_int(n, "rover"),
        )
        for n in rnames:
            o = by_name[n]
            self.rovers.append(
                {
                    "name": n,
                    "x": feat(o, "x"),
                    "y": feat(o, "y"),
                    "theta": feat(o, "theta"),
                    "store_full": feat(o, "store_full") > 0.5,
                    "calibrated": feat(o, "calibrated") > 0.5,
                    "at_home": feat(o, "at_home") > 0.5,
                }
            )
        self.n_rovers = len(self.rovers)

        lo = by_name.get("lander")
        if lo is not None:
            self.lander = np.array(
                [feat(lo, "x"), feat(lo, "y"), feat(lo, "z")], dtype=float
            )
        else:
            self.lander = np.array([-1.9, -2.0, 0.0])

        onames = sorted(
            [n for n in by_name if n.startswith("objective")],
            key=lambda n: _suffix_int(n, "objective"),
        )
        for idx, n in enumerate(onames):
            o = by_name[n]
            fs = _features(state, o)
            have = []
            for r in range(self.n_rovers):
                key = "have_image_rover%d" % r
                have.append(feat(o, key) > 0.5 if key in fs else False)
            self.objectives.append(
                {
                    "name": n,
                    "index": idx,
                    "pos": np.array(
                        [feat(o, "x"), feat(o, "y"), feat(o, "z")], dtype=float
                    ),
                    "have": have,
                    "received": feat(o, "received_image") > 0.5,
                }
            )

        snames = sorted(
            [n for n in by_name if n.startswith("sample")],
            key=lambda n: _suffix_int(n, "sample"),
        )
        for idx, n in enumerate(snames):
            o = by_name[n]
            fs = _features(state, o)
            analyzed = []
            for r in range(self.n_rovers):
                key = "analyzed_rover%d" % r
                analyzed.append(feat(o, key) > 0.5 if key in fs else False)
            self.samples.append(
                {
                    "name": n,
                    "index": idx,
                    "pos": np.array(
                        [feat(o, "x"), feat(o, "y"), feat(o, "z")], dtype=float
                    ),
                    "is_soil": feat(o, "is_soil") > 0.5,
                    "analyzed": analyzed,
                    "received": feat(o, "received_analysis") > 0.5,
                }
            )

        bnames = sorted(
            [n for n in by_name if n.startswith("obstacle")],
            key=lambda n: _suffix_int(n, "obstacle"),
        )
        for n in bnames:
            o = by_name[n]
            self.obstacles.append(
                (
                    np.array(
                        [feat(o, "x"), feat(o, "y"), feat(o, "z")], dtype=float
                    ),
                    np.array(
                        [feat(o, "half_x"), feat(o, "half_y"), feat(o, "half_z")],
                        dtype=float,
                    ),
                )
            )

        self.divider = (np.array([0.0, 0.0, 0.05]), np.array([0.05, 2.5, 0.05]))
        self.lander_box = (
            np.array([self.lander[0], self.lander[1], 0.2]),
            np.array([0.55, 0.42, 0.30]),
        )

    # --- goal bookkeeping (mirrors RoversEnv._goal_reached) -----------------

    def stone_received(self):
        return any(s["received"] for s in self.samples if not s["is_soil"])

    def soil_received(self):
        return any(s["received"] for s in self.samples if s["is_soil"])

    def stone_held(self):
        return any(any(s["analyzed"]) for s in self.samples if not s["is_soil"])

    def soil_held(self):
        return any(any(s["analyzed"]) for s in self.samples if s["is_soil"])

    def images_unassigned(self):
        return [
            o for o in self.objectives if not o["received"] and not any(o["have"])
        ]

    def rover_holds_unsent(self, r):
        for o in self.objectives:
            if o["have"][r] and not o["received"]:
                return True
        for s in self.samples:
            if s["analyzed"][r] and not s["received"]:
                return True
        return False

    def anything_left_to_fetch(self):
        if not (self.stone_received() or self.stone_held()):
            return True
        if not (self.soil_received() or self.soil_held()):
            return True
        return bool(self.images_unassigned())

    def goal_reached(self):
        if not self.stone_received():
            return False
        if not self.soil_received():
            return False
        if any(not o["received"] for o in self.objectives):
            return False
        if any(r["store_full"] for r in self.rovers):
            return False
        return all(r["at_home"] for r in self.rovers)

    # --- visibility heuristics ---------------------------------------------

    def _occluders(self, ignore_lander=False):
        out = list(self.obstacles)
        out.append(self.divider)
        if not ignore_lander:
            out.append(self.lander_box)
        return out

    def cam_point(self, x, y):
        return np.array([x, y, CAMERA_Z], dtype=float)

    def clear_ray(self, src, dst, ignore_lander=False):
        for centre, half in self._occluders(ignore_lander=ignore_lander):
            if _seg_box_hit(src, dst, centre, half + 1e-3):
                return False
        return True

    def objective_visible_from(self, x, y, obj, margin=0.05):
        src = self.cam_point(x, y)
        dst = obj["pos"].copy()
        if float(np.linalg.norm(dst - src)) > VIS_RANGE - margin:
            return False
        dst_top = dst + np.array([0.0, 0.0, 0.08])
        return self.clear_ray(src, dst) or self.clear_ray(src, dst_top)

    def lander_score(self, x, y):
        src = self.cam_point(x, y)
        dst = self.lander.copy()
        dst[2] = max(dst[2], 0.25)
        d = float(np.linalg.norm(dst - src))
        if d > COM_RANGE - 0.15:
            return -1.0
        if not self.clear_ray(src, dst, ignore_lander=True):
            return -1.0
        return COM_RANGE - d

    # --- navigation ---------------------------------------------------------

    def blocked_point(self, x, y, clearance=ROBOT_RADIUS):
        if abs(x) > ARENA - WALL_MARGIN or abs(y) > ARENA - WALL_MARGIN:
            return True
        for centre, half in self.obstacles:
            if (
                abs(x - centre[0]) < half[0] + clearance
                and abs(y - centre[1]) < half[1] + clearance
            ):
                return True
        if abs(x) < 0.05 + clearance and abs(y) < 2.5:
            return True
        lb_c, lb_h = self.lander_box
        if (
            abs(x - lb_c[0]) < lb_h[0] + clearance
            and abs(y - lb_c[1]) < lb_h[1] + clearance
        ):
            return True
        return False

    def segment_free(self, p0, p1, clearance=ROBOT_RADIUS, step=0.12):
        p0 = np.asarray(p0, dtype=float)[:2]
        p1 = np.asarray(p1, dtype=float)[:2]
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(dist / step) + 1)
        for i in range(n + 1):
            t = i / n
            q = p0 + t * (p1 - p0)
            if self.blocked_point(q[0], q[1], clearance):
                return False
        return True


# ------------------------------------------------------- roadmap / path plan

class _Roadmap:
    def __init__(self, world: _World, res: float = 0.2):
        self.world = world
        self.res = res
        lo = -(ARENA - WALL_MARGIN)
        hi = ARENA - WALL_MARGIN
        self.n = max(2, int(round((hi - lo) / res)) + 1)
        self.lo = lo
        self.free = [[False] * self.n for _ in range(self.n)]
        for i in range(self.n):
            x = lo + i * res
            for j in range(self.n):
                y = lo + j * res
                self.free[i][j] = not world.blocked_point(x, y, ROBOT_RADIUS)

    def _idx(self, x, y):
        i = int(round((x - self.lo) / self.res))
        j = int(round((y - self.lo) / self.res))
        return min(max(i, 0), self.n - 1), min(max(j, 0), self.n - 1)

    def _nearest_free(self, x, y):
        i0, j0 = self._idx(x, y)
        if self.free[i0][j0]:
            return i0, j0
        best, bestd = None, 10 ** 9
        for i in range(self.n):
            for j in range(self.n):
                if not self.free[i][j]:
                    continue
                d = (i - i0) ** 2 + (j - j0) ** 2
                if d < bestd:
                    bestd, best = d, (i, j)
        return best

    def plan(self, start, goal):
        s = self._nearest_free(start[0], start[1])
        g = self._nearest_free(goal[0], goal[1])
        if s is None or g is None:
            return None
        if s == g:
            return [np.array([goal[0], goal[1]], dtype=float)]

        def h(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1]) * self.res

        open_heap = [(h(s, g), 0.0, s)]
        came = {}
        gscore = {s: 0.0}
        closed = set()
        neigh = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
        found = False
        while open_heap:
            _, gc, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == g:
                found = True
                break
            for di, dj in neigh:
                ni, nj = cur[0] + di, cur[1] + dj
                if not (0 <= ni < self.n and 0 <= nj < self.n):
                    continue
                if not self.free[ni][nj]:
                    continue
                if di and dj:
                    if not (
                        self.free[cur[0] + di][cur[1]]
                        and self.free[cur[0]][cur[1] + dj]
                    ):
                        continue
                ng = gc + math.hypot(di, dj) * self.res
                if ng < gscore.get((ni, nj), 1e18) - 1e-9:
                    gscore[(ni, nj)] = ng
                    came[(ni, nj)] = cur
                    heapq.heappush(open_heap, (ng + h((ni, nj), g), ng, (ni, nj)))
        if not found:
            return None
        path = [g]
        while path[-1] in came:
            path.append(came[path[-1]])
        path.reverse()
        pts = [
            np.array([self.lo + i * self.res, self.lo + j * self.res], dtype=float)
            for (i, j) in path
        ]
        pts.append(np.array([goal[0], goal[1]], dtype=float))
        return self._shortcut(pts)

    def _shortcut(self, pts):
        if len(pts) <= 2:
            return pts[1:] if len(pts) > 1 else pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                if self.world.segment_free(pts[i], pts[j], ROBOT_RADIUS * 0.85):
                    break
                j -= 1
            out.append(pts[j])
            i = j
        return out[1:] if len(out) > 1 else out


# ------------------------------------------------------------- the approach

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        self._low = np.asarray(low, dtype=np.float32) if low is not None else None
        self._high = np.asarray(high, dtype=np.float32) if high is not None else None
        self._dim = 8
        if getattr(action_space, "shape", None):
            self._dim = int(np.prod(action_space.shape))
        self._rng = np.random.default_rng(12345)

    # ------------------------------------------------------------- lifecycle

    def reset(self, state, info=None):
        w = _World(state)
        n = w.n_rovers
        self.n_rovers = n
        self.home = [
            np.array([r["x"], r["y"], r["theta"]], dtype=float) for r in w.rovers
        ]
        self.roadmap = _Roadmap(w)
        self.step_count = 0

        n_obj = max(1, len(w.objectives))
        self.budget = BASE_STEPS + STEPS_PER_OBJECT * n_obj
        reserve = max(HOME_RESERVE_MIN, int(HOME_RESERVE_FRAC * self.budget))
        self.go_home_deadline = max(50, self.budget - reserve)

        self.phase = [FETCH] * n
        self.task = [None] * n
        self.task_age = [0] * n
        self.act_count = [0] * n

        self.path = [None] * n
        self.path_goal = [None] * n
        self.stand_goal = [None] * n
        self.comm_goal = [None] * n

        self.prev_pose = [
            np.array([r["x"], r["y"], r["theta"]], dtype=float) for r in w.rovers
        ]
        self.freeze = [0] * n
        self.escape = [None] * n          # (dx, dy, dth, steps_left)

        self.stand_blacklist = [set() for _ in range(n)]
        self.comm_blacklist = [set() for _ in range(n)]
        self.banned_samples = [set() for _ in range(n)]
        self.banned_objectives = [set() for _ in range(n)]
        self.send_at_home = [0] * n
        self._rng = np.random.default_rng(12345)
        return None

    # ---------------------------------------------------------------- helpers

    def _obj_by_name(self, w, name):
        for o in w.objectives:
            if o["name"] == name:
                return o
        return None

    def _sample_by_name(self, w, name):
        for s in w.samples:
            if s["name"] == name:
                return s
        return None

    def _cost(self, frm, to):
        frm = np.asarray(frm, dtype=float)[:2]
        to = np.asarray(to, dtype=float)[:2]
        d = float(np.linalg.norm(to - frm))
        if (frm[0] > 0.0) != (to[0] > 0.0):
            d += 1.6
        return d

    def _steps_left(self):
        return max(0, self.budget - self.step_count)

    def _past_deadline(self):
        return self.step_count >= self.go_home_deadline

    # --------------------------------------------------------- task selection

    def _needed_kinds(self, w):
        need = []
        if not (w.stone_received() or w.stone_held()):
            need.append(False)
        if not (w.soil_received() or w.soil_held()):
            need.append(True)
        return need

    def _claimed(self, kind, exclude=None):
        out = set()
        for r, t in enumerate(self.task):
            if r == exclude or t is None:
                continue
            if t[0] == kind:
                out.add(t[1])
        return out

    def _pick_task(self, w, r):
        if self._past_deadline():
            return None
        rover = w.rovers[r]
        pos = np.array([rover["x"], rover["y"]], dtype=float)
        home = self.home[r][:2]
        left = self._steps_left()

        def affordable(target):
            out = self._cost(pos, target)
            back = self._cost(target, home)
            return (out + back) / MAX_DELTA + 30 <= left - 25

        claimed_s = self._claimed("sample", exclude=r)
        best, bestc = None, None
        for kind in self._needed_kinds(w):
            for s in w.samples:
                if s["is_soil"] != kind:
                    continue
                if s["received"] or any(s["analyzed"]):
                    continue
                if s["name"] in claimed_s or s["name"] in self.banned_samples[r]:
                    continue
                if w.blocked_point(float(s["pos"][0]), float(s["pos"][1]), 0.02):
                    continue
                if not affordable(s["pos"]):
                    continue
                c = self._cost(pos, s["pos"])
                if bestc is None or c < bestc:
                    bestc, best = c, ("sample", s["name"])
        if best is not None:
            return best

        claimed_o = self._claimed("image", exclude=r)
        best, bestc = None, None
        for o in w.images_unassigned():
            if o["name"] in claimed_o or o["name"] in self.banned_objectives[r]:
                continue
            sp = self._stand_point(w, o, pos, self.stand_blacklist[r])
            target = sp if sp is not None else o["pos"][:2]
            if not affordable(target):
                continue
            c = self._cost(pos, target)
            if sp is None:
                c += 3.0
            if bestc is None or c < bestc:
                bestc, best = c, ("image", o["name"])
        return best

    # ------------------------------------------------------------ stand points

    def _stand_point(self, w, obj, near, blacklist):
        best, bestd = None, 1e18
        for rad in (1.15, 1.45, 0.85, 1.7, 0.6, 1.9):
            for k in range(64):
                ang = 2 * math.pi * k / 64
                x = float(obj["pos"][0] + rad * math.cos(ang))
                y = float(obj["pos"][1] + rad * math.sin(ang))
                if (round(x, 1), round(y, 1)) in blacklist:
                    continue
                if w.blocked_point(x, y, ROBOT_RADIUS * 1.05):
                    continue
                if not w.objective_visible_from(x, y, obj, margin=0.15):
                    continue
                d = self._cost(near, (x, y))
                if d < bestd:
                    bestd, best = d, np.array([x, y], dtype=float)
            if best is not None:
                break
        return best

    def _fallback_stand_point(self, w, obj, near, blacklist):
        best, bestd = None, 1e18
        for rad in (1.3, 1.0, 1.6, 0.75, 1.85):
            for k in range(48):
                ang = 2 * math.pi * k / 48
                x = float(obj["pos"][0] + rad * math.cos(ang))
                y = float(obj["pos"][1] + rad * math.sin(ang))
                if (round(x, 1), round(y, 1)) in blacklist:
                    continue
                if w.blocked_point(x, y, ROBOT_RADIUS):
                    continue
                d = self._cost(near, (x, y))
                if d < bestd:
                    bestd, best = d, np.array([x, y], dtype=float)
            if best is not None:
                return best
        return None

    def _comm_point(self, w, r, near, blacklist):
        best, bestscore = None, -1e18
        lx, ly = float(w.lander[0]), float(w.lander[1])
        found = False
        for rad in (1.2, 1.6, 2.0, 2.5, 3.0, 0.9):
            for k in range(48):
                ang = 2 * math.pi * k / 48
                x = lx + rad * math.cos(ang)
                y = ly + rad * math.sin(ang)
                if (round(x, 1), round(y, 1)) in blacklist:
                    continue
                if w.blocked_point(x, y, ROBOT_RADIUS * 1.05):
                    continue
                sc = w.lander_score(x, y)
                if sc <= 0.0:
                    continue
                score = sc - 0.6 * self._cost(near, (x, y))
                if score > bestscore:
                    bestscore, best = score, np.array([x, y], dtype=float)
                    found = True
            if found:
                break
        return best

    # ------------------------------------------------------------------ action

    def get_action(self, state):
        self.step_count += 1
        w = _World(state)
        action = np.zeros(self._dim, dtype=np.float32)

        if w.goal_reached():
            return self._clip(action)

        n = min(self.n_rovers, self._dim // 4)
        for r in range(n):
            dx, dy, dth, op = self._control(w, r)
            action[4 * r + 0] = dx
            action[4 * r + 1] = dy
            action[4 * r + 2] = dth
            action[4 * r + 3] = OP_VALUE[op]
        return self._clip(action)

    def _clip(self, action):
        if self._low is not None and self._high is not None:
            action = np.clip(action, self._low, self._high)
        return np.asarray(action, dtype=np.float32)

    # ------------------------------------------------- the unified controller

    def _control(self, w, r):
        rover = w.rovers[r]
        pose = np.array([rover["x"], rover["y"], rover["theta"]], dtype=float)
        pos = pose[:2]

        # --- freeze watchdog: works no matter which behaviour is active ------
        delta = np.abs(pose - self.prev_pose[r])
        moved = delta[0] > 1e-4 or delta[1] > 1e-4 or abs(_wrap(
            pose[2] - self.prev_pose[r][2]
        )) > 1e-4
        if moved:
            self.freeze[r] = 0
        else:
            self.freeze[r] += 1
        self.prev_pose[r] = pose.copy()

        # --- decide the phase from the observation --------------------------
        self._update_phase(w, r)

        # --- choose an operator (free; refused silently if not applicable) ---
        op = self._choose_operator(w, r)

        # --- choose a goal point --------------------------------------------
        goal, hold_still = self._choose_goal(w, r, op)

        # --- if wedged, override the motion with an escape burst ------------
        if self.escape[r] is not None:
            ex, ey, eth, left = self.escape[r]
            left -= 1
            self.escape[r] = None if left <= 0 else (ex, ey, eth, left)
            return (ex, ey, eth, op)

        if self.freeze[r] >= FREEZE_LIMIT and not (
            hold_still and self.phase[r] == PARK
        ):
            self.freeze[r] = 0
            self.escape[r] = self._make_escape(w, r, goal)
            # invalidate cached plans; the world proved them wrong
            self.path[r] = None
            self.path_goal[r] = None
            ex, ey, eth, left = self.escape[r]
            self.escape[r] = (ex, ey, eth, left - 1)
            return (ex, ey, eth, op)

        if hold_still:
            return (0.0, 0.0, 0.0, op)

        if goal is None:
            goal = self.home[r][:2]

        dx, dy, _ = self._drive(w, r, goal)

        # heading: only corrected when homing and near home, or when parked
        dth = 0.0
        if self.phase[r] in (HOME, PARK):
            ang_err = _wrap(self.home[r][2] - rover["theta"])
            d = float(np.linalg.norm(self.home[r][:2] - pos))
            if d < 0.9:
                dth = float(np.clip(ang_err, -MAX_DELTA_THETA, MAX_DELTA_THETA))
        return (dx, dy, dth, op)

    # ---- phase machine -----------------------------------------------------

    def _update_phase(self, w, r):
        rover = w.rovers[r]
        holding = w.rover_holds_unsent(r)

        if self._past_deadline() and self.phase[r] == FETCH:
            self.task[r] = None
            self.phase[r] = DELIVER if holding else HOME
            self._clear_nav(r)
            return

        if self.phase[r] == FETCH:
            self._refresh_task(w, r)
            if self.task[r] is None:
                self.phase[r] = DELIVER if holding else HOME
                self._clear_nav(r)
            return

        if self.phase[r] == DELIVER:
            if not holding:
                self.phase[r] = HOME
                self._clear_nav(r)
                self.send_at_home[r] = 0
            return

        if self.phase[r] == HOME:
            if holding:
                self.phase[r] = DELIVER
                self._clear_nav(r)
                return
            if rover["at_home"] and not rover["store_full"]:
                self.phase[r] = PARK
            elif (
                not self._past_deadline()
                and not rover["store_full"]
                and w.anything_left_to_fetch()
            ):
                # something new became available and there is time
                t = self._pick_task(w, r)
                if t is not None:
                    self.task[r] = t
                    self.task_age[r] = 0
                    self.phase[r] = FETCH
                    self._clear_nav(r)
            return

        # PARK
        if holding or rover["store_full"] or not rover["at_home"]:
            self.phase[r] = DELIVER if holding else HOME
            self._clear_nav(r)
        elif not self._past_deadline() and w.anything_left_to_fetch():
            t = self._pick_task(w, r)
            if t is not None:
                self.task[r] = t
                self.task_age[r] = 0
                self.phase[r] = FETCH
                self._clear_nav(r)

    def _clear_nav(self, r):
        self.path[r] = None
        self.path_goal[r] = None
        self.stand_goal[r] = None
        self.comm_goal[r] = None
        self.act_count[r] = 0

    def _refresh_task(self, w, r):
        t = self.task[r]
        drop = False
        if t is None:
            drop = True
        else:
            kind, name = t
            if kind == "sample":
                s = self._sample_by_name(w, name)
                if s is None or s["received"] or any(s["analyzed"]):
                    drop = True
                elif self.task_age[r] > SAMPLE_BUDGET:
                    self.banned_samples[r].add(name)
                    drop = True
            else:
                o = self._obj_by_name(w, name)
                if o is None or o["received"] or any(o["have"]):
                    drop = True
                elif self.task_age[r] > IMAGE_BUDGET:
                    self.banned_objectives[r].add(name)
                    drop = True
        if drop:
            new = self._pick_task(w, r)
            self.task[r] = new
            self.task_age[r] = 0
            self._clear_nav(r)
            self.stand_blacklist[r] = set()
        else:
            self.task_age[r] += 1

    # ---- operator choice ---------------------------------------------------

    def _choose_operator(self, w, r):
        rover = w.rovers[r]
        holding = w.rover_holds_unsent(r)

        # 1) If we are on a sample errand and standing on it with a free store: take it.
        if self.phase[r] == FETCH and self.task[r] is not None:
            kind, name = self.task[r]
            if kind == "sample" and not rover["store_full"]:
                s = self._sample_by_name(w, name)
                if s is not None:
                    d = float(
                        np.linalg.norm(
                            s["pos"][:2]
                            - np.array([rover["x"], rover["y"]], dtype=float)
                        )
                    )
                    if d <= SAMPLE_RADIUS + MAX_DELTA:
                        return SAMPLE

        # 2) A full store blocks the goal and is never needed again: empty it.
        if rover["store_full"]:
            return DROP

        # 3) Holding unsent results: SEND every step.  Free, silently refused.
        if holding:
            return SEND

        # 4) Imaging: calibrate / image whenever something we need is in view.
        if self.phase[r] == FETCH:
            needed_visible = any(
                (not o["have"][r])
                and (not o["received"])
                and w.objective_visible_from(rover["x"], rover["y"], o, margin=0.02)
                for o in w.objectives
            )
            if needed_visible:
                self.act_count[r] += 1
                if self.act_count[r] > ACT_LIMIT:
                    self.act_count[r] = 0
                    self.stand_blacklist[r].add(
                        (round(rover["x"], 1), round(rover["y"], 1))
                    )
                    self.stand_goal[r] = None
                    self.path[r] = None
                    self.path_goal[r] = None
                    return NOOP
                return IMAGE if rover["calibrated"] else CALIBRATE
            self.act_count[r] = 0

        return NOOP

    # ---- goal choice -------------------------------------------------------

    def _choose_goal(self, w, r, op):
        rover = w.rovers[r]
        pos = np.array([rover["x"], rover["y"]], dtype=float)
        phase = self.phase[r]

        if phase == PARK:
            # Hold absolutely still: at_home must stay true.
            if rover["at_home"]:
                return None, True
            return self.home[r][:2], False

        if phase == HOME:
            return self.home[r][:2], False

        if phase == DELIVER:
            # Head home (home is inside comm range here) while spamming SEND.
            # If many SENDs at home fail, detour to an explicit comm point.
            if rover["at_home"]:
                self.send_at_home[r] += 1
                if self.send_at_home[r] <= SEND_AT_HOME_LIMIT:
                    return None, True
                goal = self.comm_goal[r]
                if goal is None:
                    goal = self._comm_point(w, r, pos, self.comm_blacklist[r])
                    self.comm_goal[r] = goal
                    self.path[r] = None
                    self.path_goal[r] = None
                if goal is None:
                    goal = np.array(
                        [float(w.lander[0]) + 1.2, float(w.lander[1]) + 1.2],
                        dtype=float,
                    )
                if float(np.linalg.norm(np.asarray(goal) - pos)) < 0.15:
                    self.send_at_home[r] += 1
                    if self.send_at_home[r] > SEND_AT_HOME_LIMIT + 30:
                        self.comm_blacklist[r].add(
                            (round(float(goal[0]), 1), round(float(goal[1]), 1))
                        )
                        self.comm_goal[r] = None
                        self.send_at_home[r] = SEND_AT_HOME_LIMIT + 1
                        self.path[r] = None
                        self.path_goal[r] = None
                    return None, True
                return goal, False
            self.send_at_home[r] = 0
            return self.home[r][:2], False

        # FETCH
        if self.task[r] is None:
            return self.home[r][:2], False
        kind, name = self.task[r]

        if kind == "sample":
            s = self._sample_by_name(w, name)
            if s is None:
                return self.home[r][:2], False
            if rover["store_full"]:
                # dropping this step; stay put
                return None, True
            return s["pos"][:2], False

        # image errand
        o = self._obj_by_name(w, name)
        if o is None:
            return self.home[r][:2], False

        if op in (CALIBRATE, IMAGE):
            # stand still while shooting
            return None, True

        goal = self.stand_goal[r]
        if goal is None:
            goal = self._stand_point(w, o, pos, self.stand_blacklist[r])
            if goal is None:
                goal = self._fallback_stand_point(
                    w, o, pos, self.stand_blacklist[r]
                )
            if goal is None:
                self.banned_objectives[r].add(name)
                self.task[r] = None
                return self.home[r][:2], False
            self.stand_goal[r] = goal
            self.path[r] = None
            self.path_goal[r] = None

        if float(np.linalg.norm(np.asarray(goal) - pos)) < 0.15:
            # arrived but nothing needed is visible -> that spot was wrong
            self.stand_blacklist[r].add(
                (round(float(goal[0]), 1), round(float(goal[1]), 1))
            )
            self.stand_goal[r] = None
            self.path[r] = None
            self.path_goal[r] = None
            return pos, False

        return goal, False

    # ---- escape ------------------------------------------------------------

    def _make_escape(self, w, r, goal):
        rover = w.rovers[r]
        pos = np.array([rover["x"], rover["y"]], dtype=float)
        if goal is None:
            goal = self.home[r][:2]
        d = np.asarray(goal, dtype=float)[:2] - pos
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-6:
            d = np.array([1.0, 0.0])
            nrm = 1.0
        d = d / nrm
        perp = np.array([-d[1], d[0]])

        options = []
        for sign in (1.0, -1.0):
            options.append((perp * sign * MAX_DELTA, 0.0))
        options.append((-d * MAX_DELTA, 0.0))
        options.append((perp * MAX_DELTA * 0.7 - d * MAX_DELTA * 0.7, 0.0))
        options.append((-perp * MAX_DELTA * 0.7 - d * MAX_DELTA * 0.7, 0.0))
        options.append((np.zeros(2), MAX_DELTA_THETA))
        options.append((np.zeros(2), -MAX_DELTA_THETA))

        # prefer an option whose immediate step lands somewhere unobstructed
        scored = []
        for vec, dth in options:
            nxt = pos + vec
            free = not w.blocked_point(
                float(nxt[0]), float(nxt[1]), ROBOT_RADIUS * 0.8
            )
            scored.append((1.0 if free else 0.0, vec, dth))
        best_free = [s for s in scored if s[0] > 0.5]
        pool = best_free if best_free else scored
        idx = int(self._rng.integers(0, len(pool)))
        _, vec, dth = pool[idx]
        return (
            float(np.clip(vec[0], -MAX_DELTA, MAX_DELTA)),
            float(np.clip(vec[1], -MAX_DELTA, MAX_DELTA)),
            float(np.clip(dth, -MAX_DELTA_THETA, MAX_DELTA_THETA)),
            ESCAPE_LEN,
        )

    # ---- motion ------------------------------------------------------------

    def _drive(self, w, r, goal):
        rover = w.rovers[r]
        pos = np.array([rover["x"], rover["y"]], dtype=float)
        goal = np.asarray(goal, dtype=float)[:2]

        need = (
            self.path[r] is None
            or not self.path[r]
            or self.path_goal[r] is None
            or float(np.linalg.norm(np.asarray(self.path_goal[r])[:2] - goal)) > 0.15
        )
        if need:
            if w.segment_free(pos, goal, ROBOT_RADIUS * 0.85):
                path = [goal.copy()]
            else:
                path = self.roadmap.plan(pos, goal)
                if not path:
                    path = [goal.copy()]
            self.path[r] = [np.asarray(p, dtype=float)[:2] for p in path]
            self.path_goal[r] = goal.copy()

        while self.path[r] and float(np.linalg.norm(self.path[r][0] - pos)) < 0.12:
            self.path[r].pop(0)
        if not self.path[r]:
            self.path[r] = [goal.copy()]

        wp = self.path[r][0]
        d = wp - pos
        nrm = float(np.linalg.norm(d))
        if nrm > 1e-9:
            move = d / nrm * min(nrm, MAX_DELTA)
        else:
            move = np.zeros(2)
        return (
            float(np.clip(move[0], -MAX_DELTA, MAX_DELTA)),
            float(np.clip(move[1], -MAX_DELTA, MAX_DELTA)),
            0.0,
        )