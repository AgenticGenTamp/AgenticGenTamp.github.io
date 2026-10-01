"""Approach for Obstruction2DEnv (variable object count).

Strategy: the arena has no gravity, objects only move when carried by the
vacuum gripper.  We plan a sequence of pick-and-place moves: relocate every
obstruction that would block the goal placement (or a needed descent) to a
free slot on the table, then carry the target block onto the target surface.

All grasps are top-down (robot theta = -pi/2, arm fully extended) so the
kinematics are trivial: the gripper tip sits at (x, y - arm_joint - 0.005).
"""
import math
import time

import numpy as np

# ---- calibrated constants (measured on the real environment) ----
BASE_R = 0.1           # robot base radius
ARM_MIN, ARM_MAX = 0.1, 0.2
TIP_EXTRA = 0.005      # gripper tip distance beyond arm_joint
GRIP_HALF = 0.035      # half the gripper face width
TABLE_TOP = 0.1
X_MIN, X_MAX = 0.1005, 1.5175   # robot base-center limits (walls at 0 / 1.618)
Y_MAX = 0.895
TABLE_X1, TABLE_X2 = 0.004, 1.614
DOWN = -math.pi / 2
PLACE_BOTTOM = 0.114   # block bottom y when placed on the surface (is_on tol)
PARK_BOTTOM = 0.108    # obstruction bottom y when parked (>TABLE_TOP: touching collides)
GAP = 0.006            # tip<->object gap at grasp (grasp works iff gap < 0.015)
SNAG = 0.016           # a neighbour this close to the tip would be grabbed too
EDGE_M = 0.013         # required gripper/object horizontal overlap
EPS = 0.003            # generic safety margin (exact touching counts as collision)
CLEAR = 0.012          # vertical safety margin for travel
TOL = 1e-4
REACH = ARM_MAX + TIP_EXTRA   # 0.205


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def overlap(a1, a2, b1, b2):
    return a1 < b2 and b1 < a2


class Rect(object):
    __slots__ = ("name", "x", "y", "w", "h", "kind")

    def __init__(self, name, x, y, w, h, kind):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.kind = kind

    def copy(self):
        return Rect(self.name, self.x, self.y, self.w, self.h, self.kind)

    @property
    def x1(self):
        return self.x

    @property
    def x2(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2.0


class GeneratedApproach(object):

    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.observation_space = observation_space
        lo = np.asarray(action_space.low, dtype=np.float64)
        hi = np.asarray(action_space.high, dtype=np.float64)
        self._lo = lo * 0.99
        self._hi = hi * 0.99
        self.dtype = getattr(action_space, "dtype", np.float32)
        self.reset(None, None)

    # ------------------------------------------------------------------
    def reset(self, state, info=None):
        self.moves = []          # queued (obj_name, target_x1, off) tuples
        self.waypoints = []
        self.wp_i = 0
        self.stuck = 0
        self.replans = 0
        self.task = None
        self.prev_cfg = None
        self.time_used = 0.0
        self.carry_ref = None
        self.expected_steps = None

    # ---------------- state parsing ----------------
    def parse(self, state):
        """Read an ObjectCentricState defensively (no fixed index layout)."""
        robot = None
        rects = []
        target = None
        surface = None
        for name in state.get_object_names():
            o = state.get_object_from_name(name)
            tname = getattr(getattr(o, "type", None), "name", "")
            try:
                arm = float(state.get(o, "arm_joint"))
            except Exception:
                arm = None
            if arm is not None or tname == "crv_robot":
                robot = {"x": float(state.get(o, "x")),
                         "y": float(state.get(o, "y")),
                         "theta": float(state.get(o, "theta")),
                         "arm_joint": arm if arm is not None else ARM_MIN,
                         "vacuum": float(state.get(o, "vacuum"))}
                continue
            try:
                w = float(state.get(o, "width"))
                h = float(state.get(o, "height"))
            except Exception:
                continue
            r = Rect(name, float(state.get(o, "x")), float(state.get(o, "y")),
                     w, h, tname)
            if tname == "target_block" or name == "target_block":
                target = r
            elif tname == "target_surface" or name == "target_surface":
                surface = r
            else:
                rects.append(r)
        rects.sort(key=lambda r: r.name)
        return robot, rects, target, surface

    # ---------------- geometry ----------------
    def grasp_y(self, obj_top):
        return obj_top + REACH + GAP

    def min_y(self, xa, xb, others, carried=None, pick=None):
        """Lowest safe robot-base y while the base column sweeps [xa, xb].

        carried = (height, left_off, right_off): the held object spans
        [x - left_off, x + right_off] and hangs to y - REACH - GAP - height.
        """
        m = TABLE_TOP + BASE_R + EPS
        gl = xa - GRIP_HALF - 0.002
        gr = xb + GRIP_HALF + 0.002
        if carried is not None:
            ch, cl, cr = carried
            m = max(m, TABLE_TOP + EPS + ch + REACH + GAP)
            cl = xa - cl
            cr = xb + cr
        for o in others:
            if overlap(o.x1, o.x2, xa - BASE_R, xb + BASE_R):
                v = o.top + BASE_R + EPS
                if v > m:
                    m = v
            if overlap(o.x1, o.x2, gl, gr):
                v = o.top + REACH + (0.001 if o.name == pick else SNAG)
                if v > m:
                    m = v
            if carried is not None and overlap(o.x1, o.x2, cl, cr):
                v = o.top + REACH + GAP + ch + EPS
                if v > m:
                    m = v
        return m

    def violators(self, gx, y_goal, others, carried=None, pick=None):
        """Names of objects that make base-y = y_goal unsafe at column gx."""
        bad = []
        gl = gx - GRIP_HALF - 0.002
        gr = gx + GRIP_HALF + 0.002
        if carried is not None:
            ch, cl, cr = carried
            cl = gx - cl
            cr = gx + cr
        for o in others:
            hit = False
            if overlap(o.x1, o.x2, gx - BASE_R, gx + BASE_R) \
                    and o.top + BASE_R + EPS > y_goal:
                hit = True
            elif overlap(o.x1, o.x2, gl, gr) \
                    and o.top + REACH + (0.001 if o.name == pick else SNAG) > y_goal:
                hit = True
            elif carried is not None and overlap(o.x1, o.x2, cl, cr) \
                    and o.top + REACH + GAP + ch + EPS > y_goal:
                hit = True
            if hit:
                bad.append(o.name)
        return bad

    # ---------------- move feasibility ----------------
    def find_move(self, obj, others, target_x1, place_bottom):
        """Pick a grasp offset for moving obj to target_x1; None if infeasible."""
        off_lo = -GRIP_HALF + EDGE_M
        off_hi = obj.w + GRIP_HALF - EDGE_M
        if off_hi < off_lo:
            off_lo = off_hi = obj.w / 2.0
        y_pick = self.grasp_y(obj.top)
        y_place = self.grasp_y(place_bottom + obj.h)
        best = None
        n = 33
        for i in range(n):
            off = off_lo + (off_hi - off_lo) * i / (n - 1.0)
            gx_pick = obj.x1 + off
            gx_place = target_x1 + off
            if not (X_MIN <= gx_pick <= X_MAX and X_MIN <= gx_place <= X_MAX):
                continue
            if self.min_y(gx_pick, gx_pick, others + [obj], None, obj.name) > y_pick + 1e-9:
                continue
            carried = (obj.h, off, obj.w - off)
            if self.min_y(gx_place, gx_place, others, carried) > y_place + 1e-9:
                continue
            score = abs(off - obj.w / 2.0)
            if best is None or score < best[0]:
                best = (score, off)
        return None if best is None else best[1]

    def move_conflicts(self, obj, others, target_x1, place_bottom):
        """Return (off, conflict_sets). off is None when the move is blocked."""
        off_lo = -GRIP_HALF + EDGE_M
        off_hi = obj.w + GRIP_HALF - EDGE_M
        if off_hi < off_lo:
            off_lo = off_hi = obj.w / 2.0
        y_pick = self.grasp_y(obj.top)
        y_place = self.grasp_y(place_bottom + obj.h)
        best = None
        confs = []
        n = 17
        for i in range(n):
            off = off_lo + (off_hi - off_lo) * i / (n - 1.0)
            gx_pick = obj.x1 + off
            gx_place = target_x1 + off
            if not (X_MIN <= gx_pick <= X_MAX and X_MIN <= gx_place <= X_MAX):
                continue
            v = set(self.violators(gx_pick, y_pick, others + [obj],
                                   pick=obj.name))
            v |= set(self.violators(gx_place, y_place, others,
                                    (obj.h, off, obj.w - off)))
            if not v:
                score = abs(off - obj.w / 2.0)
                if best is None or score < best[0]:
                    best = (score, off)
            else:
                confs.append(frozenset(v))
        if best is not None:
            return best[1], []
        confs.sort(key=len)
        return None, confs[:4]

    # ---------------- parking slots ----------------
    def slot_candidates(self, obj, keep, pad, forbidden):
        forb = [(r.x1 - pad, r.x2 + pad) for r in keep]
        forb.extend(forbidden)
        forb.sort()
        merged = []
        for a, b in forb:
            if merged and a <= merged[-1][1]:
                if b > merged[-1][1]:
                    merged[-1][1] = b
            else:
                merged.append([a, b])
        free = []
        cur = TABLE_X1
        for a, b in merged:
            if a - cur >= obj.w:
                free.append((cur, a))
            if b > cur:
                cur = b
        if TABLE_X2 - cur >= obj.w:
            free.append((cur, TABLE_X2))
        cands = []
        for a, b in free:
            mid = (a + b - obj.w) / 2.0
            for c in (mid, a, b - obj.w, (a + mid) / 2.0, (b - obj.w + mid) / 2.0):
                c = _clip(c, a, b - obj.w)
                if all(abs(c - e) > 1e-6 for e in cands):
                    cands.append(c)
        cands.sort(key=lambda c: abs(c - obj.x1))
        return cands

    def park_bottom(self, obj, slot, keep, prot, ceil_top):
        """Height at which to release a parked obstruction (air-park when safe)."""
        lo = min(obj.x1, slot) - BASE_R - 0.04
        hi = max(obj.x2, slot + obj.w) + BASE_R + 0.04
        if overlap(slot - 0.14, slot + obj.w + 0.14, prot[0], prot[1]):
            return PARK_BOTTOM
        top = TABLE_TOP
        for o in keep:
            if overlap(o.x1, o.x2, lo, hi) and o.top > top:
                top = o.top
        pb = top + EPS
        hi_lim = ceil_top - obj.h
        if pb > hi_lim:
            pb = hi_lim
        if pb < PARK_BOTTOM:
            pb = PARK_BOTTOM
        return pb

    # ---------------- target placement ----------------
    def target_x_candidates(self, target, surface):
        lo = surface.x1 + 0.002
        hi = surface.x2 - target.w - 0.002
        if hi < lo:
            lo = hi = (surface.x1 + surface.x2 - target.w) / 2.0
        mid = (lo + hi) / 2.0
        cands = [mid]
        n = 13
        for i in range(n):
            c = lo + (hi - lo) * i / (n - 1.0)
            cands.append(c)
        cands.sort(key=lambda c: abs(c - mid))
        return cands

    # ---------------- planner ----------------
    def compute_plan(self, robot, rects, target, surface):
        """Beam-search a cheap sequence of moves ending with the target block."""
        if self.time_used > 25.0:
            return None
        cands = []
        t0 = time.time()
        budget = 3.0 if self.time_used < 10.0 else 0.3
        for tx in self.target_x_candidates(target, surface):
            cands.extend(self._search(robot, rects, target, surface, tx))
            if time.time() - t0 > budget:
                break
        if cands:
            cands.sort(key=lambda t: t[0])
            cands = cands[:60]
            best = None
            for _, plan in cands:
                if time.time() - t0 > budget * 2.0 and best is not None:
                    break
                n = self.sim_plan(robot, rects, target, plan)
                if best is None or n < best[0]:
                    best = (n, plan)
            self.expected_steps = best[0]
            self.time_used += time.time() - t0
            return best[1]
        self.time_used += time.time() - t0
        return None

    MOVE_OVERHEAD = 0.14
    BEAM = 10

    def _search(self, robot, rects, target, surface, tx):
        max_h = max([r.h for r in rects] + [target.h])
        ceil_top = Y_MAX - REACH - GAP - max_h - EPS
        prot = (min(target.x1, tx) - 0.14, max(target.x2, tx + target.w) + 0.14)
        tgt_final = Rect("__t__", tx, PLACE_BOTTOM, target.w, target.h, "target_block")
        tcur = target.copy()
        forb = [(tx - 0.14, tx + target.w + 0.14),
                (surface.x1 - 0.02, surface.x2 + 0.02)]
        beam = [(0.0, robot["x"], [r.copy() for r in rects], [], frozenset())]
        found = []
        best = [None]

        def expand(node, focus, out):
            cost, cx, layout, moves, moved = node
            for o in layout:
                if o.name in moved or o.name not in focus:
                    continue
                keep = [r for r in layout if r is not o] + [tcur]
                opts = []
                for pad in (0.118, 0.06, 0.025, 0.006):
                    for slot in self.slot_candidates(o, keep + [tgt_final], pad, forb):
                        if abs(slot - o.x1) < 1e-9:
                            continue
                        if any(abs(slot - v) < 0.02 for v in opts):
                            continue
                        opts.append(slot)
                    if len(opts) >= 10:
                        break
                opts.sort(key=lambda sl: abs(sl - o.x1))
                got = 0
                for slot in opts:
                    pb = self.park_bottom(o, slot, keep, prot, ceil_top)
                    off2 = self.find_move(o, keep, slot, pb)
                    if off2 is None and pb > PARK_BOTTOM:
                        pb = PARK_BOTTOM
                        off2 = self.find_move(o, keep, slot, pb)
                    if off2 is None:
                        continue
                    gp = o.x1 + off2
                    gq = slot + off2
                    c = cost + abs(gp - cx) + abs(gq - gp) + self.MOVE_OVERHEAD
                    if best[0] is not None and c >= best[0]:
                        continue
                    nl = [r.copy() for r in layout]
                    for r in nl:
                        if r.name == o.name:
                            r.x = slot
                            r.y = pb
                    out.append((c, gq, nl, moves + [(o.name, slot, off2, pb)],
                                moved | frozenset([o.name])))
                    got += 1
                    if got >= 5:
                        break

        for _ in range(len(rects) + 1):
            nxt = []
            for node in beam:
                cost, cx, layout, moves, moved = node
                if best[0] is not None and cost >= best[0]:
                    continue
                off = self.find_move(target, layout, tx, PLACE_BOTTOM)
                if off is not None:
                    gp = target.x1 + off
                    gq = tx + off
                    c = cost + abs(gp - cx) + abs(gq - gp) + self.MOVE_OVERHEAD
                    plan = moves + [(target.name, tx, off, PLACE_BOTTOM)]
                    found.append((c, plan))
                    if best[0] is None or c < best[0]:
                        best[0] = c
                    continue
                _, confs = self.move_conflicts(target, layout, tx, PLACE_BOTTOM)
                focus = set()
                for cf in confs:
                    focus |= cf
                focus -= moved
                n0 = len(nxt)
                if focus:
                    expand(node, focus, nxt)
                if len(nxt) == n0:
                    expand(node, set(o.name for o in layout), nxt)
            if not nxt:
                break
            nxt.sort(key=lambda t: t[0])
            beam = nxt[:self.BEAM]
        return found

    # ---------------- task construction ----------------
    def build_task(self, robot, rects, target, surface, move):
        name = move[0]
        allr = list(rects) + [target]
        obj = None
        for r in allr:
            if r.name == name:
                obj = r
                break
        if obj is None:
            return None
        others = [r for r in allr if r is not obj]
        return self.build_task_rects(obj, others, move)

    def build_task_rects(self, obj, others, move):
        name, tx, off, place_bottom = move
        gx_pick = obj.x1 + off
        gx_place = tx + off
        y_pick = self.grasp_y(obj.top)
        y_place = self.grasp_y(place_bottom + obj.h)
        carried = (obj.h, off, obj.w - off)
        wps = [
            dict(x=gx_pick, y=y_pick, vac=0.0, mode="go", carried=None,
                 others=others + [obj], pick=obj.name),
            dict(x=gx_pick, y=y_pick, vac=1.0, mode="grasp", carried=None,
                 others=others, pick=None),
            dict(x=gx_place, y=y_place, vac=1.0, mode="go", carried=carried,
                 others=others, pick=None),
            dict(x=gx_place, y=y_place, vac=0.0, mode="release", carried=None,
                 others=others, pick=None),
        ]
        return dict(obj=name, tx=tx, off=off, wps=wps)

    # ---------------- main entry ----------------
    def get_action(self, state):
        robot, rects, target, surface = self.parse(state)
        if self.task is None or self.wp_i >= len(self.waypoints):
            self.next_task(robot, rects, target, surface)
        elif robot["vacuum"] > 0.5 and self.task.get("obj"):
            # verify the payload is still rigidly attached
            obj = None
            for r in rects + [target]:
                if r.name == self.task["obj"]:
                    obj = r
                    break
            if obj is not None:
                ref = (robot["x"] - obj.x, robot["y"] - obj.y)
                if self.carry_ref is None:
                    self.carry_ref = ref
                elif abs(ref[0] - self.carry_ref[0]) > 0.004 \
                        or abs(ref[1] - self.carry_ref[1]) > 0.004:
                    self.carry_ref = None
                    self.replan(robot, rects, target, surface)
        act = self.execute(robot)
        # stuck detection
        cfg = (round(robot["x"], 6), round(robot["y"], 6), round(robot["arm_joint"], 6),
               round(robot["theta"], 6), robot["vacuum"], self.wp_i)
        if cfg == self.prev_cfg:
            self.stuck += 1
        else:
            self.stuck = 0
        self.prev_cfg = cfg
        if self.stuck >= 4:
            self.stuck = 0
            self.replan(robot, rects, target, surface)
            act = self.execute(robot)
        return np.asarray(act, dtype=self.dtype)

    def replan(self, robot, rects, target, surface):
        self.replans += 1
        self.moves = []
        self.task = None
        self.next_task(robot, rects, target, surface)

    def next_task(self, robot, rects, target, surface):
        for _ in range(3):
            if self.replans > 40:
                break
            if not self.moves:
                plan = self.compute_plan(robot, rects, target, surface)
                if plan is None:
                    plan = self.fallback_plan(robot, rects, target, surface)
                self.moves = plan or []
            if not self.moves:
                break
            mv = self.moves.pop(0)
            task = self.build_task(robot, rects, target, surface, mv)
            if task is not None:
                self.task = task
                self.waypoints = task["wps"]
                self.wp_i = 0
                self.carry_ref = None
                return
        # nothing to do: hover in place
        self.task = dict(obj=None, wps=[])
        self.waypoints = [dict(x=robot["x"], y=min(Y_MAX, robot["y"] + 0.05),
                               vac=0.0, mode="go", carried=None, pick=None,
                               others=list(rects) + [target])]
        self.wp_i = 0
        self.carry_ref = None

    def fallback_plan(self, robot, rects, target, surface):
        """Relax constraints: just try to shove obstructions anywhere reachable."""
        for tx in self.target_x_candidates(target, surface):
            off = self.find_move(target, rects, tx, PLACE_BOTTOM)
            if off is not None:
                return [(target.name, tx, off, PLACE_BOTTOM)]
        # move the object nearest the surface to the farthest free place
        tx = self.target_x_candidates(target, surface)[0]
        cands = sorted(rects, key=lambda o: abs(o.cx - (tx + target.w / 2.0)))
        for o in cands:
            keep = [r for r in rects if r is not o] + [target]
            for pad in (0.118, 0.06, 0.02, 0.0):
                for slot in self.slot_candidates(o, keep, pad, []):
                    if abs(slot - o.x1) < 1e-9:
                        continue
                    off2 = self.find_move(o, keep, slot, PARK_BOTTOM)
                    if off2 is not None:
                        return [(o.name, slot, off2, PARK_BOTTOM)]
        return []

    # ---------------- execution ----------------
    def motion_step(self, x, y, arm, wp):
        """One control step toward waypoint wp; returns (dx, dy, darm)."""
        dxmax = self._hi[0]
        dymax = self._hi[1]
        damax = self._hi[3]
        others = wp["others"]
        carried = wp["carried"]
        pick = wp.get("pick")
        gx, gy = wp["x"], wp["y"]
        da = _clip(ARM_MAX - arm, -damax, damax)
        dx_want = _clip(gx - x, -dxmax, dxmax)
        need_full = self.min_y(min(x, x + dx_want), max(x, x + dx_want),
                               others, carried, pick)
        if need_full <= y + dymax + 1e-9:
            dx = dx_want
            ylim = need_full
            target_y = gy
        else:
            dx = 0.0
            ylim = self.min_y(x, x, others, carried, pick)
            for scale in (0.7, 0.4, 0.2):
                cand = dx_want * scale
                nf = self.min_y(min(x, x + cand), max(x, x + cand), others,
                                carried, pick)
                if nf <= y + dymax + 1e-9:
                    dx = cand
                    ylim = nf
                    break
            target_y = min(need_full, Y_MAX)
        if target_y < ylim:
            target_y = ylim
        ny = _clip(target_y, y - dymax, y + dymax)
        if ny < ylim:
            ny = min(y + dymax, ylim)
        ny = min(ny, Y_MAX)
        return dx, ny - y, da

    def execute(self, robot):
        act = np.zeros(5, dtype=np.float64)
        x, y = robot["x"], robot["y"]
        while self.wp_i < len(self.waypoints):
            wp = self.waypoints[self.wp_i]
            if wp["mode"] in ("grasp", "release"):
                if robot["vacuum"] == wp["vac"]:
                    self.wp_i += 1
                    continue
                act[4] = wp["vac"]
                return act
            dth = DOWN - robot["theta"]
            while dth > math.pi:
                dth -= 2 * math.pi
            while dth < -math.pi:
                dth += 2 * math.pi
            da = ARM_MAX - robot["arm_joint"]
            gx, gy = wp["x"], wp["y"]
            if abs(gx - x) < TOL and abs(gy - y) < TOL and abs(da) < TOL \
                    and abs(dth) < 1e-4:
                self.wp_i += 1
                continue
            if abs(dth) > 0.02:
                act[2] = _clip(dth, self._lo[2], self._hi[2])
                act[3] = _clip(-0.05, self._lo[3], self._hi[3])
                act[4] = wp["vac"]
                return act
            dx, dy, da = self.motion_step(x, y, robot["arm_joint"], wp)
            if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(da) < TOL:
                return act
            act[0] = _clip(dx, self._lo[0], self._hi[0])
            act[1] = _clip(dy, self._lo[1], self._hi[1])
            act[2] = _clip(dth, self._lo[2], self._hi[2])
            act[3] = _clip(da, self._lo[3], self._hi[3])
            act[4] = wp["vac"]
            return act
        return act

    # ---------------- plan simulation (step counting) ----------------
    def sim_plan(self, robot, rects, target, plan, limit=900):
        layout = dict((r.name, r.copy()) for r in rects)
        layout[target.name] = target.copy()
        x = robot["x"]
        y = robot["y"]
        arm = robot["arm_joint"]
        n = 0
        for move in plan:
            name = move[0]
            obj = layout[name]
            others = [r for r in layout.values() if r.name != name]
            task = self.build_task_rects(obj, others, move)
            for wp in task["wps"]:
                if wp["mode"] in ("grasp", "release"):
                    n += 1
                    continue
                for _ in range(400):
                    if abs(wp["x"] - x) < TOL and abs(wp["y"] - y) < TOL \
                            and abs(ARM_MAX - arm) < TOL:
                        break
                    dx, dy, da = self.motion_step(x, y, arm, wp)
                    if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(da) < TOL:
                        n += 60
                        break
                    x += dx
                    y += dy
                    arm += da
                    n += 1
                    if n > limit:
                        return limit
            obj.x = move[1]
            obj.y = move[3]
        return n
