"""Fallback route: fetch a spare green block from the far table and drop it on the plate.

Public API
----------
    FarRoute()                      -- state-machine policy
        .reset(state, info=None)    -- plan the route from the reset observation
        .get_action(state) -> (11,) float32 action
        .done                       -- True once the block has been released

    plan_far_route(state_blocks, robot_state) -> list of waypoint dicts
        A declarative view of the same plan (base poses / joint targets).

Only depends on numpy + helpers imported from approach.py.
"""
import numpy as np
from approach import (world_fk, world_chain, ik_solutions, grasp_R, dq_wrap,
                      clip_q, seg_points, MAXD)

# ---------------------------------------------------------------- geometry --
SHOULDER_XY = np.array([-0.05, 0.188])   # shoulder offset in base frame
BASE_CLEAR = 0.42                        # required base clearance from a table
WORLD_LIM = 4.85
TABLE_HX, TABLE_HY = 0.30, 0.60
NEAR_C = np.array([4.5, 0.0])
FAR_C = np.array([-4.5, 0.0])
PLATE_X = (4.2, 4.8)
PLATE_Y = (-0.6, 0.0)
PLATE_TGT = np.array([4.50, -0.30])      # where the carried block should be dropped
GRASP_BACK = 0.02
GRASP_UP = 0.03
LIFT = 0.16
PRE = 0.20                               # pregrasp stand-off along -dir
OVER = 0.09                              # height of the fly-over above the grasp pose


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _rect_dist(p, c, hx=TABLE_HX, hy=TABLE_HY):
    d = np.abs(np.asarray(p, float) - c) - np.array([hx, hy])
    d = np.maximum(d, 0.0)
    return float(np.hypot(d[0], d[1]))


def _base_ok(p):
    if abs(p[0]) > WORLD_LIM or abs(p[1]) > WORLD_LIM:
        return False
    return (_rect_dist(p, NEAR_C) >= BASE_CLEAR and
            _rect_dist(p, FAR_C) >= BASE_CLEAR)


def _travel(a, b):
    """Steps needed to translate the base from a to b (x and y move together)."""
    d = np.abs(np.asarray(b, float) - np.asarray(a, float))
    return float(max(d[0], d[1]) / MAXD)


# ------------------------------------------------------------ state access --
def robot_state(state):
    r = np.asarray(state.data[state.get_object_from_name("robot")], dtype=float)
    return dict(base=r[0:3].copy(), q=r[3:10].copy(), grip=float(r[10]),
                holding=float(r[11]) > 0.5, raw=r[:12].copy())


def state_blocks(state):
    out = {}
    for o in state:
        if o.type.name == "block":
            out[o.name] = np.asarray(state.data[o][:3], dtype=float)
    return out


def block_yaws(state):
    """Z-yaw of every block, from its quaternion."""
    out = {}
    for o in state:
        if o.type.name == "block":
            d = np.asarray(state.data[o], dtype=float)
            qx, qy, qz, qw = d[3:7]
            out[o.name] = float(np.arctan2(2 * (qw * qz + qx * qy),
                                           1 - 2 * (qy * qy + qz * qz)))
    return out


def spare_greens(state):
    """Green blocks sitting on the far table (green1..greenN)."""
    b = state_blocks(state)
    return {n: p for n, p in b.items()
            if n.startswith("green") and n != "green0" and p[0] < 0.0}


# ------------------------------------------------------------------ IK bits --
def _arm_cost(q, base, tables=(FAR_C, NEAR_C)):
    """Penalise configurations whose elbow/forearm dips into a table."""
    pts, _ = world_chain(q, np.asarray(base, float))
    cost = -0.6 * min(pts[2][2], 1.15)
    tip = pts[4]
    bad = False
    for p in seg_points(pts[1], pts[2], 5) + seg_points(pts[2], pts[3], 6):
        if np.linalg.norm(p - tip) < 0.20:
            continue
        for c in tables:
            d = _rect_dist(p[:2], c)
            if d < 0.20 and p[2] < 0.88:
                cost += 4.0 * (0.88 - p[2]) * (0.20 - d) / 0.20
                if d < 0.10 and p[2] < 0.80:
                    bad = True
    return cost, bad


def _grasp_base(block_xy, dirv, radius):
    perp = np.array([-dirv[1], dirv[0]])
    return np.asarray(block_xy, float) - dirv * radius - perp * SHOULDER_XY[1]


def _descent_pts(pre, tgt, up, n1=4, n2=2):
    """Move in above the block, then descend onto the grasp pose."""
    hi = tgt + up
    pts = [pre + (hi - pre) * (i + 1.0) / n1 for i in range(n1)]
    pts += [hi + (tgt - hi) * (i + 1.0) / n2 for i in range(n2)]
    return pts


def _cart_path(base, q0, pts, R, seeds=6, iters=90, max_jump=0.8):
    """Warm-started IK along a list of cartesian points. Returns [q..] or None."""
    q = np.asarray(q0, float)
    out = []
    for p in pts:
        sols = ik_solutions(np.asarray(p, float), R, base, q, seeds=seeds, iters=iters)
        if not sols:
            return None
        best, bd = None, 1e9
        for s in sols:
            d = float(np.max(np.abs(dq_wrap(s - q))))
            if d < bd:
                best, bd = s, d
        if bd > max_jump:
            return None
        q = best
        out.append(q.copy())
    return out


def _plan_grasp(block, dirv, radius, q_seed):
    """Full pregrasp->grasp joint path for one (block, direction, radius)."""
    bxy = _grasp_base(block[:2], dirv, radius)
    if not _base_ok(bxy):
        return None
    yaw = float(np.arctan2(dirv[1], dirv[0]))
    base = np.array([bxy[0], bxy[1], yaw])
    R = grasp_R(yaw)
    d3 = np.array([dirv[0], dirv[1], 0.0])
    tgt = np.asarray(block, float) - d3 * GRASP_BACK + np.array([0, 0, GRASP_UP])
    up = np.array([0.0, 0.0, OVER])
    pre = tgt - d3 * PRE + up          # stand off high: a straight-in approach jams
    sols = []
    for sd in (12, 12):
        sols += ik_solutions(pre, R, base, q_seed, seeds=sd, iters=140,
                             rng=np.random.default_rng(sd))
    scored = []
    for qp in sols:
        c, bad = _arm_cost(qp, base)
        if bad:
            continue
        scored.append((c + 0.15 * float(np.max(np.abs(dq_wrap(qp - q_seed)))), qp))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    best = None
    for _, q_pre in scored[:4]:
        pts = _descent_pts(pre, tgt, up)
        path = _cart_path(base, q_pre, pts, R, max_jump=0.6)
        if path is None:
            continue
        cc, bad = 0.0, False
        for q in [q_pre] + path:
            c, b = _arm_cost(q, base)
            cc = max(cc, c)
            bad = bad or b
        if bad:
            continue
        if best is None or cc < best[0]:
            best = (cc, dict(base=base, q_pre=np.asarray(q_pre, float), path=path, tgt=tgt,
                             R=R, dirv=np.asarray(dirv, float),
                             block=np.asarray(block, float), arm_cost=cc))
    return None if best is None else best[1]


def _approach_angles(yaw):
    """Face normals of a box with this yaw, easterly ones first."""
    if yaw is None:
        return [np.pi] + [np.pi + s * a for a in
                          (np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi)
                          for s in (1, -1)]
    faces = [_wrap(yaw + k * np.pi / 2) for k in range(4)]
    faces.sort(key=lambda a: abs(_wrap(a - np.pi)))
    out = []
    for f in faces:                      # exact face normal first, then small tilts
        out += [f, _wrap(f + 0.17), _wrap(f - 0.17)]
    return out


def _plan_pick(blocks, q_seed, start_xy, yaws=None,
               radii=(0.80, 0.86, 0.74, 0.90, 0.68)):
    """Choose the cheapest reachable spare + approach direction.

    The approach axis is aligned with a face normal of the (randomly yawed)
    block, otherwise the gripper jams on a corner.
    """
    # easiest first: the block nearest the far table's east edge
    order = sorted(blocks.items(), key=lambda kv: -kv[1][0])
    cands = []
    for rank, (name, pos) in enumerate(order):
        angs = _approach_angles(None if yaws is None else yaws.get(name))
        for a in angs:
            dirv = np.array([np.cos(a), np.sin(a)])
            for radius in radii:
                bxy = _grasp_base(pos[:2], dirv, radius)
                if not _base_ok(bxy):
                    continue
                cost = (_travel(start_xy, bxy) + _travel(bxy, [3.72, -0.49])
                        + 3.0 * rank + 2.5 * abs(_wrap(a - np.pi)))
                cands.append((cost, name, dirv, radius, pos))
                break                       # first feasible radius for this dir
    cands.sort(key=lambda c: c[0])
    for cost, name, dirv, radius, pos in cands[:14]:
        pl = _plan_grasp(pos, dirv, radius, q_seed)
        if pl is not None:
            pl["name"] = name
            pl["cost"] = cost
            return pl
    return None


def _release_base(q, tgt_xy=PLATE_TGT, start_xy=None):
    """Base pose (with joints frozen at q) that puts the tool over tgt_xy."""
    best = None
    for yaw in np.linspace(-np.pi, np.pi, 145):
        p, _ = world_fk(q, np.array([0.0, 0.0, yaw]))     # tool offset in world for this yaw
        bxy = np.asarray(tgt_xy, float) - p[:2]
        if not _base_ok(bxy):
            continue
        c = _travel(start_xy, bxy) if start_xy is not None else -bxy[0]
        if best is None or c < best[0]:
            best = (c, np.array([bxy[0], bxy[1], yaw]))
    return None if best is None else best[1]


# ------------------------------------------------------------------ policy --
class FarRoute:
    """Whole fallback policy: far table -> grasp a spare -> near table -> drop on plate."""

    def __init__(self, action_space=None, observation_space=None, primitives=None,
                 debug=False):
        self.done = False
        self.debug = bool(debug)

    # -- planning ----------------------------------------------------------
    def reset(self, state, info=None):
        r = robot_state(state)
        self.plan = _plan_pick(spare_greens(state), r["q"], r["base"][:2],
                               yaws=block_yaws(state))
        if self.plan is None:
            raise RuntimeError("farroute: no reachable spare green block")
        self.phase = "drive_out"
        self.wp = list(self.plan["path"])          # joint waypoints for the approach
        self.q_hold = None
        self.base_goal = None
        self.prev = None
        self.prev_act = None
        self.stall = 0
        self.mode = 0            # rejection-recovery mode: 0 both, 1 base, 2 arm
        self.detour = None
        self.grip = 0.0
        self.tries = 0
        self.c_try = 0
        self.c_stage = 'move'
        self.retreat = 0
        self.steps = 0
        self.done = False
        return self.plan

    # -- helpers -----------------------------------------------------------
    def _act(self, dbase=(0, 0, 0), dq=None, grip=0.0):
        a = np.zeros(11, dtype=np.float32)
        a[0:3] = np.clip(np.asarray(dbase, float), -MAXD, MAXD)
        if dq is not None:
            a[3:10] = np.clip(np.asarray(dq, float), -MAXD, MAXD)
        a[10] = grip
        if self.mode == 1:
            a[3:10] = 0.0
        elif self.mode == 2:
            a[0:3] = 0.0
        return a

    def _drive(self, base, goal, dq=None, grip=0.0):
        g = np.asarray(goal, float)
        if self.detour is not None:
            if abs(base[1] - self.detour) < 0.05:
                self.detour = None
            else:
                g = np.array([base[0], self.detour, g[2]])
        d = np.zeros(3)
        d[0:2] = g[0:2] - base[0:2]
        d[2] = _wrap(g[2] - base[2])
        return self._act(d, dq, grip), (np.max(np.abs(d[0:2])) < 2e-3 and abs(d[2]) < 2e-3
                                        and self.detour is None)

    def _track(self, state):
        r = robot_state(state)
        if self.prev is not None and self.prev_act is not None:
            acted = np.max(np.abs(self.prev_act[:10])) > 1e-6
            moved = not np.allclose(self.prev, r["raw"][:10], atol=1e-6)
            if acted:
                if moved:
                    self.stall = 0
                    self.mode = 0
                else:
                    self.stall += 1
        drive = self.phase in ("drive_out", "drive_back")
        if not drive:
            self.mode = 0                    # isolation modes are drive-only
        if drive and self.stall == 2:
            self.mode = 1
        elif drive and self.stall == 4:
            self.mode = 2
        elif self.stall >= (6 if drive else 3):
            self.mode = 0
            self.stall = 0
            if drive:
                self._recover_drive(r)
            elif self.wp:
                self.wp.pop(0)              # skip a stuck joint waypoint
        self.prev = r["raw"][:10].copy()
        return r

    def _recover_drive(self, r):
        """Base motion refused: settle for the closest reachable pose."""
        self.retreat += 1
        if self.phase == "drive_out":
            # give up the last few cm of approach; grasp from where we are
            self.plan["base"] = r["base"].copy()
            self._replan_from(r)
        else:
            g = np.asarray(self.base_goal, float)
            d = g[:2] - r["base"][:2]
            n = np.linalg.norm(d)
            if n > 1e-6 and self.retreat <= 3:
                self.base_goal = np.array([g[0] - d[0] / n * 0.12,
                                           g[1] - d[1] / n * 0.12, g[2]])
            else:
                self.base_goal = r["base"].copy()

    def _replan_from(self, r):
        """Re-solve the approach path for the base pose we actually reached."""
        pl = self.plan
        d3 = np.array([pl["dirv"][0], pl["dirv"][1], 0.0])
        tgt = pl["block"] - d3 * GRASP_BACK + np.array([0, 0, GRASP_UP])
        base = r["base"].copy()
        base[2] = pl["base"][2]
        p_now, _ = world_fk(r["q"], base)
        pts = _descent_pts(p_now, tgt, np.array([0.0, 0.0, OVER]))
        path = _cart_path(base, r["q"], pts, pl["R"], seeds=8, iters=120, max_jump=0.6)
        self.plan["base"] = base
        self.wp = list(path) if path else []
        self.phase = "approach"

    # -- main --------------------------------------------------------------
    def get_action(self, state):
        r = self._track(state)
        a = self._step(state, r)
        self.prev_act = np.asarray(a, float).copy()
        self.steps += 1
        return np.asarray(a, dtype=np.float32)

    def _step(self, state, r):
        base, q = r["base"], r["q"]

        if self.phase == "drive_out":
            # drive to the grasp base pose while slewing the arm to the pregrasp pose
            dq = dq_wrap(self.plan["q_pre"] - q)
            a, at = self._drive(base, self.plan["base"], dq)
            if at and np.max(np.abs(dq)) < 2e-3:
                self.phase = "approach"
            return a

        if self.phase == "approach":
            while self.wp:
                dq = dq_wrap(self.wp[0] - q)
                if np.max(np.abs(dq)) < 2e-3:
                    self.wp.pop(0)
                    continue
                return self._act(dq=dq)
            self.phase = "close"
            return self._act(grip=-1.0)

        if self.phase == "close":
            if r["holding"]:
                self.phase = "lift"
                self.wp = self._lift_path(base, q)
                return self._act(dq=dq_wrap(self.wp[0] - q) if self.wp else None,
                                 grip=-1.0)
            d3 = np.array([self.plan["dirv"][0], self.plan["dirv"][1], 0.0])
            offs = [np.zeros(3), np.array([0, 0, -0.015]), d3 * 0.02,
                    np.array([0, 0, 0.02]), d3 * 0.02 + np.array([0, 0, -0.02]),
                    -d3 * 0.03, d3 * 0.04]
            if self.c_stage == "grip":                    # squeeze, verify next step
                self.c_stage = "move"
                self.c_try += 1
                return self._act(grip=-1.0)
            tgt = self.plan["tgt"] + offs[self.c_try % len(offs)]
            p, _ = world_fk(q, base)
            err = tgt - p
            if np.linalg.norm(err) < 0.008 or self.stall >= 2:
                self.stall = 0
                self.c_stage = "grip"
                return self._act(grip=-1.0)
            stepv = err if np.linalg.norm(err) < 0.04 else err / np.linalg.norm(err) * 0.04
            path = _cart_path(base, q, [p + stepv], self.plan["R"], seeds=6, iters=110)
            if path is None:
                self.c_stage = "grip"
                return self._act(grip=-1.0)
            if self.debug and self.c_try < 12:
                print("   close try %d tool %s err %.3f" %
                      (self.c_try, np.round(p, 3), np.linalg.norm(err)))
            return self._act(dq=dq_wrap(path[0] - q), grip=1.0)

        if self.phase == "lift":
            while self.wp:
                dq = dq_wrap(self.wp[0] - q)
                if np.max(np.abs(dq)) < 2e-3:
                    self.wp.pop(0)
                    continue
                return self._act(dq=dq, grip=-1.0)
            self.q_hold = q.copy()
            self.base_goal = _release_base(q, PLATE_TGT, base[:2])
            if self.base_goal is None:
                self.base_goal = _release_base(q, PLATE_TGT)
            self.phase = "drive_back"
            return self._step(state, r)

        if self.phase == "drive_back":
            dq = dq_wrap(self.q_hold - q)   # hold the arm still
            a, at = self._drive(base, self.base_goal, dq, grip=-1.0)
            if at:
                self.phase = "release"
                return self._act(grip=1.0)
            return a

        # release
        self.done = True
        return self._act(grip=1.0)

    def _lift_path(self, base, q):
        p, R = world_fk(q, base)
        for n in (2, 3, 4, 6):
            pts = [p + np.array([0, 0, LIFT * (i + 1.0) / n]) for i in range(n)]
            path = _cart_path(base, q, pts, R, max_jump=0.5)
            if path is not None:
                return path
        return []


# ------------------------------------------------------------ declarative --
def plan_far_route(blocks, robot, yaws=None):
    """blocks: {name: xyz} (spares live at x<0); robot: dict/array with base+q.

    Returns a list of waypoint dicts describing the route.
    """
    if isinstance(robot, dict):
        base0, q0 = np.asarray(robot["base"], float), np.asarray(robot["q"], float)
    else:
        robot = np.asarray(robot, float)
        base0, q0 = robot[0:3], robot[3:10]
    spares = {n: np.asarray(p, float) for n, p in blocks.items()
              if n.startswith("green") and n != "green0" and np.asarray(p)[0] < 0.0}
    pl = _plan_pick(spares, q0, base0[:2], yaws=yaws)
    if pl is None:
        return []
    wps = [dict(kind="move", base=pl["base"], q=pl["q_pre"], note="drive to far table")]
    for q in pl["path"]:
        wps.append(dict(kind="move", base=pl["base"], q=q, note="cartesian approach"))
    wps.append(dict(kind="grip", val=-1.0, note="close on " + pl["name"]))
    q_end = pl["path"][-1]
    lift = _cart_path(pl["base"], q_end,
                      [world_fk(q_end, pl["base"])[0] + np.array([0, 0, LIFT])],
                      grasp_R(pl["base"][2]))
    if lift:
        q_end = lift[-1]
        wps.append(dict(kind="move", base=pl["base"], q=q_end, note="lift"))
    rb = _release_base(q_end, PLATE_TGT, pl["base"][:2])
    wps.append(dict(kind="move", base=rb, q=q_end, note="drive to plate"))
    wps.append(dict(kind="grip", val=1.0, note="release over plate"))
    return wps
