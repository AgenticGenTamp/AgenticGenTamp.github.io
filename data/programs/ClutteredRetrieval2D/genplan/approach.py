"""Scripted policy for ClutteredRetrieval2DEnv (variable object count).

=============================================================================
a18 RESULT
=============================================================================
  obstruction5 : (1.9779, 1.6570, th=2.7147) -> (2.0015, 1.6959, th=2.5931)
                 nudged slightly; all other obstructions at initial poses.
  robot        : (1.6929, 1.4744, th=0.6282), arm_joint=0.196, vacuum=1.0
  target_block : UNCHANGED at (2.2507, 1.9765, th=-3.1244)

Better than a17 (which touched nothing): the vacuum is on, the arm is near full
extension, and one obstruction moved. But the robot sits at distance ~0.72 from
the block center with a maximum suction reach of ~0.215 from its base. It is
parked well outside contact range, in the lower-left, and again did not reach
the target.

=============================================================================
WHERE I ACTUALLY STAND, STATED PLAINLY
=============================================================================
Eighteen submissions. Exactly one (a4) ever moved the target block, and I could
not explain or reproduce it. Since then every iteration has oscillated between
two failure families -- "touches nothing" and "touches the wrong thing / drifts
away" -- and each time I have written a confident causal story for the swap.
Reviewing the sequence honestly: a14 diagnosed wrong-object grasping (correct,
and useful), and a16/a17/a18 were me repeatedly mis-tuning an attraction term
and narrating each mis-tune as a discovery. The net progress across the last
five submissions is approximately zero, and the confident framing I put on them
was not warranted.

I also want to correct something I asserted early and built on for many rounds:
I claimed grasped objects "bulldoze" clutter via `move_objects_in_contact`.
That function exists but `ClutteredRetrieval2DEnv` never calls it — it inherits
the base `get_objects_to_move`, which returns an empty set. I stated it as fact
without checking, and several of my transport designs rested on it.

I do not know why the robot cannot close on this block, and I have no way to
find out from final-state snapshots alone. I've asked for a trace three times;
the harness doesn't provide one. Rather than ship a nineteenth guess dressed as
a fix, this submission does the one thing the evidence supports without
requiring a causal theory: it removes the hand-tuned attraction/orbit field
entirely — that field is what I keep getting wrong — and replaces base motion
with a direct greedy descent on a single scalar objective (distance from the
suction point to the nearest block surface point), evaluated over a fixed set
of candidate unit moves each step, with the vacuum always on and the arm always
sweeping. Greedy descent on an observable scalar has no tuning knobs for me to
misjudge, and it cannot park itself at distance 0.72 while a closer move exists.

That is a principled simplification, not a diagnosis. I am not claiming it
solves seed 0.

TERMINATION (verified, unchanged): `is_inside(block, region)` requires ALL FOUR
corners of the 0.14 block inside the 0.21 region; block diagonal 0.198 vs
region side 0.21, so orientation must match mod 90 deg to ~+-0.06 rad.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _sad(target, source):
    return _wrap(target - source)


def _rot(x, y, t):
    c, s = math.cos(t), math.sin(t)
    return (c * x - s * y, s * x + c * y)


class _Pose:
    __slots__ = ("x", "y", "theta")

    def __init__(self, x, y, theta):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    def compose(self, o):
        dx, dy = _rot(o.x, o.y, self.theta)
        return _Pose(self.x + dx, self.y + dy, _wrap(self.theta + o.theta))

    def inverse(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return _Pose(-self.x * c - self.y * s, self.x * s - self.y * c, -self.theta)


def _rect_corners(x, y, th, w, h):
    out = []
    for lx, ly in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h)):
        dx, dy = _rot(lx, ly, th)
        out.append((x + dx, y + dy))
    return out


def _rect_center(x, y, th, w, h):
    dx, dy = _rot(w / 2.0, h / 2.0, th)
    return (x + dx, y + dy)


def _corner_from_center(cx, cy, th, w, h):
    dx, dy = _rot(-w / 2.0, -h / 2.0, th)
    return (cx + dx, cy + dy)


def _in_rect(px, py, x, y, th, w, h, tol=0.0):
    lx, ly = _rot(px - x, py - y, -th)
    return (-tol <= lx <= w + tol) and (-tol <= ly <= h + tol)


def _dist_point_rect(px, py, x, y, th, w, h):
    """Distance from a point to a rectangle (0 inside)."""
    lx, ly = _rot(px - x, py - y, -th)
    dx = max(0.0, max(-lx, lx - w))
    dy = max(0.0, max(-ly, ly - h))
    return math.hypot(dx, dy)


def _seg_point_dist(ax, ay, bx, by, px, py):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


# --------------------------------------------------------------------------


class GeneratedApproach:
    """Retrieve-and-place. Count-agnostic: never indexes obstructions."""

    _SEEK = "seek"
    _SHED = "shed"
    _EXTRACT = "extract"
    _CARRY = "carry"
    _ALIGN = "align"
    _RELEASE = "release"
    _DONE = "done"

    MAX_SHEDS = 4
    SHED_COOLDOWN = 30

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        lo = np.asarray(action_space.low, dtype=np.float64)
        hi = np.asarray(action_space.high, dtype=np.float64)
        self._mdx = float(min(abs(lo[0]), abs(hi[0])))
        self._mdy = float(min(abs(lo[1]), abs(hi[1])))
        self._mdth = float(min(abs(lo[2]), abs(hi[2])))
        self._max_darm = float(hi[3])
        self._min_darm = float(lo[3])

        self._rng = np.random.default_rng(0)
        self._reset_vars()

    # -- bookkeeping --------------------------------------------------------

    def _reset_vars(self):
        self._phase = self._SEEK
        self._t = 0
        self._pt = 0
        self._grasp_T = None
        self._prev_gtb = None
        self._stall = 0
        self._prev_robot = None
        self._extract_start = None
        self._extract_dir = 0.0
        self._rel_timer = 0
        self._block_hold_streak = 0
        self._sweep_dir = 1.0
        self._waypoint = None
        self._slide_sign = 0.0
        self._slide_left = 0

        self._prev_block = None
        self._prev_obs = {}
        self._contact_dwell = 0

        self._shed_tan = (0.0, 0.0)
        self._shed_d0 = None
        self._shed_release = 0
        self._shed_count = 0
        self._shed_cool = 0

        # Greedy-descent bookkeeping.
        self._best_gap = None
        self._no_improve = 0
        self._escape_left = 0
        self._escape_vec = (0.0, 0.0)

    def reset(self, state, info):
        self._reset_vars()
        self._rng = np.random.default_rng(23)

    # -- accessors ----------------------------------------------------------

    def _named(self, state, name):
        try:
            return state.get_object_from_name(name)
        except Exception:
            for o in state:
                if o.name == name:
                    return o
        return None

    def _robot_obj(self, state):
        o = self._named(state, "robot")
        if o is not None and o.type.name == "crv_robot":
            return o
        for o in state:
            if o.type.name == "crv_robot":
                return o
        for o in state:
            if "base_radius" in state.type_features.get(o.type, []):
                return o
        return None

    def _by_type(self, state, tn):
        o = self._named(state, tn)
        if o is not None and o.type.name == tn:
            return o
        for o in state:
            if o.type.name == tn:
                return o
        return None

    def _obstructions(self, state):
        out = []
        for o in state:
            if o.type.name != "rectangle":
                continue
            f = state.type_features.get(o.type, [])
            if "static" in f and state.get(o, "static") > 0.5:
                continue
            out.append(o)
        return out

    def _R(self, state, robot):
        return {
            "x": float(state.get(robot, "x")),
            "y": float(state.get(robot, "y")),
            "theta": float(state.get(robot, "theta")),
            "base_radius": float(state.get(robot, "base_radius")),
            "arm_joint": float(state.get(robot, "arm_joint")),
            "arm_length": float(state.get(robot, "arm_length")),
            "vacuum": float(state.get(robot, "vacuum")),
            "gripper_height": float(state.get(robot, "gripper_height")),
            "gripper_width": float(state.get(robot, "gripper_width")),
        }

    def _B(self, state, o):
        return {
            "x": float(state.get(o, "x")),
            "y": float(state.get(o, "y")),
            "theta": float(state.get(o, "theta")),
            "width": float(state.get(o, "width")),
            "height": float(state.get(o, "height")),
        }

    def _obs_discs(self, state):
        out = []
        for o in self._obstructions(state):
            oi = self._B(state, o)
            oc = _rect_center(oi["x"], oi["y"], oi["theta"], oi["width"], oi["height"])
            rad = 0.5 * math.hypot(oi["width"], oi["height"])
            out.append((oc[0], oc[1], rad))
        return out

    # -- kinematics ---------------------------------------------------------

    def _tip(self, r):
        th = r["theta"]
        d = r["arm_joint"] + r["gripper_width"] / 2.0
        return (r["x"] + math.cos(th) * d, r["y"] + math.sin(th) * d)

    def _grip_pose(self, r):
        tx, ty = self._tip(r)
        return _Pose(tx, ty, r["theta"])

    def _suction_pt(self, r, x=None, y=None, th=None, aj=None):
        """Suction-rectangle center for a hypothetical robot configuration."""
        x = r["x"] if x is None else x
        y = r["y"] if y is None else y
        th = r["theta"] if th is None else th
        aj = r["arm_joint"] if aj is None else aj
        d = aj + 1.5 * r["gripper_width"]
        return (x + math.cos(th) * d, y + math.sin(th) * d)

    def _gtb(self, r, b):
        return self._grip_pose(r).inverse().compose(_Pose(b["x"], b["y"], b["theta"]))

    # -- action -------------------------------------------------------------

    def _act(self, dx=0.0, dy=0.0, dth=0.0, darm=0.0, vac=0.0):
        a = np.array([dx, dy, dth, darm, vac], dtype=np.float32)
        lo = np.asarray(self.action_space.low, dtype=np.float32)
        hi = np.asarray(self.action_space.high, dtype=np.float32)
        if not np.all(np.isfinite(a)):
            a = np.zeros(5, dtype=np.float32)
        eps = np.float32(1e-6)
        return np.clip(a, lo + eps, hi - eps).astype(np.float32)

    def _servo(self, r, tx, ty, tth, gain=4.0, scale=1.0):
        dx = float(np.clip(gain * (tx - r["x"]), -self._mdx, self._mdx)) * scale
        dy = float(np.clip(gain * (ty - r["y"]), -self._mdy, self._mdy)) * scale
        dth = (
            0.0
            if tth is None
            else float(np.clip(2.5 * _sad(tth, r["theta"]), -self._mdth, self._mdth))
        )
        return dx, dy, dth

    # -- motion tracking ----------------------------------------------------

    def _pose_changed(self, cur, prev):
        if prev is None:
            return False
        return (
            abs(cur[0] - prev[0]) > 1e-7
            or abs(cur[1] - prev[1]) > 1e-7
            or abs(_sad(cur[2], prev[2])) > 1e-7
        )

    def _track(self, state, b):
        cur_b = (b["x"], b["y"], b["theta"])
        block_moved = self._pose_changed(cur_b, self._prev_block)
        self._prev_block = cur_b

        moving = []
        new_prev = {}
        for o in self._obstructions(state):
            oi = self._B(state, o)
            cur = (oi["x"], oi["y"], oi["theta"])
            if self._pose_changed(cur, self._prev_obs.get(o.name)):
                oc = _rect_center(
                    oi["x"], oi["y"], oi["theta"], oi["width"], oi["height"]
                )
                moving.append((o.name, oc[0], oc[1]))
            new_prev[o.name] = cur
        self._prev_obs = new_prev
        return block_moved, moving

    def _holding_block(self, r, b, block_moved):
        """Confirmed only if the BLOCK moves with a frozen gripper transform."""
        if r["vacuum"] <= 0.5:
            self._prev_gtb = None
            self._block_hold_streak = 0
            return False
        cur = self._gtb(r, b)
        prev = self._prev_gtb
        self._prev_gtb = cur
        if math.hypot(cur.x, cur.y) >= 0.5:
            self._block_hold_streak = 0
            return False
        if prev is None:
            return self._block_hold_streak >= 2
        frozen = (
            math.hypot(cur.x - prev.x, cur.y - prev.y) < 1e-4
            and abs(_sad(cur.theta, prev.theta)) < 1e-4
        )
        if frozen and block_moved:
            self._block_hold_streak += 1
        elif not frozen:
            self._block_hold_streak = 0
        return self._block_hold_streak >= 2

    def _wrong_object(self, r, block_moved, moving_obs):
        if self._shed_cool > 0 or self._shed_count >= self.MAX_SHEDS:
            return False
        if r["vacuum"] <= 0.5 or block_moved or self._block_hold_streak >= 2:
            return False
        if not moving_obs:
            return False
        tx, ty = self._tip(r)
        for (_n, ox, oy) in moving_obs:
            if math.hypot(ox - tx, oy - ty) < 0.22:
                return True
        return False

    # -- goal ---------------------------------------------------------------

    def _inside(self, b, g, tol=0.0):
        for (px, py) in _rect_corners(b["x"], b["y"], b["theta"], b["width"], b["height"]):
            if not _in_rect(px, py, g["x"], g["y"], g["theta"], g["width"], g["height"], tol):
                return False
        return True

    def _margin(self, b, g):
        worst = 1e9
        for (px, py) in _rect_corners(b["x"], b["y"], b["theta"], b["width"], b["height"]):
            lx, ly = _rot(px - g["x"], py - g["y"], -g["theta"])
            worst = min(worst, lx, g["width"] - lx, ly, g["height"] - ly)
        return worst

    def _align_theta(self, bth, gth):
        best, be = gth, None
        for k in range(4):
            c = _wrap(gth + k * (math.pi / 2.0))
            e = abs(_sad(c, bth))
            if be is None or e < be:
                best, be = c, e
        return best

    # -- stall / slide ------------------------------------------------------

    def _upd_stall(self, r):
        cur = (r["x"], r["y"], r["theta"], r["arm_joint"])
        p = self._prev_robot
        if p is not None:
            m = (
                abs(cur[0] - p[0])
                + abs(cur[1] - p[1])
                + abs(_sad(cur[2], p[2]))
                + abs(cur[3] - p[3])
            )
            if m < 1e-7:
                self._stall += 1
            else:
                self._stall = max(0, self._stall - 2)
        self._prev_robot = cur

    def _slide(self, dx, dy):
        L = math.hypot(dx, dy)
        if L < 1e-9:
            a = float(self._rng.uniform(-math.pi, math.pi))
            dx, dy = math.cos(a), math.sin(a)
            L = 1.0
        ux, uy = dx / L, dy / L
        if self._slide_left <= 0:
            self._slide_sign = 1.0 if self._slide_sign <= 0 else -1.0
            self._slide_left = int(self._rng.integers(14, 30))
        self._slide_left -= 1
        s = self._slide_sign
        tx, ty = -uy * s, ux * s
        mx, my = 0.85 * tx + 0.35 * ux, 0.85 * ty + 0.35 * uy
        n = math.hypot(mx, my) + 1e-9
        return (self._mdx * mx / n, self._mdy * my / n)

    # -- main ---------------------------------------------------------------

    def get_action(self, state):
        self._t += 1
        self._pt += 1
        if self._shed_cool > 0:
            self._shed_cool -= 1

        robot = self._robot_obj(state)
        block = self._by_type(state, "target_block")
        region = self._by_type(state, "target_region")
        if robot is None or block is None or region is None:
            return self._act()

        r = self._R(state, robot)
        b = self._B(state, block)
        g = self._B(state, region)
        bcx, bcy = _rect_center(b["x"], b["y"], b["theta"], b["width"], b["height"])
        gcx, gcy = _rect_center(g["x"], g["y"], g["theta"], g["width"], g["height"])

        block_moved, moving_obs = self._track(state, b)
        held = self._holding_block(r, b, block_moved)
        self._upd_stall(r)
        if held:
            self._grasp_T = self._gtb(r, b)

        if self._inside(b, g):
            if self._phase not in (self._RELEASE, self._DONE):
                self._phase = self._RELEASE
                self._pt = 0
            return self._act(vac=0.0)

        dist_to_block = math.hypot(bcx - r["x"], bcy - r["y"])

        if self._phase == self._SEEK and not held:
            if self._wrong_object(r, block_moved, moving_obs):
                self._phase = self._SHED
                self._pt = 0
                self._shed_release = 0
                self._shed_count += 1
                self._shed_d0 = dist_to_block
                vx, vy = r["x"] - bcx, r["y"] - bcy
                L = math.hypot(vx, vy) + 1e-9
                ux, uy = vx / L, vy / L
                s = 1.0 if (self._shed_count % 2 == 0) else -1.0
                self._shed_tan = (-uy * s, ux * s)

        if self._phase in (self._EXTRACT, self._CARRY, self._ALIGN) and not held:
            self._phase = self._SEEK
            self._pt = 0
            self._grasp_T = None
            self._prev_gtb = None
            self._contact_dwell = 0

        if self._phase == self._SEEK:
            return self._do_seek(state, r, b, bcx, bcy, held, block_moved)
        if self._phase == self._SHED:
            return self._do_shed(r, bcx, bcy, dist_to_block)
        if self._phase == self._EXTRACT:
            return self._do_extract(r)
        if self._phase == self._CARRY:
            return self._do_carry(state, r, b, g, gcx, gcy)
        if self._phase == self._ALIGN:
            return self._do_align(r, b, g, gcx, gcy)
        if self._phase == self._RELEASE:
            return self._do_release(r, b, g)
        return self._act(vac=0.0)

    # -- SEEK: greedy descent on suction-to-block gap -----------------------

    def _gap(self, r, b, x, y, th, aj):
        sx, sy = self._suction_pt(r, x, y, th, aj)
        return _dist_point_rect(sx, sy, b["x"], b["y"], b["theta"], b["width"], b["height"])

    def _do_seek(self, state, r, b, bcx, bcy, held, block_moved):
        """No tuned attraction field. Each step, evaluate a fixed set of unit
        moves and take the one that most reduces the suction-to-block-surface
        distance. Vacuum always on; arm always sweeping."""
        if held:
            self._grasp_T = self._gtb(r, b)
            self._phase = self._EXTRACT
            self._pt = 0
            self._stall = 0
            self._extract_start = (r["x"], r["y"])
            vx, vy = r["x"] - bcx, r["y"] - bcy
            self._extract_dir = math.atan2(vy, vx)
            return self._act(vac=1.0)

        if block_moved:
            self._contact_dwell = 8
        if self._contact_dwell > 0:
            self._contact_dwell -= 1
            to_block = math.atan2(bcy - r["y"], bcx - r["x"])
            dth = float(
                np.clip(2.5 * _sad(to_block, r["theta"]), -self._mdth, self._mdth)
            )
            return self._act(dth=dth * 0.2, darm=0.02 * self._max_darm, vac=1.0)

        cur_gap = self._gap(r, b, r["x"], r["y"], r["theta"], r["arm_joint"])
        if self._best_gap is None or cur_gap < self._best_gap - 1e-4:
            self._best_gap = cur_gap
            self._no_improve = 0
        else:
            self._no_improve += 1

        # Escape a plateau with a committed random push (rare).
        if self._escape_left > 0:
            self._escape_left -= 1
            ex, ey = self._escape_vec
            return self._act(
                dx=ex * self._mdx, dy=ey * self._mdy,
                darm=0.3 * self._max_darm, vac=1.0,
            )
        if self._no_improve > 90 or self._stall > 25:
            self._no_improve = 0
            self._stall = 0
            self._best_gap = None
            a = float(self._rng.uniform(-math.pi, math.pi))
            self._escape_vec = (math.cos(a), math.sin(a))
            self._escape_left = int(self._rng.integers(20, 45))
            return self._act(vac=1.0)

        # Candidate unit moves: 16 base directions x {rotate -,0,+} x arm sweep.
        best = None
        best_val = None
        for k in range(16):
            a = -math.pi + k * (2.0 * math.pi / 16)
            cx, cy = math.cos(a), math.sin(a)
            for rot in (-1.0, 0.0, 1.0):
                for aj_d in (-1.0, 0.0, 1.0):
                    nx = r["x"] + cx * self._mdx
                    ny = r["y"] + cy * self._mdy
                    nth = _wrap(r["theta"] + rot * self._mdth)
                    naj = float(
                        np.clip(
                            r["arm_joint"] + aj_d * self._max_darm,
                            r["base_radius"],
                            r["arm_length"],
                        )
                    )
                    val = self._gap(r, b, nx, ny, nth, naj)
                    if best_val is None or val < best_val:
                        best_val = val
                        best = (cx, cy, rot, aj_d)

        if best is None:
            return self._act(vac=1.0)

        cx, cy, rot, aj_d = best
        dx = cx * self._mdx
        dy = cy * self._mdy
        dth = rot * self._mdth
        darm = aj_d * self._max_darm

        # Keep the arm alive even when descent prefers holding it still.
        if abs(darm) < 1e-9:
            if r["arm_joint"] >= r["arm_length"] - 1e-6:
                self._sweep_dir = -1.0
            elif r["arm_joint"] <= r["base_radius"] + 1e-6:
                self._sweep_dir = 1.0
            darm = self._sweep_dir * 0.15 * self._max_darm

        if self._stall > 5:
            sx, sy = self._slide(dx, dy)
            return self._act(dx=sx, dy=sy, dth=dth, darm=darm, vac=1.0)

        return self._act(dx=dx, dy=dy, dth=dth, darm=darm, vac=1.0)

    # -- SHED ---------------------------------------------------------------

    def _do_shed(self, r, bcx, bcy, dist):
        if self._shed_release > 0:
            self._shed_release -= 1
            if self._shed_release == 0:
                self._phase = self._SEEK
                self._pt = 0
                self._prev_gtb = None
                self._block_hold_streak = 0
                self._contact_dwell = 0
                self._shed_cool = self.SHED_COOLDOWN
                self._best_gap = None
                self._no_improve = 0
            vx, vy = bcx - r["x"], bcy - r["y"]
            L = math.hypot(vx, vy) + 1e-9
            return self._act(
                dx=(vx / L) * self._mdx * 0.3,
                dy=(vy / L) * self._mdy * 0.3,
                vac=0.0,
            )

        if self._shed_d0 is not None and dist > self._shed_d0 + 0.04:
            self._shed_release = 3
            return self._act(vac=0.0)
        if self._pt > 35:
            self._shed_release = 3
            return self._act(vac=0.0)

        tx, ty = self._shed_tan
        n = math.hypot(tx, ty) + 1e-9
        dx = self._mdx * tx / n
        dy = self._mdy * ty / n
        vx, vy = bcx - r["x"], bcy - r["y"]
        L = math.hypot(vx, vy) + 1e-9
        dx += 0.20 * self._mdx * vx / L
        dy += 0.20 * self._mdy * vy / L

        if self._stall > 4:
            sx, sy = self._slide(dx, dy)
            return self._act(dx=sx, dy=sy, vac=1.0)
        return self._act(dx=dx, dy=dy, vac=1.0)

    # -- EXTRACT ------------------------------------------------------------

    def _do_extract(self, r):
        ang = self._extract_dir
        dx = math.cos(ang) * self._mdx
        dy = math.sin(ang) * self._mdy
        darm = self._min_darm * 0.4 if r["arm_joint"] > r["base_radius"] + 1e-6 else 0.0

        travelled = 0.0
        if self._extract_start is not None:
            travelled = math.hypot(
                r["x"] - self._extract_start[0], r["y"] - self._extract_start[1]
            )
        if travelled > 0.22 or self._pt > 60:
            self._phase = self._CARRY
            self._pt = 0
            self._stall = 0
            self._waypoint = None
            return self._act(vac=1.0)

        if self._stall > 4:
            sx, sy = self._slide(dx, dy)
            return self._act(dx=sx, dy=sy, darm=darm, vac=1.0)
        return self._act(dx=dx, dy=dy, darm=darm, vac=1.0)

    # -- base pose for a desired block pose --------------------------------

    def _base_for_block(self, r, b, cx, cy, bth):
        T = self._grasp_T
        if T is None:
            return (r["x"], r["y"], r["theta"])
        bx, by = _corner_from_center(cx, cy, bth, b["width"], b["height"])
        wgrip = _Pose(bx, by, bth).compose(T.inverse())
        reach = r["arm_joint"] + r["gripper_width"] / 2.0
        th = wgrip.theta
        return (wgrip.x - math.cos(th) * reach, wgrip.y - math.sin(th) * reach, th)

    # -- waypoint -----------------------------------------------------------

    def _pick_waypoint(self, state, r, gcx, gcy):
        discs = self._obs_discs(state)
        br = r["base_radius"]
        best, best_c = None, None
        for k in range(48):
            ang = -math.pi + k * (2 * math.pi / 48)
            for rad in (0.35, 0.55, 0.8):
                px = r["x"] + math.cos(ang) * rad
                py = r["y"] + math.sin(ang) * rad
                if not (0.15 < px < 2.35 and 0.15 < py < 2.35):
                    continue
                clear = 1e9
                for (ox, oy, orad) in discs:
                    clear = min(clear, math.hypot(ox - px, oy - py) - orad - br)
                    clear = min(
                        clear,
                        _seg_point_dist(r["x"], r["y"], px, py, ox, oy) - orad - br,
                    )
                if clear < 0.02:
                    continue
                c = math.hypot(px - gcx, py - gcy) - 0.6 * min(clear, 0.3)
                if best_c is None or c < best_c:
                    best_c, best = c, (px, py)
        return best

    # -- CARRY --------------------------------------------------------------

    def _do_carry(self, state, r, b, g, gcx, gcy):
        bcx, bcy = _rect_center(b["x"], b["y"], b["theta"], b["width"], b["height"])
        err = math.hypot(gcx - bcx, gcy - bcy)

        if err < 0.12 or self._pt > 700:
            self._phase = self._ALIGN
            self._pt = 0
            self._stall = 0
            return self._act(vac=1.0)

        darm = self._min_darm * 0.3 if r["arm_joint"] > r["base_radius"] + 1e-6 else 0.0

        if self._waypoint is None or self._pt % 60 == 0 or self._stall > 18:
            self._waypoint = self._pick_waypoint(state, r, gcx, gcy)
        wp = self._waypoint

        des_th = self._align_theta(b["theta"], g["theta"])
        tbx, tby, tth = self._base_for_block(r, b, gcx, gcy, des_th)

        if wp is not None and math.hypot(wp[0] - r["x"], wp[1] - r["y"]) > 0.05:
            tx, ty = wp
        else:
            tx, ty = tbx, tby
            self._waypoint = None

        dx, dy, dth = self._servo(r, tx, ty, tth, gain=3.0)
        dth *= 0.4

        if self._stall > 4:
            sx, sy = self._slide(dx, dy)
            return self._act(dx=sx, dy=sy, dth=dth * 0.5, darm=darm, vac=1.0)

        return self._act(dx=dx, dy=dy, dth=dth, darm=darm, vac=1.0)

    # -- ALIGN --------------------------------------------------------------

    def _do_align(self, r, b, g, gcx, gcy):
        bcx, bcy = _rect_center(b["x"], b["y"], b["theta"], b["width"], b["height"])
        des_th = self._align_theta(b["theta"], g["theta"])
        th_err = _sad(des_th, b["theta"])

        if self._margin(b, g) > 0.002:
            self._phase = self._RELEASE
            self._pt = 0
            return self._act(vac=1.0)

        tx, ty, tth = self._base_for_block(r, b, gcx, gcy, des_th)

        # Orientation first: block diagonal 0.198 vs region side 0.21.
        if abs(th_err) > 0.04:
            dx, dy, dth = self._servo(r, tx, ty, tth, gain=2.0, scale=0.6)
            if self._stall > 6:
                sx, sy = self._slide(dx, dy)
                return self._act(dx=0.5 * sx, dy=0.5 * sy, dth=dth, vac=1.0)
            return self._act(dx=dx, dy=dy, dth=dth, vac=1.0)

        dx, dy, dth = self._servo(r, tx, ty, tth, gain=1.6, scale=0.5)
        darm = 0.0
        ex, ey = gcx - bcx, gcy - bcy
        ux, uy = math.cos(r["theta"]), math.sin(r["theta"])
        radial = ex * ux + ey * uy
        if abs(radial) > 0.002:
            darm = float(np.clip(2.0 * radial, self._min_darm, self._max_darm)) * 0.5

        if self._stall > 8:
            sx, sy = self._slide(dx, dy)
            return self._act(dx=0.4 * sx, dy=0.4 * sy, dth=dth, vac=1.0)

        if self._pt > 400:
            self._phase = self._CARRY
            self._pt = 0
            self._waypoint = None
            return self._act(vac=1.0)

        return self._act(dx=dx, dy=dy, dth=dth, darm=darm, vac=1.0)

    # -- RELEASE ------------------------------------------------------------

    def _do_release(self, r, b, g):
        if not self._inside(b, g):
            self._rel_timer += 1
            if self._rel_timer > 3:
                self._rel_timer = 0
                self._phase = self._ALIGN
                self._pt = 0
            return self._act(vac=1.0)
        self._rel_timer += 1
        if self._rel_timer > 2:
            self._phase = self._DONE
        return self._act(vac=0.0)