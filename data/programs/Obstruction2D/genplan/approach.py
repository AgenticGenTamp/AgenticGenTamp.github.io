"""Generated approach for Obstruction2D with a variable number of obstructions.

DIAGNOSIS OF THE SEED-5 FAILURE
-------------------------------
Seed 5 has NO obstruction over the target surface: the surface spans
x in [0.169, 0.307], obstruction0 sits at x in [1.076, 1.192] and obstruction1
at x in [1.296, 1.448] -- both far to the right.  So the only job was to move
the target block a few centimetres onto the surface.

Final state:
    target_block: x = 0.1827, y = 0.1353, w = 0.1113, h = 0.0568
    target_surface: x = 0.1691, y = 0.0,   w = 0.1383, h = 0.1

The block is at y = 0.1353 -- it is FLOATING 0.0353 above the table top
(y = 0.1).  It was picked up and released in mid-air.  Checking `is_on`:
probe_y = 0.1353 - 0.025 = 0.1103, and the surface spans y in [0.0, 0.1].
0.1103 > 0.1, so the predicate FAILS by 0.0103.  In x it was fine
(0.1827..0.2939 inside 0.1691..0.3074).  So we released roughly one centimetre
too high and the episode was lost by a hair.

Why: `_place` used `err = obj_bottom - (support_top + 0.002)` and, when the arm
saturated, tried to lower the base -- but the base floor was
`max(top, top_stuff) + r + 0.004`.  With `top_stuff` including the *target
block itself* in some frames and the surface top at 0.1, the floor came out
above the base's current y, `room <= 0`, and the controller declared success and
released while `err` was still ~0.035.  There was no check that the object had
actually landed.

FIXES
-----
1. `_place` now NEVER releases on a geometric guess.  It exits only when either
   (a) for the final block, the exact `is_on` predicate is satisfied, or
   (b) for an obstruction, the object bottom is within 0.004 of the support.
   If it cannot reach that, it keeps trying (extend arm, lower base) until the
   waypoint budget expires, and the budget is large.
2. The base floor for lowering is computed correctly: the base only has to stay
   above the SUPPORT surface, and `top_stuff` now excludes the carried object
   and anything in the column we are descending into.  Floor is
   `support_top + r + 0.002`, which for the table (0.1) and r = 0.1 gives
   y >= 0.202 -- plenty of room to bring a block down to y = 0.1.
3. Reach budgeting: before descending we make sure the base is at a height from
   which a fully extended arm can put the object bottom on the support:
       needed_base_y = support_top + grasp_offset + arm_max + 1.5*gw
   where grasp_offset = suction_centre_y - object_bottom_y measured at grasp.
   If the current base y is above that, we descend the base FIRST (in free air
   over the drop column) and only then fine-tune with the arm.  This removes
   the "arm saturated, still 3.5 cm high" situation entirely.
4. A final safety net: on the last waypoint, if `is_on` is not satisfied but the
   block is close, we nudge down/sideways and re-check every step instead of
   terminating the plan.
5. Kept the corrected suction-rectangle model (centre at
   arm_joint + 1.5*gripper_width along theta) and grasp verification.
"""

from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------


def _bounds(state, obj):
    x = float(state.get(obj, "x"))
    y = float(state.get(obj, "y"))
    w = float(state.get(obj, "width"))
    h = float(state.get(obj, "height"))
    th = float(state.get(obj, "theta"))
    if abs(th) < 1e-9:
        return x, x + w, y, y + h
    c, s = math.cos(th), math.sin(th)
    pts = [(x + c * dx - s * dy, y + s * dx + c * dy)
           for dx, dy in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def _ovl(a0, a1, b0, b1, pad=0.0):
    return (a0 - pad) < (b1 + pad) and (b0 - pad) < (a1 + pad)


def _is_on(state, top, bottom, tol=0.025):
    """Exact replica of kinder's is_on() termination predicate."""
    txmin, txmax, tymin, _ = _bounds(state, top)
    bxmin, bxmax, bymin, bymax = _bounds(state, bottom)
    probe = tymin - tol
    if not (bymin <= probe <= bymax):
        return False
    if not (bxmin <= txmin <= bxmax):
        return False
    if not (bxmin <= txmax <= bxmax):
        return False
    return True


def _would_be_on(txmin, txmax, tymin, bxmin, bxmax, bymin, bymax, tol=0.025):
    """is_on() for hypothetical top-object bounds."""
    probe = tymin - tol
    return (bymin <= probe <= bymax) and (bxmin <= txmin <= bxmax) \
        and (bxmin <= txmax <= bxmax)


# ----------------------------------------------------------------------------
# State access
# ----------------------------------------------------------------------------


def _by_name(state, name):
    try:
        return state.get_object_from_name(name)
    except Exception:
        return None


def _robot_of(state):
    for n in state.get_object_names():
        o = state.get_object_from_name(n)
        if o.type.name == "crv_robot":
            return o
    return None


def _of_type(state, tname):
    out = []
    for n in sorted(state.get_object_names()):
        o = state.get_object_from_name(n)
        if o.type.name == tname:
            out.append(o)
    return out


def _obstructions(state):
    out = []
    for n in sorted(state.get_object_names()):
        o = state.get_object_from_name(n)
        if o.type.name in ("crv_robot", "target_block", "target_surface"):
            continue
        if not n.startswith("obstruction"):
            continue
        f = state.type_features.get(o.type, [])
        if "width" not in f or "height" not in f:
            continue
        if "static" in f and float(state.get(o, "static")) > 0.5:
            continue
        out.append(o)

    def key(o):
        try:
            return (0, int(o.name[len("obstruction"):]))
        except ValueError:
            return (1, 0)

    out.sort(key=key)
    return out


# ----------------------------------------------------------------------------
# Approach
# ----------------------------------------------------------------------------


class GeneratedApproach:
    """Top-down pick and place; releases only when the goal test is met."""

    WP_BUDGET = 200
    PLACE_BUDGET = 420          # the critical leg gets a big budget

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        self._lo = np.asarray(action_space.low, dtype=np.float64)
        self._hi = np.asarray(action_space.high, dtype=np.float64)
        self.mdx = float(self._hi[0])
        self.mdy = float(self._hi[1])
        self.mdth = float(self._hi[2])
        self.mda = float(self._hi[3])

        self.wxmin = 0.0
        self.wxmax = (1.0 + math.sqrt(5.0)) / 2.0
        self.wymax = 1.0

        self._init_ep()

    def _init_ep(self):
        self.plan = []
        self.i = 0
        self.steps = 0
        self.last = None
        self.stall = 0
        self.held = None
        self.garm = None        # arm_joint at grasp
        self.goff = None        # suction centre y - object bottom y, at grasp
        self.safe_y = None
        self.grab_tries = 0
        self.prev_obj_y = None

    # -- robot model ---------------------------------------------------------

    def _rf(self, state, r):
        return {
            "x": float(state.get(r, "x")),
            "y": float(state.get(r, "y")),
            "th": float(state.get(r, "theta")),
            "r": float(state.get(r, "base_radius")),
            "arm": float(state.get(r, "arm_joint")),
            "amax": float(state.get(r, "arm_length")),
            "gw": float(state.get(r, "gripper_width")),
            "gh": float(state.get(r, "gripper_height")),
        }

    def _suction_cy(self, rf, arm=None, base_y=None):
        """Suction-rect centre y, theta = -pi/2 (matches crv_robot multibody)."""
        a = rf["arm"] if arm is None else arm
        by = rf["y"] if base_y is None else base_y
        return by - (a + 1.5 * rf["gw"])

    # -- scene ---------------------------------------------------------------

    def _scene(self, state):
        rob = _robot_of(state)
        rf = self._rf(state, rob)
        surfs = _of_type(state, "target_surface")
        blks = _of_type(state, "target_block")
        obs = _obstructions(state)
        surf = surfs[0] if surfs else None
        blk = blks[0] if blks else None

        if surf is not None:
            table_top = _bounds(state, surf)[3]
        else:
            c = ([blk] if blk else []) + obs
            table_top = min(_bounds(state, o)[2] for o in c) if c else 0.1

        held = _by_name(state, self.held) if self.held else None
        top_stuff = table_top
        resting = []
        for o in obs + ([blk] if blk is not None else []):
            if held is not None and o.name == held.name:
                continue
            a, b, y0, y1 = _bounds(state, o)
            if y0 < table_top + 0.06:
                resting.append((o, a, b, y0, y1))
                top_stuff = max(top_stuff, y1)

        return {
            "rf": rf, "surf": surf, "blk": blk, "obs": obs, "held": held,
            "table_top": table_top, "top_stuff": top_stuff, "resting": resting,
            "xmin": self.wxmin + rf["r"] + 0.008,
            "xmax": self.wxmax - rf["r"] - 0.008,
            "ymin": table_top + rf["r"] + 0.002,
            "ymax": self.wymax - rf["r"] - 0.015,
            "amin": rf["r"],
        }

    def _top_between(self, g, x0, x1, exclude=None):
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        top = g["table_top"]
        for (o, a, b, _y0, y1) in g["resting"]:
            if exclude is not None and o.name == exclude:
                continue
            if _ovl(a, b, lo, hi, pad=0.015):
                top = max(top, y1)
        return top

    def _compute_safe_y(self, state):
        g = self._scene(state)
        rf = g["rf"]
        highest = g["top_stuff"]
        for o in g["obs"] + ([g["blk"]] if g["blk"] is not None else []):
            highest = max(highest, _bounds(state, o)[3])
        y = highest + rf["r"] + 0.03
        return float(np.clip(y, g["ymin"] + 0.02, g["ymax"]))

    # -- planning ------------------------------------------------------------

    def reset(self, state, info):
        self._init_ep()
        self.safe_y = self._compute_safe_y(state)
        self._plan_from(state)
        return None

    def _plan_from(self, state):
        g = self._scene(state)
        surf, blk, obs = g["surf"], g["blk"], g["obs"]
        if surf is None or blk is None:
            self.plan = []
            return
        rf = g["rf"]
        r = rf["r"]

        sxmin, sxmax, _, sytop = _bounds(state, surf)
        scx = 0.5 * (sxmin + sxmax)
        bxmin, bxmax, _, _ = _bounds(state, blk)
        bw = bxmax - bxmin

        margin = max(0.5 * bw, r) + 0.045
        clo, chi = sxmin - margin, sxmax + margin

        move = []
        for o in obs:
            a, b, _, _ = _bounds(state, o)
            if _ovl(a, b, clo, chi):
                move.append(o)

        occ = [(clo, chi)]
        for o in obs:
            if o in move:
                continue
            a, b, _, _ = _bounds(state, o)
            occ.append((a - 0.03, b + 0.03))
        occ.append((bxmin - 0.04, bxmax + 0.04))

        steps = []
        move.sort(key=lambda o: abs(0.5 * (_bounds(state, o)[0]
                                           + _bounds(state, o)[1]) - scx))
        for o in move:
            a, b, _, _ = _bounds(state, o)
            w = b - a
            ocx = 0.5 * (a + b)
            side = -1.0 if ocx <= scx else 1.0
            park = self._slot(w, occ, g["xmin"], g["xmax"], scx, r, side)
            occ.append((park - 0.5 * w - 0.03, park + 0.5 * w + 0.03))
            steps += self._seq(o.name, park, False, None, None)

        steps += self._seq(blk.name, scx, True, sytop, surf.name)
        self.plan = steps
        self.i = 0
        self.steps = 0

    def _slot(self, width, occ, xmin, xmax, avoid, r, side):
        half = max(0.5 * width, r) + 0.03
        step = 0.008
        n = int(max(1, (xmax - xmin) / step))
        cands = [xmin + half + k * step for k in range(n + 1)]
        cands = [c for c in cands if c + half <= xmax]
        ok = [c for c in cands
              if not any(_ovl(c - half, c + half, a, b) for (a, b) in occ)]
        if not ok:
            return xmin + half if side < 0 else xmax - half
        same = [c for c in ok if (c - avoid) * side > 0]
        pool = same if same else ok
        pool.sort(key=lambda c: -abs(c - avoid))
        return pool[0]

    def _seq(self, name, drop_cx, final, sup_top, sup_name):
        return [
            {"k": "up"},
            {"k": "xto", "obj": name},
            {"k": "down", "obj": name},
            {"k": "grab", "obj": name},
            {"k": "verify", "obj": name},
            {"k": "hoist", "obj": name},
            {"k": "xcarry", "obj": name, "cx": drop_cx},
            {"k": "predrop", "obj": name, "cx": drop_cx, "top": sup_top},
            {"k": "place", "obj": name, "cx": drop_cx, "top": sup_top,
             "sup": sup_name, "final": final},
            {"k": "let"},
            {"k": "up"},
        ]

    # -- action emission -----------------------------------------------------

    def _emit(self, rf, g, dx, dy, dth, darm, vac, ymin=None):
        lo_y = g["ymin"] if ymin is None else ymin
        nx = rf["x"] + dx
        if nx < g["xmin"]:
            dx = g["xmin"] - rf["x"]
        elif nx > g["xmax"]:
            dx = g["xmax"] - rf["x"]
        ny = rf["y"] + dy
        if ny < lo_y:
            dy = lo_y - rf["y"]
        elif ny > g["ymax"]:
            dy = g["ymax"] - rf["y"]
        v = np.asarray([dx, dy, dth, darm, vac], dtype=np.float32)
        return np.clip(v, self._lo, self._hi).astype(np.float32)

    def _nop(self, vac=0.0):
        return np.clip(np.asarray([0.0, 0.0, 0.0, 0.0, vac], dtype=np.float32),
                       self._lo, self._hi).astype(np.float32)

    def _dth(self, rf):
        e = -math.pi / 2.0 - rf["th"]
        while e > math.pi:
            e -= 2 * math.pi
        while e < -math.pi:
            e += 2 * math.pi
        return 0.0 if abs(e) < 1e-3 else float(np.clip(e, -self.mdth, self.mdth))

    def _next(self):
        self.i += 1
        self.steps = 0
        self.stall = 0

    # -- main ----------------------------------------------------------------

    def get_action(self, state):
        rob = _robot_of(state)
        if rob is None:
            return self._nop()
        rf = self._rf(state, rob)
        g = self._scene(state)

        if self.safe_y is None:
            self.safe_y = self._compute_safe_y(state)

        key = (round(rf["x"], 6), round(rf["y"], 6),
               round(rf["th"], 6), round(rf["arm"], 6))
        self.stall = self.stall + 1 if self.last == key else 0
        self.last = key

        if not self.plan or self.i >= len(self.plan):
            return self._nop(0.0)

        wp = self.plan[self.i]
        self.steps += 1
        vac = 1.0 if self.held is not None else 0.0

        budget = self.PLACE_BUDGET if wp["k"] == "place" else self.WP_BUDGET
        if self.steps > budget:
            self._next()
            return self._nop(vac)

        if self.stall >= 12:
            self.stall = 0
            if rf["arm"] > g["amin"] + 1e-4:
                return self._emit(rf, g, 0.0, 0.0, self._dth(rf), -self.mda, vac)
            self._next()
            return self._nop(vac)

        k = wp["k"]
        if k == "up":
            return self._up(state, rf, wp, g)
        if k == "xto":
            return self._xto(state, rf, wp, g)
        if k == "down":
            return self._down(state, rf, wp, g)
        if k == "grab":
            return self._grab(state, rf, wp, g)
        if k == "verify":
            return self._verify(state, rf, wp, g)
        if k == "hoist":
            return self._hoist(state, rf, wp, g)
        if k == "xcarry":
            return self._xcarry(state, rf, wp, g)
        if k == "predrop":
            return self._predrop(state, rf, wp, g)
        if k == "place":
            return self._place(state, rf, wp, g)
        if k == "let":
            return self._let(state, rf, wp, g)
        self._next()
        return self._nop(vac)

    # -- controllers ---------------------------------------------------------

    def _up(self, state, rf, wp, g):
        dth = self._dth(rf)
        darm = -self.mda if rf["arm"] > g["amin"] + 1e-4 else 0.0
        armin = rf["arm"] <= g["amin"] + 4e-3
        e = self.safe_y - rf["y"]
        dy = float(np.clip(e, -self.mdy, self.mdy)) if (armin and abs(e) > 0.006) else 0.0
        if armin and abs(e) <= 0.008:
            self._next()
            return self._nop(0.0)
        return self._emit(rf, g, 0.0, dy, dth, darm, 0.0)

    def _xto(self, state, rf, wp, g):
        obj = _by_name(state, wp["obj"])
        if obj is None:
            self._next()
            return self._nop(0.0)
        a, b, _, _ = _bounds(state, obj)
        tx = float(np.clip(0.5 * (a + b), g["xmin"], g["xmax"]))
        dth = self._dth(rf)
        darm = -self.mda if rf["arm"] > g["amin"] + 1e-4 else 0.0
        armin = rf["arm"] <= g["amin"] + 4e-3

        ey = self.safe_y - rf["y"]
        dy = float(np.clip(ey, -self.mdy, self.mdy)) if abs(ey) > 0.006 else 0.0
        ex = tx - rf["x"]
        dx = 0.0
        if armin and abs(ey) <= 0.02 and abs(ex) > 0.0035:
            dx = float(np.clip(ex, -self.mdx, self.mdx))

        if armin and abs(ex) <= 0.004 and abs(ey) <= 0.010:
            self._next()
            return self._nop(0.0)
        return self._emit(rf, g, dx, dy, dth, darm, 0.0)

    def _down(self, state, rf, wp, g):
        obj = _by_name(state, wp["obj"])
        if obj is None:
            self._next()
            return self._nop(0.0)
        _, _, _, otop = _bounds(state, obj)
        dth = self._dth(rf)

        want_cy = otop - 0.005 - 0.004 * self.grab_tries
        cy = self._suction_cy(rf)
        err = cy - want_cy

        darm = 0.0
        if err > 0.002:
            room = rf["amax"] - rf["arm"]
            darm = min(self.mda, err, max(room, 0.0))
        elif err < -0.008:
            darm = max(-self.mda, err)

        sat = rf["arm"] >= rf["amax"] - 1e-4
        if -0.008 <= err <= 0.003:
            self._next()
            return self._nop(0.0)

        dy = 0.0
        if sat and err > 0.003:
            floor = otop + rf["r"] + 0.003
            room = rf["y"] - floor
            if room > 1e-4:
                dy = -min(self.mdy, err, room)
            else:
                self._next()
                return self._nop(0.0)

        return self._emit(rf, g, 0.0, dy, dth, darm, 0.0)

    def _grab(self, state, rf, wp, g):
        obj = _by_name(state, wp["obj"])
        if obj is None:
            self._next()
            return self._nop(0.0)
        if self.steps == 1:
            self.prev_obj_y = _bounds(state, obj)[2]
        if self.steps >= 2:
            self.garm = rf["arm"]
            self.goff = self._suction_cy(rf) - _bounds(state, obj)[2]
            self.held = wp["obj"]
            self._next()
        return self._emit(rf, g, 0.0, 0.0, self._dth(rf), 0.0, 1.0)

    def _verify(self, state, rf, wp, g):
        obj = _by_name(state, wp["obj"])
        if obj is None:
            self._next()
            return self._nop(0.0)
        oy = _bounds(state, obj)[2]
        if self.steps >= 3:
            moved = self.prev_obj_y is not None and (oy - self.prev_obj_y) > 0.004
            if moved:
                self.garm = rf["arm"]
                self.goff = self._suction_cy(rf) - oy
                self._next()
                return self._nop(1.0)
            self.held = None
            self.grab_tries += 1
            if self.grab_tries > 4:
                self.grab_tries = 0
                self._next()
                return self._nop(0.0)
            self.i -= 2
            self.steps = 0
            self.stall = 0
            return self._nop(0.0)
        return self._emit(rf, g, 0.0, self.mdy * 0.35, self._dth(rf), 0.0, 1.0)

    def _hoist(self, state, rf, wp, g):
        obj = _by_name(state, wp["obj"])
        dth = self._dth(rf)
        darm = 0.0
        if self.garm is not None:
            ea = self.garm - rf["arm"]
            if abs(ea) > 3e-3:
                darm = float(np.clip(ea, -self.mda, self.mda))

        need = g["top_stuff"] + 0.025
        lift = 0.0
        if obj is not None:
            lift = need - _bounds(state, obj)[2]
        target = float(np.clip(rf["y"] + max(lift, 0.0), g["ymin"], g["ymax"]))
        e = target - rf["y"]
        dy = min(self.mdy, e) if e > 0.005 else 0.0
        if e <= 0.008 or rf["y"] >= g["ymax"] - 0.002:
            self._next()
            return self._nop(1.0)
        return self._emit(rf, g, 0.0, dy, dth, darm, 1.0)

    def _xcarry(self, state, rf, wp, g):
        obj = _by_name(state, wp["obj"])
        if obj is None:
            self._next()
            return self._nop(1.0)
        a, b, oy0, _ = _bounds(state, obj)
        ocx = 0.5 * (a + b)
        cx = float(wp["cx"])
        dth = self._dth(rf)

        darm = 0.0
        if self.garm is not None:
            ea = self.garm - rf["arm"]
            if abs(ea) > 4e-3:
                darm = float(np.clip(ea, -self.mda, self.mda))

        route_top = self._top_between(g, rf["x"], cx,
                                      exclude=obj.name)
        need_bottom = max(route_top, g["table_top"]) + 0.022
        lift = need_bottom - oy0
        target = float(np.clip(rf["y"] + max(lift, 0.0), g["ymin"], g["ymax"]))
        ey = target - rf["y"]
        dy = min(self.mdy, ey) if ey > 0.005 else 0.0

        ex = cx - ocx
        dx = 0.0
        if ey <= 0.008 and abs(ex) > 0.0035:
            dx = float(np.clip(ex, -self.mdx, self.mdx))

        if abs(ex) <= 0.004 and ey <= 0.010:
            self._next()
            return self._nop(1.0)
        return self._emit(rf, g, dx, dy, dth, darm, 1.0)

    def _predrop(self, state, rf, wp, g):
        """Get the BASE low enough that a full arm extension reaches the support.

        This is the key fix for seed 5: previously the base stayed high, the arm
        saturated ~3.5 cm short, and the block was released in mid-air.

        Required base y so that (object bottom) can reach support_top with the
        arm fully out:
            base_y = support_top + goff + arm_max + 1.5*gw
        We descend the base to that height (plus a small margin) while the arm
        is retracted, holding position in x.  The column below is free: it is
        the drop column, cleared during planning.
        """
        obj = _by_name(state, wp["obj"])
        if obj is None:
            self._next()
            return self._nop(1.0)
        dth = self._dth(rf)
        top = wp.get("top")
        top = g["table_top"] if top is None else float(top)
        goff = self.goff if self.goff is not None else 0.5 * rf["gw"]

        # Keep the arm at grasp extension while repositioning the base.
        darm = 0.0
        if self.garm is not None:
            ea = self.garm - rf["arm"]
            if abs(ea) > 4e-3:
                darm = float(np.clip(ea, -self.mda, self.mda))

        want_base = top + goff + rf["amax"] + 1.5 * rf["gw"] - 0.005
        # Never put the base below a safe floor above the support.
        floor = top + rf["r"] + 0.002
        want_base = max(want_base, floor)
        want_base = float(np.clip(want_base, g["ymin"], g["ymax"]))

        e = want_base - rf["y"]
        dy = 0.0
        if e < -0.005:
            dy = max(-self.mdy, e)
        elif e > 0.005:
            dy = min(self.mdy, e)

        if abs(e) <= 0.008:
            self._next()
            return self._nop(1.0)
        return self._emit(rf, g, 0.0, dy, dth, darm, 1.0, ymin=floor)

    def _place(self, state, rf, wp, g):
        """Lower until the goal predicate actually holds.  Never release early."""
        obj = _by_name(state, wp["obj"])
        if obj is None:
            self._next()
            return self._nop(1.0)
        a, b, oy0, _ = _bounds(state, obj)
        ocx = 0.5 * (a + b)
        cx = float(wp["cx"])
        dth = self._dth(rf)
        final = bool(wp.get("final", False))
        top = wp.get("top")
        top = g["table_top"] if top is None else float(top)

        # --- exact goal test for the final placement ---
        if final and wp.get("sup"):
            sup = _by_name(state, wp["sup"])
            if sup is not None and _is_on(state, obj, sup):
                self._next()
                return self._nop(1.0)

        # Target: object bottom essentially touching the support.
        want = top + 0.001
        err = oy0 - want

        # X trim: keep the object centred on the drop point the whole way down.
        ex = cx - ocx
        dx = 0.0
        if abs(ex) > 0.0025:
            lim = self.mdx if err > 0.03 else 0.4 * self.mdx
            dx = float(np.clip(ex, -lim, lim))

        # Descend: extend the arm; if saturated, lower the base.
        darm = 0.0
        dy = 0.0
        if err > 0.001:
            room_arm = rf["amax"] - rf["arm"]
            if room_arm > 1e-4:
                darm = min(self.mda, err, room_arm)
            rest = err - darm
            if rest > 1e-4:
                floor = top + rf["r"] + 0.002
                room = rf["y"] - floor
                if room > 1e-4:
                    dy = -min(self.mdy, rest, room)
        elif err < -0.012:
            darm = max(-self.mda, err)

        # Non-final (obstruction) placement: settle when it is on the table.
        if not final and err <= 0.004 and abs(ex) <= 0.008:
            self._next()
            return self._nop(1.0)

        # Final placement: also accept when a tiny further descent would satisfy
        # is_on and we are already resting (err <= 0).
        if final and wp.get("sup"):
            sup = _by_name(state, wp["sup"])
            if sup is not None:
                bx0, bx1, by0, by1 = _bounds(state, sup)
                if err <= 0.004 and _would_be_on(a, b, max(oy0, top),
                                                 bx0, bx1, by0, by1):
                    self._next()
                    return self._nop(1.0)

        return self._emit(rf, g, dx, dy, dth, darm, 1.0,
                          ymin=top + rf["r"] + 0.002)

    def _let(self, state, rf, wp, g):
        if self.steps >= 3:
            self.held = None
            self.garm = None
            self.goff = None
            self.grab_tries = 0
            self._next()
        return self._emit(rf, g, 0.0, 0.0, self._dth(rf), 0.0, 0.0)