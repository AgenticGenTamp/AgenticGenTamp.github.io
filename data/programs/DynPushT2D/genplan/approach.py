"""
Policy for kinder/DynPushT2D-t1-v0.

DIAGNOSIS OF THE 1000-STEP FAILURE ON SEED 0
============================================
Compare the initial and final states reported for seed 0:

    initial tblock: x=1.04958291  y=3.43981072  theta=2.59341978  v=0  omega=0
    final   tblock: x=1.04958291  y=3.43981072  theta=2.59341978  v=0  omega=0

The block is byte-for-byte identical after 1000 steps. It was never touched.
Only the robot moved (1.637, 1.270) -> (0.371, 3.577), i.e. up and to the LEFT,
away from the block, and it parked there against the left wall.

Root cause: the routing/standoff logic drove the robot to a "standoff" point
computed as `contact - push * (rad * standoff)` and then required the robot to
be *behind* the contact before ever driving through it. On seed 0 the goal is
to the RIGHT of the block (goal x=3.805, block x=1.050), so `push = +x`, so the
standoff sits to the LEFT of the block's left surface. `_safe_target` clamped
that toward the wall, `_route` kept generating fresh detour waypoints, and the
`behind < rad*0.8 or lateral > rad*1.2` gate never opened because the clamped
standoff was never actually reached. The controller livelocked in "go to
standoff" forever and never emitted a push. The `_flip` stall handler only
flipped rotation direction, which did nothing for this failure mode.

Secondary bug: `_angle_error` was incoherent. It computed three candidate
errors and then unconditionally overwrote `used = raw` whenever
`abs(raw) <= pi`, making the earlier branch logic dead code. It also servoed
on `e_th` (possibly wrapped) while the success gate `ang_ok` used `e_th_raw`,
so the two could disagree about which way to spin.

FIXES IN THIS VERSION
=====================
1. Removed the standoff/"behind the contact" gating entirely. The controller
   now always commands a position for the robot each step and never waits to
   "arrive" somewhere first. Approach and push are the same continuous motion:
   the robot servos to a point just behind the contact along the push axis, and
   because the command is a clipped delta it naturally walks in and pushes
   through. No gate can livelock.

2. Replaced ad-hoc routing with a hard rule: if the straight line to the
   commanded point passes through the block, orbit around the block on the
   shorter arc at a fixed clearance radius. The orbit target is recomputed
   every step from the current geometry (no cached waypoint that can go stale),
   and orbit targets are never wall-clamped in a way that traps the robot --
   if the orbit point is out of bounds we take the other arc direction.

3. Angle control now servos on exactly the quantity the environment tests:
   `e_raw = theta_goal - theta`, unwrapped, no shortcuts. This is the whole
   seed-0 trap: theta_goal = -3.1244 and the block sits at +2.593. The wrapped
   error is only -0.565 rad (~32 deg) going CCW through +pi, but the env's
   check is `abs(theta - theta_goal) < deg2rad(8)` with theta renormalized to
   [-pi, pi]. Rotating CCW parks theta near +3.13, giving raw error ~6.25 rad
   -- never terminates. So we must rotate CW (negative), taking theta from
   +2.593 down through 0 and on to -3.124. That is 5.72 rad of rotation the
   "long way", and it is the only way to satisfy the literal test. The sign of
   the commanded torque is therefore `sign(e_raw)`, always.

4. Progress watchdog with real escalation: if the block's pose error has not
   improved for a while, we cycle through distinct contact choices (different
   lever points / push offsets) rather than merely flipping a sign. If the
   block has not moved AT ALL for many steps (the seed-0 symptom), we force a
   direct centroid ram that ignores all niceties.

5. Explicit no-op guard: the action returned is never allowed to be ~zero
   unless we are genuinely done and settled. A zero action while the task is
   unsolved is always a bug in this env.

Theta branch, stated once more because it is the crux
-----------------------------------------------------
Termination uses raw, unwrapped `abs(theta - theta_goal)`. theta is
renormalized into ~[-pi, pi] every step by `_read_state_from_space`. Therefore
a target angle near -pi can only be reached by driving theta down through zero
to the negative branch; converging to the numerically-equivalent +pi side
scores ~2*pi of error and can never terminate. We always rotate in the
direction of the raw difference and we measure success with the raw difference.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Observation layout (from the environment's observation-space table)
# ---------------------------------------------------------------------------
B_X, B_Y, B_TH = 0, 1, 2
B_VX, B_VY, B_OM = 3, 4, 5
B_W, B_LH, B_LV = 12, 13, 14

R_X, R_Y = 16, 17
R_VX, R_VY = 19, 20
R_RAD = 28

G_X, G_Y, G_TH = 29, 30, 31

POS_TOL = 0.03
ANG_TOL = np.deg2rad(8.0)

WORLD_MIN = 0.0
WORLD_MAX = 5.0


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _unit(v):
    n = float(np.hypot(float(v[0]), float(v[1])))
    if n < 1e-12:
        return np.array([1.0, 0.0]), 0.0
    return np.array([float(v[0]) / n, float(v[1]) / n]), n


def _rot(v, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([c * float(v[0]) - s * float(v[1]),
                     s * float(v[0]) + c * float(v[1])])


def _inb(p, m):
    return (WORLD_MIN + m) <= float(p[0]) <= (WORLD_MAX - m) and \
           (WORLD_MIN + m) <= float(p[1]) <= (WORLD_MAX - m)


class _Block:
    """
    T-block geometry, body frame per dyn_pusht2d._add_state_to_space:

        horizontal bar: x in [-lh/2, lh/2], y in [-w, 0]
        vertical stem : x in [-w/2,  w/2],  y in [-w-lv, -w]

    The pose origin (x, y) is the TOP-CENTER of the bar, not the centroid.
    """

    def __init__(self, s):
        self.x = float(s[B_X])
        self.y = float(s[B_Y])
        self.th = float(s[B_TH])
        self.w = float(s[B_W])
        self.lh = float(s[B_LH])
        self.lv = float(s[B_LV])
        self.p = np.array([self.x, self.y])

        a_bar = self.lh * self.w
        a_stem = self.w * self.lv
        c_bar = np.array([0.0, -self.w * 0.5])
        c_stem = np.array([0.0, -self.w - self.lv * 0.5])
        tot = max(a_bar + a_stem, 1e-9)
        self.c_local = (a_bar * c_bar + a_stem * c_stem) / tot

        # Max distance from centroid to any vertex -> safe orbit radius.
        verts = [
            (-self.lh * 0.5, 0.0), (self.lh * 0.5, 0.0),
            (-self.lh * 0.5, -self.w), (self.lh * 0.5, -self.w),
            (-self.w * 0.5, -self.w - self.lv), (self.w * 0.5, -self.w - self.lv),
        ]
        self.reach = max(
            float(np.hypot(vx - self.c_local[0], vy - self.c_local[1]))
            for vx, vy in verts
        )

    @property
    def centroid(self):
        return self.p + _rot(self.c_local, self.th)

    def to_local(self, wp):
        return _rot(np.asarray(wp, dtype=float) - self.p, -self.th)

    def to_world(self, lp):
        return self.p + _rot(lp, self.th)

    def contains_local(self, q, pad=0.0):
        x, y = float(q[0]), float(q[1])
        in_bar = (-self.lh * 0.5 - pad <= x <= self.lh * 0.5 + pad
                  and -self.w - pad <= y <= 0.0 + pad)
        in_stem = (-self.w * 0.5 - pad <= x <= self.w * 0.5 + pad
                   and -self.w - self.lv - pad <= y <= -self.w + pad)
        return in_bar or in_stem

    def contains(self, wp, pad=0.0):
        return self.contains_local(self.to_local(wp), pad=pad)

    def surface_hit(self, target_local, push_local):
        """First boundary point met marching along +push toward target_local."""
        d, _ = _unit(push_local)
        span = self.reach * 3.0
        start = np.asarray(target_local, dtype=float) - d * span
        n = 200
        hit = None
        for i in range(n + 1):
            t = span * 2.0 * i / n
            if self.contains_local(start + d * t, pad=0.0):
                hit = t
                break
        if hit is None:
            return None
        a = max(0.0, hit - span * 2.0 / n)
        b = hit
        for _ in range(36):
            m = 0.5 * (a + b)
            if self.contains_local(start + d * m, pad=0.0):
                b = m
            else:
                a = m
        return start + d * b

    def segment_blocked(self, p0, p1, pad):
        d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-9:
            return self.contains(p0, pad=pad)
        steps = max(6, int(n / max(pad * 0.5, 1e-3)) + 1)
        for i in range(steps + 1):
            if self.contains(p0 + d * (i / steps), pad=pad):
                return True
        return False


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives

        self.a_lo = np.asarray(action_space.low, dtype=float)
        self.a_hi = np.asarray(action_space.high, dtype=float)
        self.step_max = float(min(self.a_hi[0], self.a_hi[1]))  # 0.05

        self.slow_v = 0.04
        self.slow_w = 0.05

        self._t = 0
        self._prev_err = None
        self._stall = 0
        self._variant = 0
        self._blk_prev = None
        self._frozen = 0
        self._ram = 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self, state, info):
        self._t = 0
        self._prev_err = None
        self._stall = 0
        self._variant = 0
        self._blk_prev = None
        self._frozen = 0
        self._ram = 0
        return None

    # -- action plumbing ---------------------------------------------------
    def _clip(self, v):
        a = np.asarray(v, dtype=float).reshape(2)
        if not np.all(np.isfinite(a)):
            a = np.zeros(2, dtype=float)
        return np.clip(a, self.a_lo, self.a_hi).astype(np.float64)

    def _go(self, robot, target, gain=1.0):
        d = np.asarray(target, dtype=float) - np.asarray(robot, dtype=float)
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-9:
            return self._clip(np.zeros(2))
        step = min(self.step_max * max(gain, 0.05), max(n, self.step_max * 0.35))
        return self._clip(d / n * step)

    def _route(self, robot, target, blk, rad):
        """
        Straight line if clear; otherwise orbit the block on an arc.
        Recomputed fresh every step -- no cached waypoints, no arrival gates.
        """
        pad = rad * 1.05
        if not blk.segment_blocked(robot, target, pad):
            return target

        c = blk.centroid
        R = blk.reach + rad * 2.3
        v_r, nr = _unit(robot - c)
        v_t, _ = _unit(target - c)
        a_r = np.arctan2(v_r[1], v_r[0])
        a_t = np.arctan2(v_t[1], v_t[0])
        sweep = _wrap(a_t - a_r)
        if abs(sweep) < 1e-6:
            sweep = np.pi

        margin = rad + 0.03
        for sgn in (np.sign(sweep) if sweep != 0 else 1.0, -np.sign(sweep)):
            span = sweep if (sweep * sgn) > 0 else sweep - sgn * 2.0 * np.pi
            # Step a bounded slice along the arc so we keep making progress.
            slice_ang = float(np.clip(abs(span), 0.0, 0.7)) * np.sign(span)
            ang = a_r + slice_ang
            cand = c + np.array([np.cos(ang), np.sin(ang)]) * R
            if _inb(cand, margin) and not blk.contains(cand, pad=rad * 0.9):
                return cand
        # Both arcs unusable (cornered): push straight out from the block.
        out = c + v_r * (R + rad)
        return np.array([
            float(np.clip(out[0], WORLD_MIN + margin, WORLD_MAX - margin)),
            float(np.clip(out[1], WORLD_MIN + margin, WORLD_MAX - margin)),
        ])

    # -- main --------------------------------------------------------------
    def get_action(self, state):
        s = np.asarray(state, dtype=float).reshape(-1)
        self._t += 1

        blk = _Block(s)
        rad = float(s[R_RAD])
        robot = np.array([float(s[R_X]), float(s[R_Y])])
        goal = np.array([float(s[G_X]), float(s[G_Y])])
        th_goal = float(s[G_TH])

        bv = np.array([float(s[B_VX]), float(s[B_VY])])
        bw = float(s[B_OM])
        speed = float(np.hypot(bv[0], bv[1]))

        # Errors measured EXACTLY as the env's termination test does.
        e_pos = goal - blk.p
        e_pos_n = float(np.hypot(e_pos[0], e_pos[1]))
        e_raw = th_goal - blk.th          # unwrapped, signed: what we servo on
        ang_mag = abs(e_raw)

        pos_ok = abs(e_pos[0]) < POS_TOL * 0.85 and abs(e_pos[1]) < POS_TOL * 0.85
        ang_ok = ang_mag < ANG_TOL * 0.85
        settled = speed < self.slow_v and abs(bw) < self.slow_w

        if pos_ok and ang_ok and settled:
            away, n = _unit(robot - blk.centroid)
            if n < blk.reach + rad * 2.2:
                return self._go(robot, robot + away * 0.3, gain=0.6)
            return self._clip(np.zeros(2))

        # --- watchdogs ----------------------------------------------------
        # Frozen-block detector: the exact seed-0 symptom was zero block motion.
        cur = np.array([blk.x, blk.y, blk.th])
        if self._blk_prev is not None and float(np.max(np.abs(cur - self._blk_prev))) < 1e-9:
            self._frozen += 1
        else:
            self._frozen = 0
        self._blk_prev = cur

        err = e_pos_n + 0.30 * ang_mag
        if self._prev_err is not None and err > self._prev_err - 2e-4:
            self._stall += 1
        else:
            self._stall = 0
        self._prev_err = err

        if self._frozen > 25 or self._stall > 60:
            self._variant = (self._variant + 1) % 4
            self._stall = 0
            self._frozen = 0
            self._ram = 30  # force a blunt centroid ram for a while

        # --- brute-force ram: guarantees contact, breaks any livelock -----
        if self._ram > 0:
            self._ram -= 1
            push, _ = _unit(e_pos if e_pos_n > POS_TOL else np.array([1.0, 0.0]))
            hit = blk.surface_hit(blk.to_local(blk.centroid), push)
            contact = blk.to_world(hit) if hit is not None else \
                blk.centroid - push * (blk.reach * 0.6)
            aim = contact + push * (rad * 2.0)   # aim THROUGH the block
            tgt = self._route(robot, aim, blk, rad)
            return self._go(robot, tgt, gain=1.0)

        # --- braking: kill drift once the pose is nearly right ------------
        near = (e_pos_n < POS_TOL * 2.0) and (ang_mag < ANG_TOL * 2.0)
        if near and not settled:
            if speed > self.slow_v * 0.5:
                d, _ = _unit(bv)
                hit = blk.surface_hit(blk.to_local(blk.centroid), d)
                face = blk.to_world(hit) if hit is not None else \
                    blk.centroid + d * (blk.reach * 0.6)
                stop = face + d * (rad * 0.9)
                return self._go(robot, self._route(robot, stop, blk, rad), gain=1.0)
            if abs(bw) > self.slow_w * 0.5:
                c = blk.centroid
                v_r, _ = _unit(robot - c)
                tang = np.array([-v_r[1], v_r[0]]) * (1.0 if bw > 0 else -1.0)
                stop = c + v_r * (blk.reach + rad * 1.0) - tang * (rad * 1.0)
                return self._go(robot, self._route(robot, stop, blk, rad), gain=0.9)
            return self._go(robot, blk.centroid, gain=0.15)

        # --- choose ROTATE vs TRANSLATE -----------------------------------
        # Angle is the expensive DOF (translating always induces some spin), and
        # on seed 0 it needs ~5.7 rad of CW rotation, so fix it first.
        if ang_mag > ANG_TOL * 2.0:
            mode = "ROTATE"
        elif e_pos_n > POS_TOL * 2.0:
            mode = "TRANSLATE"
        elif ang_mag > ANG_TOL * 0.85:
            mode = "ROTATE"
        else:
            mode = "TRANSLATE"

        # --- ROTATE -------------------------------------------------------
        if mode == "ROTATE":
            c = blk.centroid
            sign = 1.0 if e_raw > 0.0 else -1.0     # sign of the RAW error
            # Candidate lever points around the block; variant cycles which we
            # prefer so a stalled configuration gets a genuinely new contact.
            cands = []
            levers = [
                np.array([+blk.lh * 0.44, -blk.w * 0.5]),
                np.array([-blk.lh * 0.44, -blk.w * 0.5]),
                np.array([0.0, -blk.w - blk.lv * 0.85]),
                np.array([+blk.lh * 0.30, -blk.w * 0.5]),
                np.array([-blk.lh * 0.30, -blk.w * 0.5]),
            ]
            order = levers[self._variant % len(levers):] + \
                levers[:self._variant % len(levers)]
            for lp in order:
                tip = blk.to_world(lp)
                v_r, rn = _unit(tip - c)
                if rn < 1e-6:
                    continue
                push = np.array([-v_r[1], v_r[0]]) * sign   # torque of wanted sign
                hit = blk.surface_hit(blk.to_local(tip), push)
                if hit is None:
                    continue
                contact = blk.to_world(hit)
                aim = contact + push * (rad * 1.8)
                approach = contact - push * (rad * 1.15)
                cost = float(np.hypot(*(approach - robot))) - 0.5 * rn
                if not _inb(approach, rad + 0.03):
                    cost += 4.0
                cands.append((cost, contact, aim, approach, push))
            if cands:
                cands.sort(key=lambda z: z[0])
                _, contact, aim, approach, push = cands[0]
                # One continuous motion: if we're on the wrong side of the
                # contact, head for `approach`; otherwise drive through `aim`.
                if float(np.dot(robot - contact, push)) > rad * 0.35:
                    tgt = self._route(robot, approach, blk, rad)
                    return self._go(robot, tgt, gain=1.0)
                gain = float(np.clip(ang_mag / (ANG_TOL * 5.0), 0.20, 1.0))
                return self._go(robot, self._route(robot, aim, blk, rad), gain=gain)
            # No lever worked: fall through to translation rather than idle.

        # --- TRANSLATE ----------------------------------------------------
        push, pn = _unit(e_pos if e_pos_n > 1e-6 else np.array([1.0, 0.0]))
        hit = blk.surface_hit(blk.to_local(blk.centroid), push)
        contact = blk.to_world(hit) if hit is not None else \
            blk.centroid - push * (blk.reach * 0.6)

        # Small lateral bias bleeds off residual angle error while translating.
        if ang_mag > ANG_TOL * 0.6:
            perp = np.array([-push[1], push[0]])
            bias = float(np.clip(e_raw / (ANG_TOL * 8.0), -1.0, 1.0))
            contact = contact - perp * bias * min(rad * 1.1, blk.lh * 0.10)

        aim = contact + push * (rad * 1.8)
        approach = contact - push * (rad * 1.15)

        if float(np.dot(robot - contact, push)) > rad * 0.35:
            return self._go(robot, self._route(robot, approach, blk, rad), gain=1.0)

        gain = float(np.clip(e_pos_n / (POS_TOL * 10.0), 0.22, 1.0))
        act = self._go(robot, self._route(robot, aim, blk, rad), gain=gain)

        # Never emit a no-op while unsolved -- that is always a bug here.
        if float(np.hypot(act[0], act[1])) < self.step_max * 0.05:
            act = self._go(robot, blk.centroid, gain=0.5)
        return act