"""Policy for Obstruction3DEnv (variable object count).

NINETEEN ATTEMPTS -- THE SEARCH WAS DEADLOCKED, AND THAT IS FIXABLE
-------------------------------------------------------------------
Goal (`goal_reached` / `_target_block_on_target_region`):
  * nothing grasped, AND
  * target_block centre at region_z + region_hz + block_hz (within 5mm), AND
  * all four bottom corners inside the region footprint.

Seed 0 failed again: target_block at its exact sampled start pose
(0.33598167, 0.17328143, 0.1), obstructions untouched, no grasp.

The decisive clue: attempt 19's final state is **bit-identical to attempt 18's**
-- base (-0.15656744, 0.32800322), joints (0, -0.1, -pi, -1.8, 0, -0.82, pi/2).
Two different builds cannot land on identical floats by chance. The search was
not exploring at all; it was stuck in a fixed point and then idling for the rest
of the episode. Reading the state machine, the cause is clear:

  * `_goto` advances `self.off_i` only inside `_wiggle`, and `_wiggle` returns
    to GOTO only after the whole WIGGLE_SEQ. But `_wiggle` calls
    `np.clip(q[j]+delta, Q_LO[j], Q_HI[j])`; with j4 pinned at -1.8 and j2/j6
    at their posture values, several wiggle entries clip to a zero step and
    return `self._act(grip=-1.0)` **without incrementing `wg_i`** on the
    zero-step path... and `_goto`'s "arrived" branch resets `wg_i = 0` every
    time it re-enters. The pair ping-pongs between the same base pose and the
    same wiggle index forever.
  * Because the base target is recomputed from the same object and same offset
    each step and the error is already < 0.025, `_goto` immediately hands back
    to WIGGLE, which immediately hands back. Nothing ever changes.

FIXES IN THIS VERSION
---------------------
1. **Monotonic search index.** A single counter `self.probe_i` advances by one
   on *every* exploration step, and the (posture, offset, wiggle) triple is
   derived from it by integer division/modulo. It is structurally impossible to
   revisit the same configuration twice or to stall.
2. **No zero-step dead ends.** If a computed joint step clips to ~0, the step is
   skipped *and* the counter still advances.
3. **Watchdog on staleness.** If neither the joints nor the base have changed
   for 12 consecutive steps, the search jumps forward by a whole offset block.
4. Retained from earlier fixes: verified UNFOLD, elbow bounds that cannot fold
   the arm back to retract, budget-sized offset grid, gripper closed on every
   exploration step (free grasp attempts), and the corrected `EE.solve` that
   returns `(db, dq, d)`.

On the broader problem I will not overclaim. The root obstacle is unchanged
across all nineteen attempts: the observation exposes joint angles but not the
end-effector pose, `primitives` is empty (no IK), and I have no FK model for
this robot; the only exact EE readout is a held object's pose, which is gated
behind the grasp itself. This submission fixes a genuine deadlock that was
wasting the entire budget, which is worth doing on its own, but it does not
solve that underlying gap. Low confidence on the grasp.
"""

from __future__ import annotations

import numpy as np

MAX_MAG = 0.2
NJ = 7
ADIM = 11

TABLE_P = np.array([0.3, 0.0, -0.175])
TABLE_H = np.array([0.2, 0.4, 0.25])
TABLE_TOP = TABLE_P[2] + TABLE_H[2]          # 0.075

CLEAR_Z = TABLE_TOP + 0.16
MID_Z = TABLE_TOP + 0.10

HOME = np.array([0.0, -0.35, -np.pi, -2.5, 0.0, -0.87, np.pi / 2])

Q_LO = np.array([-2.2, -1.80, -2.0 * np.pi, -2.00, -2.0 * np.pi, -1.80,
                 -2.0 * np.pi])
Q_HI = np.array([2.2, 1.80, 2.0 * np.pi, -0.60, 2.0 * np.pi, 1.80,
                 2.0 * np.pi])

BSTEP = 0.12
BX_LO, BX_HI = -1.00, 0.35
BY_LO, BY_HI = -0.65, 0.65

JSTEP = 0.05


def P(state, o):
    return np.array([float(state.get(o, "pose_x")),
                     float(state.get(o, "pose_y")),
                     float(state.get(o, "pose_z"))])


def HE(state, o):
    return np.array([float(state.get(o, "half_extent_x")),
                     float(state.get(o, "half_extent_y")),
                     float(state.get(o, "half_extent_z"))])


def JQ(state, r):
    return np.array([float(state.get(r, "joint_%d" % i)) for i in range(1, 8)])


def BP(state, r):
    return np.array([float(state.get(r, "pos_base_x")),
                     float(state.get(r, "pos_base_y"))])


def overlap(p1, h1, p2, h2, pad=0.0):
    return (abs(p1[0] - p2[0]) < h1[0] + h2[0] + pad and
            abs(p1[1] - p2[1]) < h1[1] + h2[1] + pad)


def cuboids(state):
    out = {}
    for n in state.get_object_names():
        if n == "robot":
            continue
        try:
            o = state.get_object_from_name(n)
            state.get(o, "pose_x")
            out[n] = o
        except Exception:
            pass
    return out


class EE:
    """EE tracker: base translation is exact 1:1; arm part learned when held."""

    def __init__(self):
        self.pos = None
        self.Jarm = None
        self.pq = None
        self.pb = None
        self.pee = None

    def anchor(self, pos, q, b):
        self.pos = np.asarray(pos, float).copy()
        self.pq = np.asarray(q, float).copy()
        self.pb = np.asarray(b, float).copy()
        self.pee = self.pos.copy()

    def propagate(self, q, b):
        q = np.asarray(q, float)
        b = np.asarray(b, float)
        if self.pos is None:
            self.pq = q.copy()
            self.pb = b.copy()
            return
        if self.pb is not None:
            db = b - self.pb
            self.pos[0] += db[0]
            self.pos[1] += db[1]
        if self.Jarm is not None and self.pq is not None:
            self.pos = self.pos + self.Jarm @ (q - self.pq)
        self.pq = q.copy()
        self.pb = b.copy()

    def observe(self, pos, q, b):
        pos = np.asarray(pos, float)
        q = np.asarray(q, float)
        b = np.asarray(b, float)
        if (self.Jarm is not None and self.pq is not None and
                self.pee is not None and self.pb is not None):
            dq = q - self.pq
            db = b - self.pb
            dee_arm = (pos - self.pee) - np.array([db[0], db[1], 0.0])
            n = float(dq @ dq)
            if n > 1e-9:
                self.Jarm = self.Jarm + np.outer(
                    dee_arm - self.Jarm @ dq, dq) / n
        self.pos = pos.copy()
        self.pee = pos.copy()
        self.pq = q.copy()
        self.pb = b.copy()

    def ready(self):
        return self.pos is not None

    def solve(self, target, cap_base, cap_arm):
        """Return (delta_base_xy, delta_joints, distance)."""
        if self.pos is None:
            return np.zeros(2), np.zeros(NJ), 1e9
        err = np.asarray(target, float) - self.pos
        d = float(np.linalg.norm(err))
        db = np.clip(err[:2], -cap_base, cap_base)
        dq = np.zeros(NJ)
        if self.Jarm is not None and abs(err[2]) > 1e-4:
            ez = np.array([0.0, 0.0, err[2]])
            J = self.Jarm
            lam = 0.06
            try:
                dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(3), ez)
            except np.linalg.LinAlgError:
                dq = J.T @ ez
            m = float(np.max(np.abs(dq))) if dq.size else 0.0
            if m > cap_arm:
                dq = dq * (cap_arm / (m + 1e-12))
        return db, dq, d


class GeneratedApproach:

    UNFOLD = 0
    SEARCH = 1
    MEASURE = 2
    RUN = 3
    DONE = 4

    ABOVE, DOWN, CLOSE, LIFT, OVER, LOWER, OPEN, UP = range(8)

    POSTURES = [
        (0.05, -1.40, -0.62),
        (0.35, -1.05, -0.40),
        (-0.15, -1.75, -0.85),
        (0.50, -0.85, -0.25),
    ]

    OFFSETS = [(ox, oy)
               for ox in (-0.60, -0.48, -0.36, -0.26, -0.18)
               for oy in (-0.16, -0.08, 0.0, 0.08, 0.16)]

    # Per-offset micro-sweep: alternate base nudges and bounded joint nudges.
    # Entries are ('b', dx, dy) or ('j', joint, delta).
    MICRO = [
        ('b', 0.0, 0.0),
        ('j', 3, -JSTEP),
        ('j', 3, -JSTEP),
        ('b', 0.03, 0.0),
        ('j', 1, +JSTEP),
        ('j', 3, +JSTEP),
        ('b', -0.03, 0.03),
        ('j', 5, +JSTEP),
        ('j', 3, -JSTEP),
        ('b', 0.0, -0.03),
        ('j', 1, -JSTEP),
        ('j', 5, -JSTEP),
    ]

    STEPS_PER_MICRO = 3   # base moves get a few steps to converge

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives
        self.lo = np.asarray(action_space.low, float)
        self.hi = np.asarray(action_space.high, float)
        self._init()

    def _init(self):
        self.ee = EE()
        self.mode = self.UNFOLD
        self.t = 0
        self.prev_q = None
        self.prev_b = None
        self.rejects = 0
        self.stale = 0
        self.last_cmd_joint = None
        # unfold
        self.post_i = 0
        self.uf_steps = 0
        self.uf_dir = {1: +1, 3: +1, 5: +1}
        # monotonic search counter -- cannot stall or revisit
        self.probe_i = 0
        # measure
        self.m_j = 0
        self.m_ph = 0
        self.m_q0 = None
        self.m_ee0 = None
        self.m_cols = [None] * NJ
        self.m_delta = 0.08
        self.m_tries = 0
        # plan
        self.plan = []
        self.pi = 0
        self.ph = self.ABOVE
        self.ps = 0
        self.dumps = []

    def reset(self, state, info):
        self._init()
        try:
            self._plan(state)
        except Exception:
            self.plan = []
        return None

    # -- planning -------------------------------------------------------

    def _plan(self, state):
        objs = cuboids(state)
        if "target_region" not in objs or "target_block" not in objs:
            self.plan = []
            return
        reg = objs["target_region"]
        rp, rh = P(state, reg), HE(state, reg)
        names = sorted([n for n in objs if n.startswith("obstruction")],
                       key=lambda n: (len(n), n))
        blocking = [n for n in names
                    if overlap(P(state, objs[n]), HE(state, objs[n]),
                               rp, rh, pad=0.006)]

        def iso(n):
            p = P(state, objs[n])
            d = [float(np.linalg.norm(p[:2] - P(state, objs[m])[:2]))
                 for m in blocking if m != n]
            return -min(d) if d else -1e9

        blocking.sort(key=iso)
        self.dumps = self._dumps(state, objs, len(blocking))
        self.plan = [(n, self.dumps[i % len(self.dumps)])
                     for i, n in enumerate(blocking)]
        self.plan.append(("target_block", np.array([rp[0], rp[1]])))
        self.pi = 0

    def _dumps(self, state, objs, k):
        reg, blk = objs["target_region"], objs["target_block"]
        rp, rh = P(state, reg), HE(state, reg)
        bp, bh = P(state, blk), HE(state, blk)
        mg = 0.05
        xs = np.linspace(TABLE_P[0] - TABLE_H[0] + mg,
                         TABLE_P[0] + TABLE_H[0] - mg, 4)
        ys = np.linspace(TABLE_P[1] - TABLE_H[1] + mg,
                         TABLE_P[1] + TABLE_H[1] - mg, 9)
        c = []
        for y in ys:
            for x in xs:
                p = np.array([x, y])
                if (abs(p[0] - rp[0]) < rh[0] + 0.085 and
                        abs(p[1] - rp[1]) < rh[1] + 0.085):
                    continue
                if (abs(p[0] - bp[0]) < bh[0] + 0.07 and
                        abs(p[1] - bp[1]) < bh[1] + 0.07):
                    continue
                c.append(p)
        c.sort(key=lambda p: -float(np.linalg.norm(p - rp[:2])))
        out = []
        for p in c:
            if all(float(np.linalg.norm(p - q)) > 0.09 for q in out):
                out.append(p)
            if len(out) >= max(k, 1) + 3:
                break
        if not out:
            out = [np.array([TABLE_P[0], TABLE_P[1] + TABLE_H[1] - 0.07])]
        return out

    # -- helpers --------------------------------------------------------

    def _z(self):
        return np.zeros(ADIM, dtype=np.float32)

    def _clip(self, a):
        return np.clip(np.asarray(a, float), self.lo, self.hi).astype(np.float32)

    def _act(self, db=None, dq=None, grip=0.0):
        a = self._z()
        if db is not None:
            a[0:2] = np.clip(np.asarray(db, float), -MAX_MAG, MAX_MAG)
        if dq is not None:
            a[3:10] = np.clip(np.asarray(dq, float), -MAX_MAG, MAX_MAG)
        a[10] = grip
        return self._clip(a)

    def _single(self, jidx, delta, grip=0.0):
        dq = np.zeros(NJ)
        dq[jidx] = delta
        self.last_cmd_joint = jidx
        return self._act(dq=dq, grip=grip)

    def _held(self, state):
        for n, o in cuboids(state).items():
            try:
                if float(state.get(o, "grasp_active")) > 0.5:
                    return n
            except Exception:
                pass
        return None

    def _cur_target_name(self, state):
        objs = cuboids(state)
        for i in range(self.pi, len(self.plan)):
            if self.plan[i][0] in objs:
                return self.plan[i][0]
        return "target_block" if "target_block" in objs else None

    def _posture_err(self, q):
        j2, j4, j6 = self.POSTURES[self.post_i % len(self.POSTURES)]
        want = HOME.copy()
        want[1] = j2
        want[3] = j4
        want[5] = j6
        return want - q

    # -- main -----------------------------------------------------------

    def get_action(self, state):
        self.t += 1
        try:
            return self._inner(state)
        except Exception:
            return self._z()

    def _inner(self, state):
        robot = state.get_object_from_name("robot")
        q = JQ(state, robot)
        b = BP(state, robot)
        held = self._held(state)

        moved = True
        if self.prev_q is not None and self.prev_b is not None:
            moved = (np.max(np.abs(q - self.prev_q)) > 1e-6 or
                     np.max(np.abs(b - self.prev_b)) > 1e-6)
        self.rejects = 0 if moved else self.rejects + 1
        self.stale = 0 if moved else self.stale + 1
        self.prev_q = q.copy()
        self.prev_b = b.copy()

        if held is not None:
            self.ee.observe(P(state, state.get_object_from_name(held)), q, b)
        else:
            self.ee.propagate(q, b)

        if held is not None and self.mode in (self.UNFOLD, self.SEARCH):
            self._start_measure(state, q, b, held)
            return self._z()

        if self.mode == self.UNFOLD:
            return self._unfold(q, moved)
        if self.mode == self.SEARCH:
            return self._search(state, q, b)
        if self.mode == self.MEASURE:
            return self._measure(state, q, b, held)
        if self.mode == self.DONE or self.pi >= len(self.plan):
            return self._act(grip=1.0)
        return self._run(state, q, b, held)

    # -- UNFOLD ----------------------------------------------------------

    def _unfold(self, q, moved):
        self.uf_steps += 1
        err = self._posture_err(q)
        idxs = [3, 1, 5]
        k = max(idxs, key=lambda i: abs(err[i]))

        if abs(err[k]) < 0.06:
            self.mode = self.SEARCH
            self.uf_steps = 0
            return self._act(grip=-1.0)

        if self.uf_steps > 80:
            self.post_i += 1
            self.uf_steps = 0
            return self._act(grip=-1.0)

        if (not moved) and self.last_cmd_joint == k:
            self.uf_dir[k] = -self.uf_dir.get(k, 1)
            if self.rejects >= 5:
                self.post_i += 1
                self.uf_steps = 0
                self.rejects = 0
                return self._act(grip=-1.0)
            nxt = np.clip(q[k] + self.uf_dir[k] * JSTEP, Q_LO[k], Q_HI[k])
            return self._single(k, nxt - q[k], grip=-1.0)

        d = np.clip(err[k], -JSTEP, JSTEP)
        nxt = np.clip(q[k] + d, Q_LO[k], Q_HI[k])
        if abs(nxt - q[k]) < 1e-6:
            self.post_i += 1
            self.uf_steps = 0
            return self._act(grip=-1.0)
        return self._single(k, nxt - q[k], grip=-1.0)

    # -- SEARCH: strictly monotonic, cannot stall -------------------------

    def _search(self, state, q, b):
        name = self._cur_target_name(state)
        if name is None:
            return self._act(grip=1.0)
        objs = cuboids(state)
        op = P(state, objs[name])

        n_micro = len(self.MICRO) * self.STEPS_PER_MICRO
        n_off = len(self.OFFSETS)

        # Staleness watchdog: jump a whole offset block forward.
        if self.stale >= 12:
            self.stale = 0
            self.probe_i += n_micro

        # The counter advances every single step, so no configuration can be
        # revisited and no ping-pong is possible.
        idx = self.probe_i
        self.probe_i += 1

        micro_slot = (idx // self.STEPS_PER_MICRO) % len(self.MICRO)
        off_idx = (idx // n_micro) % n_off
        post_idx = (idx // (n_micro * n_off)) % len(self.POSTURES)

        if post_idx != (self.post_i % len(self.POSTURES)):
            # Move to the next posture and re-verify the unfold.
            self.post_i = post_idx
            self.mode = self.UNFOLD
            self.uf_steps = 0
            return self._act(grip=-1.0)

        ox, oy = self.OFFSETS[off_idx]
        kind = self.MICRO[micro_slot]

        if kind[0] == 'b':
            _, dx, dy = kind
            tgt = np.array([np.clip(op[0] + ox + dx, BX_LO, BX_HI),
                            np.clip(op[1] + oy + dy, BY_LO, BY_HI)])
            err = tgt - b
            if float(np.max(np.abs(err))) < 0.02:
                return self._act(grip=-1.0)
            return self._act(db=np.clip(err, -BSTEP, BSTEP), grip=-1.0)

        _, jidx, delta = kind
        nxt = np.clip(q[jidx] + delta, Q_LO[jidx], Q_HI[jidx])
        step = nxt - q[jidx]
        if abs(step) < 1e-6:
            # Zero step: skip, but the counter already advanced.
            return self._act(grip=-1.0)
        return self._single(jidx, step, grip=-1.0)

    # -- MEASURE ---------------------------------------------------------

    def _start_measure(self, state, q, b, held):
        self.mode = self.MEASURE
        self.m_j = 0
        self.m_ph = 0
        self.m_cols = [None] * NJ
        self.m_tries = 0
        pos = P(state, state.get_object_from_name(held))
        self.ee.Jarm = None
        self.ee.anchor(pos, q, b)

    def _measure(self, state, q, b, held):
        if held is None:
            self.mode = self.SEARCH
            return self._z()

        pos = P(state, state.get_object_from_name(held))

        if self.m_j >= NJ:
            J = np.zeros((3, NJ))
            ok = 0
            for i in range(NJ):
                if self.m_cols[i] is not None:
                    J[:, i] = self.m_cols[i]
                    if np.linalg.norm(self.m_cols[i]) > 1e-4:
                        ok += 1
            if ok < 1 and self.m_tries < 3:
                self.m_j = 0
                self.m_ph = 0
                self.m_cols = [None] * NJ
                self.m_delta = min(0.16, self.m_delta * 1.5)
                self.m_tries += 1
                return self._z()
            if ok < 1:
                J[2, 3] = 0.3
            self.ee.Jarm = J
            self.ee.anchor(pos, q, b)
            self.mode = self.RUN
            for i in range(self.pi, len(self.plan)):
                if self.plan[i][0] == held:
                    self.plan.insert(self.pi, self.plan.pop(i))
                    break
            else:
                spot = self.dumps[0] if self.dumps else \
                    np.array([TABLE_P[0], TABLE_P[1] + 0.3])
                self.plan.insert(self.pi, (held, spot))
            self.ph = self.LIFT
            self.ps = 0
            return self._z()

        if self.m_ph == 0:
            self.m_q0 = q.copy()
            self.m_ee0 = pos.copy()
            self.m_ph = 1
            nxt = np.clip(q[self.m_j] + self.m_delta,
                          Q_LO[self.m_j], Q_HI[self.m_j])
            step = nxt - q[self.m_j]
            if abs(step) < 1e-6:
                self.m_ph = 3
                nxt = np.clip(q[self.m_j] - self.m_delta,
                              Q_LO[self.m_j], Q_HI[self.m_j])
                step = nxt - q[self.m_j]
                if abs(step) < 1e-6:
                    self.m_cols[self.m_j] = np.zeros(3)
                    self.m_ph = 0
                    self.m_j += 1
                    return self._z()
            return self._single(self.m_j, step, grip=0.0)

        if self.m_ph == 1:
            a = float((q - self.m_q0)[self.m_j])
            if abs(a) > 1e-4:
                self.m_cols[self.m_j] = (pos - self.m_ee0) / a
                self.m_ph = 2
                return self._single(self.m_j, -a, grip=0.0)
            self.m_ph = 3
            nxt = np.clip(q[self.m_j] - self.m_delta,
                          Q_LO[self.m_j], Q_HI[self.m_j])
            return self._single(self.m_j, nxt - q[self.m_j], grip=0.0)

        if self.m_ph == 3:
            a = float((q - self.m_q0)[self.m_j])
            if abs(a) > 1e-4:
                self.m_cols[self.m_j] = (pos - self.m_ee0) / a
            else:
                self.m_cols[self.m_j] = np.zeros(3)
            self.m_ph = 2
            return self._single(self.m_j, -a, grip=0.0)

        self.m_ph = 0
        self.m_j += 1
        return self._z()

    # -- RUN --------------------------------------------------------------

    def _run(self, state, q, b, held):
        name, spot = self.plan[self.pi]
        objs = cuboids(state)
        if name not in objs:
            self._next()
            return self._z()
        obj = objs[name]
        op, oh = P(state, obj), HE(state, obj)
        self.ps += 1

        if self.ps > 150 or self.rejects > 18:
            self.rejects = 0
            if self.ph in (self.ABOVE, self.DOWN, self.CLOSE):
                self.mode = self.SEARCH
                self.ps = 0
                return self._z()
            self.ph = (self.ph + 1) % 8
            self.ps = 0

        if self.ph == self.ABOVE:
            if held == name:
                self.ph = self.LIFT
                self.ps = 0
                return self._z()
            if held is not None:
                self.plan.insert(self.pi,
                                 (held, self.dumps[0] if self.dumps
                                  else np.array([TABLE_P[0], 0.3])))
                self.ph = self.LIFT
                self.ps = 0
                return self._z()
            db, dq, d = self.ee.solve(np.array([op[0], op[1], CLEAR_Z]),
                                      BSTEP, 0.10)
            if d < 0.03:
                self.ph = self.DOWN
                self.ps = 0
            return self._act(db=db, dq=dq, grip=-1.0)

        if self.ph == self.DOWN:
            tgt = np.array([op[0], op[1], op[2] + oh[2] - 0.004])
            db, dq, d = self.ee.solve(tgt, 0.05, 0.045)
            if d < 0.015:
                self.ph = self.CLOSE
                self.ps = 0
            return self._act(db=db, dq=dq, grip=-1.0)

        if self.ph == self.CLOSE:
            if self.ps >= 2:
                if held == name:
                    self.ph = self.LIFT
                    self.ps = 0
                    return self._z()
                if held is not None:
                    self.plan.insert(self.pi,
                                     (held, self.dumps[0] if self.dumps
                                      else np.array([TABLE_P[0], 0.3])))
                    self.ph = self.LIFT
                    self.ps = 0
                    return self._z()
                if self.ps > 6:
                    self.mode = self.SEARCH
                    self.ps = 0
                    return self._z()
            return self._act(grip=-1.0)

        if self.ph == self.LIFT:
            if held is None:
                self.ph = self.ABOVE
                self.ps = 0
                return self._z()
            hp = P(state, objs[held])
            db, dq, _ = self.ee.solve(np.array([hp[0], hp[1], CLEAR_Z]),
                                      0.04, 0.09)
            if hp[2] > TABLE_TOP + 0.07 or self.ps > 45:
                self.ph = self.OVER
                self.ps = 0
            return self._act(db=db, dq=dq, grip=0.0)

        if self.ph == self.OVER:
            if held is None:
                self.ph = self.ABOVE
                self.ps = 0
                return self._z()
            hn, hs = self.plan[self.pi]
            if held != hn:
                hs = spot
            if held == "target_block":
                rp = P(state, objs["target_region"])
                hs = np.array([rp[0], rp[1]])
            hp = P(state, objs[held])
            db, dq, _ = self.ee.solve(np.array([hs[0], hs[1], CLEAR_Z]),
                                      BSTEP, 0.08)
            if float(np.linalg.norm(hp[:2] - np.asarray(hs, float))) < 0.010:
                self.ph = self.LOWER
                self.ps = 0
            return self._act(db=db, dq=dq, grip=0.0)

        if self.ph == self.LOWER:
            if held is None:
                self._next()
                return self._z()
            ho = objs[held]
            hp, hh = P(state, ho), HE(state, ho)
            hn, hs = self.plan[self.pi]
            if held != hn:
                hs = spot
            if held == "target_block":
                reg = objs["target_region"]
                rp, rh = P(state, reg), HE(state, reg)
                rest = rp[2] + rh[2] + hh[2]
                hs = np.array([rp[0], rp[1]])
            else:
                rest = TABLE_TOP + hh[2]
            db, dq, _ = self.ee.solve(
                np.array([hs[0], hs[1], rest + 0.0015]), 0.04, 0.04)
            if (hp[2] - rest) < 0.004:
                self.ph = self.OPEN
                self.ps = 0
            return self._act(db=db, dq=dq, grip=0.0)

        if self.ph == self.OPEN:
            if self.ps >= 2:
                if held is None:
                    self.ph = self.UP
                    self.ps = 0
                    return self._z()
                if self.ps > 6:
                    self.ph = self.LOWER
                    self.ps = 0
                    return self._z()
            return self._act(grip=1.0)

        if self.ph == self.UP:
            if self.ee.ready():
                db, dq, _ = self.ee.solve(
                    np.array([self.ee.pos[0], self.ee.pos[1], MID_Z]),
                    0.02, 0.09)
            else:
                db, dq = np.zeros(2), np.zeros(NJ)
            if self.ps > 6:
                self._next()
            return self._act(db=db, dq=dq, grip=1.0)

        return self._z()

    def _next(self):
        self.pi += 1
        self.ph = self.ABOVE
        self.ps = 0
        if self.pi >= len(self.plan):
            self.mode = self.DONE