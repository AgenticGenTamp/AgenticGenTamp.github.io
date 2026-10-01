"""Policy for Packing3DEnv: pick each part and place it on the rack."""
import numpy as np

import kinlib
from kinlib import RDOWN, linpos, jac, ik, rotz

JNAMES = ['joint_%d' % i for i in range(1, 8)]
POSE_F = ['pose_x', 'pose_y', 'pose_z']
GRASP_D = np.array([-0.015, 0.0, -0.006])   # gripper-frame grasp offset


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.rng = np.random.default_rng(0)

    # ---------------- state helpers ----------------
    def _robot(self, s):
        return s.get_object_from_name('robot')

    def _q(self, s):
        r = self._robot(s)
        return np.array([s.get(r, n) for n in JNAMES])

    def _base(self, s):
        r = self._robot(s)
        return np.array([s.get(r, 'pos_base_x'), s.get(r, 'pos_base_y'), 0.0])

    def _grasping(self, s):
        return s.get(self._robot(s), 'grasp_active') > 0.5

    def _feat(self, s, name):
        o = s.get_object_from_name(name)
        return {k: s.get(o, k) for k in s.type_features[o.type]}

    def _ppos(self, s, name):
        o = s.get_object_from_name(name)
        return np.array([s.get(o, f) for f in POSE_F])

    def _yaw(self, s, name):
        o = s.get_object_from_name(name)
        return 2 * np.arctan2(s.get(o, 'pose_qz'), s.get(o, 'pose_qw'))

    def _linpos_world(self, s):
        return linpos(self._q(s)) + self._base(s)

    # ---------------- planning ----------------
    def reset(self, state, info):
        self._state = state
        self.part_names = sorted([n for n in state.get_object_names()
                                  if n.startswith('part')],
                                 key=lambda n: int(n[4:]))
        self.rack = self._ppos(state, 'rack')
        rf = self._feat(state, 'rack')
        self.rack_hx = rf.get('half_extent_x', 0.1)
        self.rack_hy = rf.get('half_extent_y', 0.15)
        self.rack_top = self.rack[2]
        self.slots = self._make_slots(len(self.part_names))
        self._gen = self._run()

    def _make_slots(self, n):
        """Slot targets. For n<=2: (x, y) footprint centers (y used).
        For n>=3: absolute reference-point targets (packed layout)."""
        rx, ry = self.rack[0], self.rack[1]
        if n <= 1:
            return [np.array([rx, ry])]
        if n == 2:
            return [np.array([rx, ry - 0.075]), np.array([rx, ry + 0.075])]
        packed = [(-0.0447, -0.0947), (0.0133, -0.0417), (0.0, 0.080),
                  (-0.055, 0.045), (0.03, 0.10), (-0.03, -0.13)]
        out = []
        for i in range(n):
            dx, dy = packed[i % len(packed)]
            out.append(np.array([rx + dx, ry + dy]))
        return out

    def _part_grasp_target(self, s, name):
        """World position that linpos must reach to grasp `name`."""
        f = self._feat(s, name)
        p = self._ppos(s, name)
        c = np.zeros(3)
        if 'triangle_type' in f and f['triangle_type'] > 0.5:
            c = np.array([f['side_a'] / 3.0, f['side_b'] / 3.0, 0.0])
        yaw = self._yaw(s, name)
        return p + rotz(yaw) @ c + RDOWN @ GRASP_D

    def _place_ref(self, s, name):
        """Offset from part pose to the reference point we position on a slot."""
        f = self._feat(s, name)
        if 'triangle_type' in f and f['triangle_type'] > 0.5:
            return np.array([f['side_a'] / 3.0, f['side_b'] / 3.0, 0.0])
        return np.zeros(3)

    # ---------------- action generation ----------------
    def get_action(self, state):
        self._state = state
        try:
            a = next(self._gen)
        except StopIteration:
            a = np.zeros(11, dtype=np.float32)
        return np.asarray(a, dtype=np.float32)

    def _act(self, dq, grip=0.0):
        a = np.zeros(11)
        a[3:10] = np.clip(dq, -0.2, 0.2)
        a[10] = grip
        return a

    def _move(self, target_fn, R=RDOWN, step=0.03, tol=3e-3, grip=0.0,
              max_steps=200, stop_fn=None):
        """Cartesian move of linpos toward target (world). Generator.

        target_fn: callable(state) -> world target for linpos.
        Yields actions; returns 'ok' / 'blocked' / 'stop' / 'slow' in
        self._last_result.
        """
        blocked = 0
        for _ in range(max_steps):
            s = self._state
            if stop_fn is not None and stop_fn(s):
                self._last_result = 'stop'
                return
            q = self._q(s)
            base = self._base(s)
            p = linpos(q) + base
            tgt = np.asarray(target_fn(s), float)
            d = tgt - p
            n = np.linalg.norm(d)
            if n < tol:
                self._last_result = 'ok'
                return
            w = p + d * min(1.0, step / n)
            qd, ok = ik(q, w - base, R, restarts=4, rng=self.rng)
            if not ok:
                qd, ok = ik(q, tgt - base, R, restarts=6, rng=self.rng)
                if not ok:
                    self._last_result = 'ikfail'
                    return
            dq = qd - q
            if np.max(np.abs(dq)) < 1e-6:
                self._last_result = 'ok'
                return
            yield self._act(dq, grip)
            q2 = self._q(self._state)
            if np.max(np.abs(q2 - q)) < 1e-9:
                blocked += 1
                if blocked >= 2:
                    self._last_result = 'blocked'
                    return
            else:
                blocked = 0
        self._last_result = 'slow'

    def _grip(self, val):
        a = np.zeros(11)
        a[10] = val
        yield a

    # ---------------- main pipeline ----------------
    def _placed(self, s, name):
        p = self._ppos(s, name)
        if abs(p[0] - self.rack[0]) > self.rack_hx + 0.06:
            return False
        if abs(p[1] - self.rack[1]) > self.rack_hy + 0.06:
            return False
        return p[2] > self.rack_top - 0.02

    def _run(self):
        CARRY_Z = 0.32
        slots = list(self.slots)
        order = list(self.part_names)
        if len(order) >= 3:
            def _iscub(n):
                return 'triangle_type' not in self._feat(self._state, n)
            order.sort(key=lambda n: (_iscub(n), n))
        for idx, name in enumerate(order):
            if self._placed(self._state, name):
                continue
            slot = slots[idx % len(slots)] if slots else self.rack[:2]
            for attempt in range(3):
                done = yield from self._pick_and_place(name, slot, CARRY_Z)
                if done:
                    break
                # try an alternative slot
                if len(slots) > 1:
                    slot = slots[(idx + attempt + 1) % len(slots)]
        # idle
        while True:
            yield np.zeros(11)

    def _pick_and_place(self, name, slot, carry_z):
        s = self._state
        f = self._feat(s, name)
        tri1 = ('triangle_type' in f and f['triangle_type'] > 0.5)
        foot_off = np.array([0.0167, 0.0167]) if tri1 else np.zeros(2)
        # ---- grasp ----
        if not self._grasping(s):
            tgt = self._part_grasp_target(s, name)
            hi = tgt + np.array([0.0, 0.0, 0.12])
            yield from self._move(lambda st, h=hi: h, step=0.06, tol=0.012)
            if self._last_result == 'ikfail':
                return False
            yield from self._move(lambda st, t=tgt: t, step=0.03, tol=2e-3)
            yield from self._grip(-1.0)
            if not self._grasping(self._state):
                for dz in (0.012, -0.012, 0.025):
                    t2 = tgt + np.array([0.0, 0.0, dz])
                    yield from self._move(lambda st, t=t2: t, step=0.012, tol=2e-3)
                    yield from self._grip(-1.0)
                    if self._grasping(self._state):
                        break
            if not self._grasping(self._state):
                return False
        # ---- carry ----
        ref = self._place_ref(self._state, name)

        def part_ref(st):
            return self._ppos(st, name) + rotz(self._yaw(st, name)) @ ref

        def lin_for(st, T):
            off = part_ref(st) - self._linpos_world(st)
            return np.asarray(T, float) - off

        if len(self.part_names) >= 3:
            tx, ty = float(slot[0]), float(slot[1])
        else:
            tx = self.rack[0]
            ty = slot[1] - foot_off[1]
        yield from self._move(lambda st, z=carry_z: lin_for(
            st, [part_ref(st)[0], part_ref(st)[1], z]), step=0.05, tol=6e-3)
        yield from self._move(lambda st: lin_for(st, [tx, ty, carry_z]),
                              step=0.05, tol=5e-3)
        if self._last_result == 'blocked':
            return False
        # ---- descend & release ----
        yield from self._move(lambda st: lin_for(st, [tx, ty, 0.125]),
                              step=0.04, tol=4e-3)
        # descend (gripper still closed) until contact blocks further motion
        z = 0.125
        while z > 0.0885:
            z -= 0.005 if z > 0.107 else 0.002
            yield from self._move(lambda st, zz=z: lin_for(st, [tx, ty, zz]),
                                  step=0.01, tol=1.5e-3,
                                  stop_fn=lambda st: not self._grasping(st))
            if self._last_result in ('blocked', 'stop'):
                break
        # release
        for k in range(6):
            if not self._grasping(self._state):
                break
            yield from self._grip(1.0)
            if not self._grasping(self._state):
                break
            z -= 0.0025
            yield from self._move(lambda st, zz=z: lin_for(st, [tx, ty, zz]),
                                  step=0.01, tol=2e-3, grip=1.0,
                                  stop_fn=lambda st: not self._grasping(st))
        if self._grasping(self._state):
            return False
        if self._ppos(self._state, name)[2] > 0.1015:
            return False
        if self._all_placed(self._state):
            return True
        # ---- retreat ----
        p = self._linpos_world(self._state)
        up = p + np.array([0.0, 0.0, 0.16])
        yield from self._move(lambda st, u=up: u, step=0.06, tol=0.012)
        return True

    def _all_placed(self, s):
        return all(self._placed(s, n) for n in self.part_names)
