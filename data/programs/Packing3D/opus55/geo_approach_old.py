"""Packing3D approach: pick each part by its handle (top-down), place into rack."""
import numpy as np
from fk import fk
from ik import ik, ik_pose, tcp_for_part_pose, wrap_near

RACK_XY = np.array([0.3, 0.0])
STEP = 0.195
GRASP_DZ = 0.055      # TCP height above part center when closing
PRE_DZ = 0.11         # pre-grasp TCP height above part center
CARRY_Z = 0.17        # part center height during transport
PLACE_Z = 0.098       # part center height when releasing


def _handle_offset(state, obj):
    if 'Triangle' in str(obj.type):
        if int(round(state.get(obj, 'triangle_type'))) == 1:
            return np.array([0.0333, 0.0333])
    return np.zeros(2)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    # ------------------------------------------------------------------ utils
    def _robot(self, state):
        R = state.get_object_from_name('robot')
        base = np.array([state.get(R, f) for f in ('pos_base_x', 'pos_base_y', 'pos_base_rot')])
        q = np.array([state.get(R, f'joint_{i}') for i in range(1, 8)])
        ga = state.get(R, 'grasp_active') > 0.5
        gtf = np.array([state.get(R, f'grasp_tf_{k}') for k in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')])
        return base, q, ga, gtf

    def _parts(self, state):
        out = []
        for name in state.get_object_names():
            name = str(name)
            if name.startswith('part'):
                out.append(name)
        return sorted(out, key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)

    def _part_pose(self, state, name):
        P = state.get_object_from_name(name)
        return np.array([state.get(P, f) for f in ('pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw')])

    def reset(self, state, info):
        self.parts = self._parts(state)
        self.done = set()
        self.phase = 'pick'
        self.cur = None
        self.queue = []          # list of (dq(7), grip)
        self.last_q = None
        self.expect_move = False
        self.stage = None
        self.goals = self._assign_goals(state)
        self.fail = 0

    def _assign_goals(self, state):
        names = self.parts
        goals = {}
        n = len(names)
        # slots along y in rack
        if n == 1:
            slots = [np.array([0.3, 0.0])]
        else:
            ys = np.linspace(-0.075, 0.075, n) if n <= 2 else np.linspace(-0.1, 0.1, n)
            slots = [np.array([0.3, y]) for y in ys]
        # assign by y ordering of parts
        order = sorted(names, key=lambda nm: self._part_pose(state, nm)[1])
        for nm, s in zip(order, slots):
            goals[nm] = s  # goal for the handle position
        return goals

    # ------------------------------------------------------------ planning
    def _plan_to(self, q, qt, grip=0.0):
        qt = wrap_near(qt, q)
        d = qt - q
        n = max(1, int(np.ceil(np.max(np.abs(d)) / STEP)))
        return [(d / n, grip) for _ in range(n)]

    def _ik_pos(self, pos, q, base, yaw=0.0):
        qs, pe, _ = ik(pos, q, base=base, yaw=yaw)
        return qs, pe

    def _next_stage(self, state):
        base, q, ga, gtf = self._robot(state)
        if self.cur is None:
            remaining = [p for p in self.parts if p not in self.done]
            if not remaining:
                return [(np.zeros(7), 1.0)]
            # choose closest part to current TCP
            M, _ = fk(q, base=base)
            remaining.sort(key=lambda nm: np.linalg.norm(self._part_pose(state, nm)[:2] - M[:2, 3]))
            self.cur = remaining[0]
            self.stage = 'pre'
        nm = self.cur
        pp = self._part_pose(state, nm)
        P = state.get_object_from_name(nm)
        h = pp[:2] + _handle_offset(state, P)
        st = self.stage
        if st == 'pre':
            M, _ = fk(q, base=base)
            plan = []
            if M[2, 3] < 0.2:  # retreat upward first
                qs, _ = self._ik_pos([M[0, 3], M[1, 3], M[2, 3] + 0.08], q, base)
                plan += self._plan_to(q, qs)
                q = wrap_near(qs, q)
            qs, pe = self._ik_pos([h[0], h[1], pp[2] + PRE_DZ], q, base)
            plan += self._plan_to(q, qs, grip=1.0)
            self.stage = 'down'
            return plan
        if st == 'down':
            qs, pe = self._ik_pos([h[0], h[1], pp[2] + GRASP_DZ], q, base)
            self.stage = 'close'
            return self._plan_to(q, qs)
        if st == 'close':
            self.stage = 'check_grasp'
            return [(np.zeros(7), -1.0)]
        if st == 'check_grasp':
            if not ga:
                self.stage = 'pre'
                self.fail += 1
                return self._next_stage(state)
            self.gtf = gtf.copy()
            self.stage = 'lift'
            return self._next_stage(state)
        if st == 'lift':
            M, _ = fk(q, base=base)
            qs, _ = self._ik_pos([M[0, 3], M[1, 3], M[2, 3] + (CARRY_Z - pp[2])], q, base)
            plan = self._plan_to(q, qs)
            q2 = wrap_near(qs, q)
            g = self.goals[nm] - _handle_offset(state, P)
            Mt = tcp_for_part_pose([g[0], g[1], CARRY_Z], pp[3:7], self.gtf)
            qs2, _, _ = ik_pose(Mt, q2, base=base)
            plan += self._plan_to(q2, qs2)
            self.stage = 'place'
            return plan
        if st == 'place':
            g = self.goals[nm] - _handle_offset(state, P)
            Mt = tcp_for_part_pose([g[0], g[1], PLACE_Z], pp[3:7], self.gtf)
            qs, _, _ = ik_pose(Mt, q, base=base)
            self.stage = 'open'
            return self._plan_to(q, qs)
        if st == 'open':
            self.stage = 'check_open'
            return [(np.zeros(7), 1.0)]
        if st == 'check_open':
            if ga:
                # not released: lower a bit
                M, _ = fk(q, base=base)
                qs, _ = self._ik_pos([M[0, 3], M[1, 3], M[2, 3] - 0.002], q, base, yaw=None)
                self.stage = 'open'
                return self._plan_to(q, qs)
            self.done.add(nm)
            self.cur = None
            return self._next_stage(state)
        return [(np.zeros(7), 0.0)]

    def get_action(self, state):
        base, q, ga, gtf = self._robot(state)
        if not self.queue:
            self.queue = self._next_stage(state)
        dq, grip = self.queue.pop(0)
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = np.clip(dq, -0.2, 0.2)
        a[10] = grip
        return a
