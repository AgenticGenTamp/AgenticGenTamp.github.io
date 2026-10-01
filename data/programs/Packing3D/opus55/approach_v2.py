"""Packing3D approach: pick each part by its handle (top-down), place into rack.

Motion: straight-line Cartesian TCP paths, adaptively subdivided so that each
env step moves every joint by <= STEP. Rejected steps (collisions) are detected
and trigger a replan with recovery.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from fk import fk
from ik import ik_pose, tcp_for_part_pose, wrap_near, down_R

STEP = 0.195
GRASP_DZ = 0.055      # TCP height above part center when closing
PRE_DZ = 0.10         # pre-grasp TCP height above part center
CARRY_Z = 0.16        # part center height during transport
PLACE_Z = 0.098       # part center height when releasing
RACK_XY = np.array([0.3, 0.0])


def _handle_offset(state, obj):
    if 'Triangle' in str(obj.type):
        if int(round(state.get(obj, 'triangle_type'))) == 1:
            return np.array([0.0333, 0.0333])
    return np.zeros(2)


def _is_tri(obj):
    return 'Triangle' in str(obj.type)


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
        out = [str(n) for n in state.get_object_names() if str(n).startswith('part')]
        return sorted(out, key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)

    def _part_pose(self, state, name):
        P = state.get_object_from_name(name)
        return np.array([state.get(P, f) for f in ('pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw')])

    def reset(self, state, info):
        self.t0 = time.time()
        self.parts = self._parts(state)
        self.done = set()
        self.cur = None
        self.stage = 'select'
        self.queue = []
        self.prev_q = None
        self.prev_dq = None
        self.goals = self._assign_goals(state)
        self.goal_shift = {nm: np.zeros(2) for nm in self.parts}
        self.nrej = 0
        self.gtf = None

    def _assign_goals(self, state):
        """Goal part-pose xy for each part."""
        names = self.parts
        n = len(names)
        goals = {}
        if n == 1:
            slots = [RACK_XY.copy()]
        else:
            ys = np.linspace(-0.075, 0.075, n) if n <= 2 else np.linspace(-0.1, 0.1, n)
            slots = [np.array([0.3, y]) for y in ys]
        order = sorted(names, key=lambda nm: self._part_pose(state, nm)[1])
        for nm, s in zip(order, slots):
            P = state.get_object_from_name(nm)
            goals[nm] = s - _handle_offset(state, P)
        return goals

    # ------------------------------------------------------------ planning
    def _cart_path(self, q, base, M1, max_iter=40):
        """Straight-line TCP path from current pose to M1. Returns list of q."""
        M0, _ = fk(q, base=base)
        p0, p1 = M0[:3, 3], M1[:3, 3]
        rots = Rotation.from_matrix(np.stack([M0[:3, :3], M1[:3, :3]]))
        slerp = Slerp([0.0, 1.0], rots)

        def pose_at(s):
            M = np.eye(4)
            M[:3, :3] = slerp([s]).as_matrix()[0]
            M[:3, 3] = p0 + s * (p1 - p0)
            return M

        out = []
        s = 0.0
        qc = q.copy()
        it = 0
        while s < 1.0 - 1e-9 and it < max_iter:
            it += 1
            sn = 1.0
            while True:
                qn, pe, _ = ik_pose(pose_at(sn), qc, base=base)
                qn = wrap_near(qn, qc)
                if np.max(np.abs(qn - qc)) <= STEP or sn - s < 1e-3:
                    break
                sn = s + (sn - s) * 0.5
            if np.max(np.abs(qn - qc)) > STEP:
                # split in joint space
                d = qn - qc
                k = int(np.ceil(np.max(np.abs(d)) / STEP))
                for i in range(1, k + 1):
                    out.append(qc + d * i / k)
            else:
                out.append(qn)
            qc = qn
            s = sn
        return out

    def _joint_path(self, q, qt):
        qt = wrap_near(qt, q)
        d = qt - q
        n = max(1, int(np.ceil(np.max(np.abs(d)) / STEP)))
        return [q + d * i / n for i in range(1, n + 1)]

    def _to_actions(self, q, qs, grip=0.0, first_grip=None):
        acts = []
        qc = q
        for i, qn in enumerate(qs):
            g = first_grip if (i == 0 and first_grip is not None) else grip
            acts.append((qn - qc, g))
            qc = qn
        return acts

    def _tcp(self, q, base):
        M, _ = fk(q, base=base)
        return M

    # ------------------------------------------------------------ stages
    def _plan_stage(self, state):
        base, q, ga, gtf = self._robot(state)
        st = self.stage
        if st == 'select':
            remaining = [p for p in self.parts if p not in self.done]
            if not remaining:
                self.stage = 'finished'
                return [(np.zeros(7), 1.0)]
            M = self._tcp(q, base)
            remaining.sort(key=lambda nm: np.linalg.norm(self._part_pose(state, nm)[:2] - M[:2, 3]))
            self.cur = remaining[0]
            self.stage = 'pre'
            return self._plan_stage(state)
        nm = self.cur
        pp = self._part_pose(state, nm)
        P = state.get_object_from_name(nm)
        h = pp[:2] + _handle_offset(state, P)
        if st == 'pre':
            M = self._tcp(q, base)
            qs = []
            qc = q
            if M[2, 3] < pp[2] + PRE_DZ - 0.01:
                Mu = M.copy(); Mu[2, 3] = pp[2] + PRE_DZ + 0.02
                qs += self._cart_path(qc, base, Mu)
                qc = qs[-1]
            Mt = np.eye(4); Mt[:3, :3] = down_R(0.0); Mt[:3, 3] = [h[0], h[1], pp[2] + PRE_DZ]
            qt, _, _ = ik_pose(Mt, qc, base=base)
            qs += self._joint_path(qc, qt)
            return self._to_actions(q, qs, grip=0.0)
        if st == 'down':
            Mt = np.eye(4); Mt[:3, :3] = down_R(0.0); Mt[:3, 3] = [h[0], h[1], pp[2] + GRASP_DZ]
            return self._to_actions(q, self._cart_path(q, base, Mt))
        if st == 'close':
            return [(np.zeros(7), -1.0)]
        if st == 'lift':
            self.gtf = gtf.copy()
            M = self._tcp(q, base)
            Mu = M.copy(); Mu[2, 3] += CARRY_Z - pp[2]
            return self._to_actions(q, self._cart_path(q, base, Mu))
        if st == 'transfer':
            g = self.goals[nm] + self.goal_shift[nm]
            Mt = tcp_for_part_pose([g[0], g[1], CARRY_Z], [0, 0, 0, 1], self.gtf)
            qt, _, _ = ik_pose(Mt, q, base=base)
            return self._to_actions(q, self._joint_path(q, qt))
        if st == 'place':
            g = self.goals[nm] + self.goal_shift[nm]
            Mt = tcp_for_part_pose([g[0], g[1], PLACE_Z], [0, 0, 0, 1], self.gtf)
            return self._to_actions(q, self._cart_path(q, base, Mt))
        if st == 'open':
            return [(np.zeros(7), 1.0)]
        return [(np.zeros(7), 1.0)]

    def _advance(self, state):
        """Called when the queue for the current stage is exhausted."""
        base, q, ga, gtf = self._robot(state)
        st = self.stage
        nm = self.cur
        order = ['pre', 'down', 'close', 'lift', 'transfer', 'place', 'open']
        if st == 'close':
            self.stage = 'lift' if ga else 'pre'
            return
        if st == 'place':
            pz = self._part_pose(state, nm)[2]
            if pz > PLACE_Z + 0.006:
                # did not reach the floor (hit wall/other part): shift toward center
                g = self.goals[nm] + self.goal_shift[nm]
                P = state.get_object_from_name(nm)
                c = self._part_pose(state, nm)[:2] + _handle_offset(state, P)
                d = RACK_XY - c
                self.goal_shift[nm] += 0.015 * d / (np.linalg.norm(d) + 1e-9)
                self.stage = 'transfer'
                return
            self.stage = 'open'
            return
        if st == 'open':
            if ga:
                # lower slightly and retry
                self.stage = 'open'
                self._lower = True
                return
            self.done.add(nm)
            self.stage = 'select'
            return
        if st in order:
            self.stage = order[order.index(st) + 1]

    def get_action(self, state):
        base, q, ga, gtf = self._robot(state)
        # rejection detection
        if self.prev_q is not None and np.max(np.abs(self.prev_dq)) > 1e-6:
            if np.max(np.abs(q - self.prev_q)) < 1e-7:
                self.nrej += 1
                self.queue = []
                if self.stage in ('place',):
                    # blocked while descending: treat as end of descent
                    pass
                elif self.stage not in ('select', 'finished'):
                    # recovery: move straight up a bit, then redo stage
                    M = self._tcp(q, base)
                    Mu = M.copy(); Mu[2, 3] += 0.03
                    qs = self._cart_path(q, base, Mu)
                    self.queue = self._to_actions(q, qs)
                    self._recovering = True
        lower = getattr(self, '_lower', False)
        if lower and not self.queue:
            self._lower = False
            M = self._tcp(q, base)
            Mu = M.copy(); Mu[2, 3] -= 0.002
            qs = self._cart_path(q, base, Mu)
            self.queue = self._to_actions(q, qs) + [(np.zeros(7), 1.0)]
        guard = 0
        while not self.queue and guard < 10:
            guard += 1
            if getattr(self, '_recovering', False):
                self._recovering = False
            elif self.prev_q is not None:
                self._advance(state)
            self.queue = self._plan_stage(state)
        if not self.queue:
            self.queue = [(np.zeros(7), 0.0)]
        dq, grip = self.queue.pop(0)
        a = np.zeros(11, dtype=np.float32)
        a[3:10] = np.clip(dq, -0.2, 0.2)
        a[10] = grip
        self.prev_q = q.copy()
        self.prev_dq = a[3:10].copy()
        return a
