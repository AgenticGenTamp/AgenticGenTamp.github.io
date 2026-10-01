"""Packing3D approach v3: step-optimized pick & place.

- Config c = [base_x, base_y, base_rot, q1..q7]; each env step can change every
  entry by <= 0.2, so a move costs ceil(max|dc| / 0.2) steps.
- IK (ik10.solve) minimizes max|dc| using base + arm.
- Each pick/place is planned from a few waypoint templates; intermediate step
  endpoints are checked against a simple clearance model; the cheapest feasible
  plan is used. Gripper close/open is merged into the final step.
- Rejected steps (collisions) trigger replanning in conservative mode.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from fk import fk
from ik10 import solve, tcp, QLO, QHI
from ik import down_R

STEP = 0.199
GRASP_DZ = 0.055
PLACE_Z = 0.098
FLOOR = (0.21, 0.39, -0.14, 0.14)
RACK = (0.2, 0.4, -0.15, 0.15)
MARGIN = 0.007

# local footprint vertices relative to the handle (centroid) at yaw 0
FOOT = {
    'cub': np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]),
    't0': np.array([[-0.05, -0.028868], [0.05, -0.028868], [0.0, 0.057735]]),
    't1': np.array([[-0.033333, -0.033333], [0.066667, -0.033333], [-0.033333, 0.066667]]),
}
HANDLE_OFF = {'cub': np.zeros(2), 't0': np.zeros(2), 't1': np.array([0.033333, 0.033333])}


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def quat_R(q):
    return Rotation.from_quat(q).as_matrix()


def yaw_of(q):
    R = quat_R(q)
    return np.arctan2(R[1, 0], R[0, 0])


def poly_sep(A, B):
    """Max separation distance along SAT axes (positive => disjoint by that much)."""
    best = -np.inf
    for P in (A, B):
        n = len(P)
        for i in range(n):
            e = P[(i + 1) % n] - P[i]
            ax = np.array([-e[1], e[0]]); ax /= np.linalg.norm(ax) + 1e-12
            pa = A @ ax; pb = B @ ax
            sep = max(pb.min() - pa.max(), pa.min() - pb.max())
            best = max(best, sep)
    return best


def pose_mat(p, q):
    M = np.eye(4); M[:3, :3] = quat_R(q); M[:3, 3] = p; return M


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    # ---------------------------------------------------------------- state
    def _robot(self, state):
        R = state.get_object_from_name('robot')
        c = np.array([state.get(R, f) for f in ('pos_base_x', 'pos_base_y', 'pos_base_rot')] +
                     [state.get(R, f'joint_{i}') for i in range(1, 8)])
        ga = state.get(R, 'grasp_active') > 0.5
        gtf = np.array([state.get(R, f'grasp_tf_{k}') for k in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')])
        return c, ga, gtf

    def _part_info(self, state, nm):
        P = state.get_object_from_name(nm)
        pose = np.array([state.get(P, f) for f in ('pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw')])
        if 'Triangle' in str(P.type):
            kind = 't1' if int(round(state.get(P, 'triangle_type'))) == 1 else 't0'
        else:
            kind = 'cub'
        yaw = yaw_of(pose[3:7])
        R = quat_R(pose[3:7])
        off3 = np.array([HANDLE_OFF[kind][0], HANDLE_OFF[kind][1], 0.0])
        h = pose[:3] + R @ off3
        return dict(name=nm, kind=kind, pose=pose, yaw=yaw, handle=h,
                    poly=h[:2] + FOOT[kind] @ rot2(yaw).T,
                    rad=np.max(np.linalg.norm(FOOT[kind], axis=1)))

    def reset(self, state, info):
        self.t0 = time.time()
        self.parts = sorted([str(n) for n in state.get_object_names() if str(n).startswith('part')],
                            key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)
        self.done = {}
        self.cur = None
        self.mode = 'pick'      # pick -> check_grasp -> place -> check_release
        self.queue = []
        self.prev_c = None
        self.prev_dc = None
        self.conservative = 0
        self.nrej = 0
        self.goal = None
        self.lower_tries = 0
        self.fallback_lifts = 0

    # ---------------------------------------------------------------- goals
    def _goal_candidates(self, kind, occupied):
        cands = []
        xs = np.arange(FLOOR[0], FLOOR[1] + 1e-9, 0.01)
        ys = np.arange(FLOOR[2], FLOOR[3] + 1e-9, 0.01)
        for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
            F = FOOT[kind] @ rot2(yaw).T
            lo = F.min(0); hi = F.max(0)
            for x in xs:
                if x + lo[0] < FLOOR[0] + MARGIN or x + hi[0] > FLOOR[1] - MARGIN:
                    continue
                for y in ys:
                    if y + lo[1] < FLOOR[2] + MARGIN or y + hi[1] > FLOOR[3] - MARGIN:
                        continue
                    poly = np.array([x, y]) + F
                    if all(poly_sep(poly, o) > MARGIN for o in occupied):
                        cands.append((np.array([x, y]), yaw, poly))
        return cands

    def _choose_goals(self, state, info_cur, others_todo, placed_polys):
        """Candidate goals for the current part (list sorted by heuristic)."""
        cands = self._goal_candidates(info_cur['kind'], placed_polys)
        h = info_cur['handle'][:2]

        def score(c):
            dyaw = abs((c[1] - info_cur['yaw'] + np.pi) % (2 * np.pi) - np.pi)
            return np.linalg.norm(c[0] - h) + 0.02 * dyaw
        cands.sort(key=score)
        out = []
        seen = {}
        for g in cands:
            k = round(g[1], 2)
            if seen.get(k, 0) >= 2:
                continue
            occ = placed_polys + [g[2]]
            if all(self._goal_candidates(o['kind'], occ) for o in others_todo):
                out.append(g); seen[k] = seen.get(k, 0) + 1
            if len(out) >= 5:
                break
        if not out:
            out = cands[:3]
        return out

    # ---------------------------------------------------------------- planning
    @staticmethod
    def _nsteps(d):
        return max(1, int(np.ceil(np.max(np.abs(d)) / STEP - 1e-9)))

    def _plan(self, c0, waypoints, check, nsub=8):
        """waypoints: list of (pos, R or None(yaw-free down)). Returns list of configs or None.
        Every step is checked at nsub sub-points (env checks the swept path)."""
        seq = []
        cc = c0
        for i, (p, R) in enumerate(waypoints):
            yaw_free = R is None
            cn, md, err = solve(cc, np.asarray(p, float), R if R is not None else down_R(0.0), yaw_free=yaw_free)
            if err > 1e-3:
                return None
            d = cn - cc
            n = self._nsteps(d)
            for k in range(1, n + 1):
                for j in range(1, nsub + 1):
                    if not check(cc + d * (k - 1 + j / nsub) / n):
                        return None
                seq.append(cc + d * k / n)
            cc = cn
        return seq

    def _obstacles(self, state, exclude):
        obs = []
        for nm in self.parts:
            if nm == exclude:
                continue
            obs.append(self._part_info(state, nm))
        return obs

    @staticmethod
    def _in_box(xy, box, pad):
        return box[0] - pad < xy[0] < box[1] + pad and box[2] - pad < xy[1] < box[3] + pad

    def _check_free(self, obst, cons):
        """Clearance check for the (empty) gripper TCP."""
        extra = 0.03 * cons

        def f(c):
            M = tcp(c)
            p = M[:3, 3]
            if p[2] < 0.115 + extra:
                return False
            for o in obst:
                dxy = np.linalg.norm(p[:2] - o['handle'][:2])
                if dxy < 0.012 and p[2] >= o['pose'][2] + 0.045:
                    return True  # aligned above a handle (grasp / release pose)
            for o in obst:
                if np.linalg.norm(p[:2] - o['handle'][:2]) < 0.08 + o['rad'] * 0.5 and p[2] < o['pose'][2] + 0.10 + extra:
                    return False
            if self._in_box(p[:2], RACK, 0.04):
                inner = self._in_box(p[:2], FLOOR, -0.035)
                if p[2] < (0.13 if inner else 0.165) + extra:
                    return False
            return True
        return f

    def _check_hold(self, obst, G, kind, off3, cons, c0=None):
        extra = 0.03 * cons
        foot = FOOT[kind]
        thr = [0.004] * len(obst)
        if c0 is not None:
            M = tcp(c0) @ G
            cen = M[:3, 3] + M[:3, :3] @ off3
            yaw = np.arctan2(M[1, 0], M[0, 0])
            poly = cen[:2] + foot @ rot2(yaw).T
            thr = [min(0.004, max(0.0, poly_sep(poly, o['poly']) - 0.0004)) for o in obst]

        def f(c):
            T = tcp(c)
            M = T @ G
            pz = M[2, 3]
            cen = M[:3, 3] + M[:3, :3] @ off3
            yaw = np.arctan2(M[1, 0], M[0, 0])
            poly = cen[:2] + foot @ rot2(yaw).T
            if pz < 0.089:
                return False
            lo = poly.min(0); hi = poly.max(0)
            if hi[0] > RACK[0] - 0.005 and lo[0] < RACK[1] + 0.005 and hi[1] > RACK[2] - 0.005 and lo[1] < RACK[3] + 0.005:
                inside = (lo[0] > FLOOR[0] + 0.001 and hi[0] < FLOOR[1] - 0.001 and
                          lo[1] > FLOOR[2] + 0.001 and hi[1] < FLOOR[3] - 0.001)
                if not ((inside and pz >= 0.0955) or pz >= 0.14 + extra):
                    return False
            for o, th in zip(obst, thr):
                if pz < o['pose'][2] + 0.022 + extra and poly_sep(poly, o['poly']) < th:
                    return False
                if np.linalg.norm(T[:2, 3] - o['handle'][:2]) < 0.07 and T[2, 3] < o['pose'][2] + 0.10:
                    return False
            if T[2, 3] < 0.115:
                return False
            return True
        return f

    def _plan_pick(self, state, c):
        info = self._part_info(state, self.cur)
        obst = self._obstacles(state, None)
        others = [o for o in obst if o['name'] != self.cur]
        chk = self._check_free(obst, self.conservative)
        h = info['handle']
        gz = info['pose'][2] + GRASP_DZ
        M = tcp(c)
        templates = []
        for yoff in (0.0, np.pi):
            Rg = down_R(info['yaw'] + yoff)
            if self.conservative == 0:
                templates.append([(h[:2].tolist() + [gz], Rg)])
            for d in (0.04, 0.08):
                templates.append([(h[:2].tolist() + [gz + d], Rg), (h[:2].tolist() + [gz], Rg)])
            up = M[:3, 3] + np.array([0, 0, 0.08 + 0.04 * self.conservative])
            templates.append([(up, None), (h[:2].tolist() + [gz + 0.08], Rg), (h[:2].tolist() + [gz], Rg)])
        best = None
        for tpl in templates:
            seq = self._plan(c, tpl, chk if tpl is not templates[-1] or True else None)
            if seq is not None and (best is None or len(seq) < len(best)):
                best = seq
            if best is not None and len(best) <= 2:
                break
        if best is None:
            # unchecked fallback
            best = self._plan(c, templates[-1], lambda cc: True)
        if best is None:
            return [(np.zeros(10), 0.0)]
        return self._to_actions(c, best, last_grip=-1.0)

    def _plan_place(self, state, c, gtf):
        info = self._part_info(state, self.cur)
        others = self._obstacles(state, self.cur)
        G = pose_mat(gtf[:3], gtf[3:7])
        off3 = np.array([HANDLE_OFF[info['kind']][0], HANDLE_OFF[info['kind']][1], 0.0])
        chk = self._check_hold(others, G, info['kind'], off3, self.conservative, c0=c)
        placed = [self._part_info(state, nm)['poly'] for nm in self.done]
        todo = [self._part_info(state, nm) for nm in self.parts if nm not in self.done and nm != self.cur]
        if self.goal is None or self.conservative:
            goals = self._choose_goals(state, info, todo, placed)
        else:
            goals = [self.goal]
        # current part rotation relative to its yaw (keep tilt as is)
        Rp = quat_R(info['pose'][3:7])
        best = None
        Ginv = np.linalg.inv(G)
        M = tcp(c)
        for gi, (cxy, yaw, poly) in enumerate(goals):
            dyaw = yaw - info['yaw']
            Rg = np.eye(3); Rg[:2, :2] = rot2(dyaw)
            Rpart = Rg @ Rp
            pose_xy = cxy - (Rpart @ off3)[:2]

            def tcp_for(z):
                P = np.eye(4); P[:3, :3] = Rpart; P[:3, 3] = [pose_xy[0], pose_xy[1], z]
                T = P @ Ginv
                return (T[:3, 3], T[:3, :3])
            templates = []
            if self.conservative == 0:
                templates.append([tcp_for(PLACE_Z)])
            for d in (0.04, 0.07):
                templates.append([tcp_for(PLACE_Z + d), tcp_for(PLACE_Z)])
            up = (M[:3, 3] + np.array([0, 0, 0.07 + 0.04 * self.conservative]), M[:3, :3])
            templates.append([up, tcp_for(PLACE_Z + 0.06), tcp_for(PLACE_Z)])
            for tpl in templates:
                seq = self._plan(c, tpl, chk)
                if seq is not None and (best is None or len(seq) < len(best[0])):
                    best = (seq, (cxy, yaw, poly))
                if seq is not None:
                    break
            if time.time() - self.t0 > 40:
                break
        if best is None and self.fallback_lifts < 4:
            self.fallback_lifts += 1
            seq = []
            cc = c
            for k in range(1, 4):
                cn, _, _ = solve(cc, M[:3, 3] + np.array([0, 0, 0.01 * k]), M[:3, :3], use_base=False)
                seq.append(cn); cc = cn
            self.mode = 'place'
            return self._to_actions(c, seq)
        if best is None:
            cxy, yaw, poly = goals[0]
            dyaw = yaw - info['yaw']
            Rg = np.eye(3); Rg[:2, :2] = rot2(dyaw)
            Rpart = Rg @ Rp
            pose_xy = cxy - (Rpart @ off3)[:2]
            P = np.eye(4); P[:3, :3] = Rpart
            seqs = []
            for z in (0.2, PLACE_Z):
                P[:3, 3] = [pose_xy[0], pose_xy[1], z]
                T = P @ Ginv
                seqs.append((T[:3, 3].copy(), T[:3, :3].copy()))
            up = (M[:3, 3] + np.array([0, 0, 0.1]), M[:3, :3])
            seq = self._plan(c, [up] + seqs, lambda cc: True)
            if seq is None:
                return [(np.zeros(10), 0.0)]
            best = (seq, goals[0])
        self.goal = best[1]
        return self._to_actions(c, best[0], last_grip=1.0)

    def _to_actions(self, c, seq, last_grip=0.0):
        acts = []
        cc = c
        for i, cn in enumerate(seq):
            g = last_grip if i == len(seq) - 1 else 0.0
            acts.append((cn - cc, g))
            cc = cn
        return acts

    # ---------------------------------------------------------------- control
    def _over_floor(self, info):
        P = info['poly']
        return (P[:, 0].min() > FLOOR[0] - 0.002 and P[:, 0].max() < FLOOR[1] + 0.002 and
                P[:, 1].min() > FLOOR[2] - 0.002 and P[:, 1].max() < FLOOR[3] + 0.002)

    def _in_rack(self, info):
        return self._over_floor(info) and info['pose'][2] < 0.104

    def _select(self, state, c):
        self.done = {nm: True for nm in self.parts if self._in_rack(self._part_info(state, nm))}
        remaining = [p for p in self.parts if p not in self.done]
        if not remaining:
            return None
        M = tcp(c)
        remaining.sort(key=lambda nm: np.linalg.norm(self._part_info(state, nm)['handle'][:2] - M[:2, 3]))
        return remaining[0]

    def get_action(self, state):
        c, ga, gtf = self._robot(state)
        rejected = False
        if self.prev_c is not None and np.max(np.abs(self.prev_dc)) > 1e-6:
            if np.max(np.abs(c - self.prev_c)) < 1e-7:
                rejected = True
                self.nrej += 1
        if rejected:
            self.queue = []
            self.conservative = min(3, self.conservative + 1)
            if self.mode == 'placing' and ga:
                # blocked while descending near the goal?
                info = self._part_info(state, self.cur)
                if self._over_floor(info) and info['pose'][2] < PLACE_Z + 0.006:
                    self.queue = [(np.zeros(10), 1.0)]
        if not self.queue:
            self.queue = self._next(state, c, ga, gtf)
        dc, grip = self.queue.pop(0)
        dc = np.clip(dc, -0.2, 0.2)
        a = np.zeros(11, dtype=np.float32)
        a[:10] = dc
        a[10] = grip
        self.prev_c = c.copy()
        self.prev_dc = dc.copy()
        return a

    def _next(self, state, c, ga, gtf):
        for _ in range(4):
            if self.mode == 'pick':
                if ga:
                    self.mode = 'place'
                    continue
                if self.cur is None or self.cur in self.done:
                    self.cur = self._select(state, c)
                    self.goal = None
                    if self.cur is None:
                        return [(np.zeros(10), 1.0)]
                self.last_pick_mode = True
                self.mode = 'picking'
                return self._plan_pick(state, c)
            if self.mode == 'picking':
                if ga:
                    self.conservative = 0
                    self.fallback_lifts = 0
                    self.mode = 'place'
                    continue
                self.conservative = min(3, self.conservative + 1)
                self.mode = 'pick'
                continue
            if self.mode == 'place':
                if not ga:
                    self.mode = 'pick'
                    continue
                self.mode = 'placing'
                self.lower_tries = 0
                return self._plan_place(state, c, gtf)
            if self.mode == 'placing':
                info = self._part_info(state, self.cur)
                if not ga:
                    if self._in_rack(info):
                        self.done[self.cur] = True
                        self.conservative = 0
                        self.cur = None
                    self.mode = 'pick'
                    continue
                # still holding
                if self._over_floor(info) and info['pose'][2] < PLACE_Z + 0.006 and self.lower_tries < 4:
                    self.lower_tries += 1
                    M = tcp(c)
                    cn, _, _ = solve(c, M[:3, 3] - np.array([0, 0, 0.002]), M[:3, :3])
                    return [(cn - c, 1.0)]
                if info['pose'][2] > 0.11:
                    # stuck on rim / other part: choose another goal
                    self.goal = None
                    self.conservative = max(1, self.conservative)
                self.mode = 'place'
                continue
        return [(np.zeros(10), 0.0)]
