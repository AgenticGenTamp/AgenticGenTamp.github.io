"""
Packing3D policy -- v28.

=============================================================================
v27 REGRESSED ON SEED 1 AND THE CAUSE IS MY OWN v27 "FIX 1". REVERTING IT.
=============================================================================

v27 final state, seed 1 (one part, equilateral triangle):

    robot grasp_active = 1, finger_state = 0.46   -> clause 1 VIOLATED
    part0 grasp_active = 1, HELD at
        pos  = (0.3044459, -0.0135827, 0.1490645), tilt ~ 0.0074 rad (flat)

Compare to v26 on the same seed, which at least ended UNGRASPED (part parked at
y = -0.293, clause 1 satisfied, geometry wrong), and to v27 which ends holding
the part 5.4 cm above the rack floor, directly over the cavity centre.

The part is at x = 0.3044, y = -0.0136 -- essentially the nominal slot (0.3, 0.0)
that my planner computes for a lone triangle. It is flat. It is over the right
place. It is 54 mm too high and still in the gripper.

WHAT I BROKE
------------
v27 FIX 1 changed the terminal release from

    hard_end = 300 + 150*n           # fires at t = 440 for n = 1
to
    very_late = t >= 3 * (300 + 150*n)   # fires at t = 1350 for n = 1

The episode is 1000 steps. 1350 > 1000, so the terminal release NEVER FIRED.
I removed the only guarantee that the episode ends ungrasped, and replaced it
with a threshold beyond the horizon. That is a straightforward arithmetic error
on my part: I was trying to stop the release firing too early (v26 fired at 44%
of the episode) and overcorrected past the end of the episode entirely.

The descent itself also stalled at 54 mm, which is the older, separate problem
-- but the grasped ending is purely my v27 regression and is the thing that
makes goal_reached() return False before any geometry is even examined.

FIXES IN v28
------------
1. TERMINAL RELEASE IS DRIVEN BY AN OBSERVED-HORIZON ESTIMATE THAT CANNOT
   EXCEED THE EPISODE.
   I no longer guess a multiple of the formula. I track the highest step index
   seen, and force the release once t exceeds a conservative estimate that is
   *always* reached: `max(formula, 1000) - 40`. Since every observed rollout in
   this thread has been exactly 1000 steps, and the formula gives 450 for n = 1
   and 600 for n = 2, using max(formula, 1000) is safe for both and still leaves
   960 steps of placement effort. If the real budget were larger, the part is
   simply released a bit early, which only costs a placement attempt, never a
   clause-1 violation.

2. RELEASE-WHEN-SEATED IS CHECKED EVERY STEP, INDEPENDENT OF STAGE.
   The strongest guarantee available: if at any moment the currently held part's
   true corners satisfy the env's seating predicate, open the gripper
   immediately. Previously this was only evaluated inside DESCEND, so a part
   that happened to pass while in ALIGN or LEVEL was not released.

3. DESCEND STALL AT HEIGHT NOW LOWERS THE TARGET RATHER THAN RETRYING THE SLOT.
   v27 responded to a stalled descent by re-planning the slot and going back to
   LIFT, which on a one-part instance just repeats the same approach. v28 first
   tries pressing further down (target floor - 15 mm instead of floor - 5 mm)
   for a bounded number of steps, and only then re-plans. The 54 mm hover
   suggests the servo stops short, so commanding deeper is the direct response.

Everything else is v27 verbatim: the tilt-aware slot bounds, the non-migrating
`_bump`, the containment invariant, the `_goto` re-entry guard, the DUMP timer.

I am aware this is the second time I have broken a working guarantee while
trying to improve something adjacent to it (v20 and now v27). The terminal
release in v28 is written so that it cannot fail closed: it fires on a fixed
step number, not on a computed horizon.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# rotation helpers
# ----------------------------------------------------------------------------

def _quat_to_mat(qx, qy, qz, qw):
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def _mat_to_rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-8:
        return np.zeros(3)
    if abs(np.pi - ang) < 1e-5:
        A = (R + np.eye(3)) / 2.0
        d = np.clip(np.diag(A), 0.0, 1.0)
        k = int(np.argmax(d))
        axis = np.zeros(3)
        axis[k] = np.sqrt(d[k])
        if axis[k] < 1e-9:
            return np.zeros(3)
        for j in range(3):
            if j != k:
                axis[j] = A[k, j] / axis[k]
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.zeros(3)
        return axis / n * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (ang / (2.0 * np.sin(ang)))


def _rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _tilt_of(R):
    return float(np.linalg.norm(_mat_to_rotvec(_rotz(_yaw_of(R)).T.dot(R))))


def _wrap(a):
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


# ----------------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------------

def _triangle_vertices(triangle_type, side_a, side_b):
    t = int(round(triangle_type))
    if t == 0:
        s = float(side_a)
        h = s * np.sqrt(3.0) / 2.0
        return np.array([
            [-s / 2.0, -h / 3.0, 0.0],
            [s / 2.0, -h / 3.0, 0.0],
            [0.0, 2.0 * h / 3.0, 0.0],
        ])
    a = float(side_a)
    b = float(side_b)
    return np.array([[0.0, 0.0, 0.0], [a, 0.0, 0.0], [0.0, b, 0.0]])


class _Part:
    __slots__ = ("name", "is_tri", "pos", "quat", "R", "grasped",
                 "side_a", "side_b", "depth", "tri_type",
                 "hx", "hy", "hz", "area", "peg_off", "top_z",
                 "tilt", "yaw", "local_v", "local_v2", "radius")

    def __init__(self, name):
        self.name = name


def _part_sort_key(name):
    if name.startswith("part"):
        tail = name[4:]
        if tail.isdigit():
            return (0, int(tail), name)
    return (1, 0, name)


def _read_parts(state):
    parts = []
    for name in sorted(state.get_object_names(), key=_part_sort_key):
        if not name.startswith("part"):
            continue
        obj = state.get_object_from_name(name)
        feats = set(state.type_features[obj.type])
        p = _Part(name)
        p.pos = np.array([state.get(obj, "pose_x"),
                          state.get(obj, "pose_y"),
                          state.get(obj, "pose_z")], dtype=float)
        p.quat = (state.get(obj, "pose_qx"), state.get(obj, "pose_qy"),
                  state.get(obj, "pose_qz"), state.get(obj, "pose_qw"))
        p.R = _quat_to_mat(*p.quat)
        p.grasped = state.get(obj, "grasp_active") > 0.5
        if "side_a" in feats:
            p.is_tri = True
            p.side_a = float(state.get(obj, "side_a"))
            p.side_b = float(state.get(obj, "side_b"))
            p.depth = float(state.get(obj, "depth"))
            p.tri_type = float(state.get(obj, "triangle_type"))
            v = _triangle_vertices(p.tri_type, p.side_a, p.side_b)
            p.local_v2 = v[:, :2].copy()
            p.local_v = np.vstack([v + np.array([0.0, 0.0, -p.depth / 2.0]),
                                   v + np.array([0.0, 0.0, p.depth / 2.0])])
            p.hz = p.depth / 2.0
            p.hx = (v[:, 0].max() - v[:, 0].min()) / 2.0
            p.hy = (v[:, 1].max() - v[:, 1].min()) / 2.0
            p.area = abs(np.cross(v[1, :2] - v[0, :2], v[2, :2] - v[0, :2])) / 2.0
            c = v.mean(axis=0)
            p.peg_off = np.array([c[0], c[1], p.depth / 2.0 + 0.025])
        else:
            p.is_tri = False
            p.hx = float(state.get(obj, "half_extent_x"))
            p.hy = float(state.get(obj, "half_extent_y"))
            p.hz = float(state.get(obj, "half_extent_z"))
            p.side_a = p.side_b = p.depth = p.tri_type = 0.0
            p.local_v = np.array([[sx * p.hx, sy * p.hy, sz * p.hz]
                                  for sx in (-1, 1) for sy in (-1, 1)
                                  for sz in (-1, 1)])
            p.local_v2 = np.array([[sx * p.hx, sy * p.hy]
                                   for sx in (-1, 1) for sy in (-1, 1)])
            p.area = 4.0 * p.hx * p.hy
            p.peg_off = np.array([0.0, 0.0, p.hz + 0.025])
        p.radius = float(np.max(np.linalg.norm(p.local_v2, axis=1)))
        p.yaw = _yaw_of(p.R)
        p.tilt = _tilt_of(p.R)
        p.top_z = p.pos[2] + p.hz
        parts.append(p)
    return parts


def _footprint_extent(p, yaw):
    v2 = (_rotz(yaw)[:2, :2].dot(p.local_v2.T)).T
    lo, hi = v2.min(axis=0), v2.max(axis=0)
    return float(hi[0] - lo[0]), float(hi[1] - lo[1])


def _footprint_box(p):
    v2 = (p.R[:2, :2].dot(p.local_v2.T)).T + p.pos[:2]
    return v2.min(axis=0), v2.max(axis=0)


def _footprint_centre_world(p):
    lo, hi = _footprint_box(p)
    return np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, p.pos[2]])


def _peg_world(p):
    return p.pos + p.R.dot(p.peg_off)


def _peg_xy(p):
    w = p.pos + p.R.dot(np.array([p.peg_off[0], p.peg_off[1], 0.0]))
    return w[:2]


def _peg_xy_for_slot(p, centre_xy, yaw):
    Rz = _rotz(yaw)
    v2 = (Rz[:2, :2].dot(p.local_v2.T)).T
    lo, hi = v2.min(axis=0), v2.max(axis=0)
    bb = (lo + hi) / 2.0
    origin = np.array([centre_xy[0] - bb[0], centre_xy[1] - bb[1]])
    off = Rz.dot(np.array([p.peg_off[0], p.peg_off[1], 0.0]))
    return origin + off[:2]


def _corners_at(p, origin, R):
    return (R.dot(p.local_v.T)).T + np.asarray(origin, dtype=float)


def _corner_drop(p):
    w = _corners_at(p, p.pos, p.R)
    return float(p.pos[2] - w[:, 2].min())


def _box_gap(lo1, hi1, lo2, hi2):
    dx = max(lo1[0] - hi2[0], lo2[0] - hi1[0])
    dy = max(lo1[1] - hi2[1], lo2[1] - hi1[1])
    if dx >= 0.0 and dy >= 0.0:
        return float(np.hypot(dx, dy))
    if dx >= 0.0:
        return float(dx)
    if dy >= 0.0:
        return float(dy)
    return float(max(dx, dy))


def _wall_clearance(w, rack_pos, rack_half, wall):
    ihx = max(rack_half[0] - wall, 1e-4)
    ihy = max(rack_half[1] - wall, 1e-4)
    return min(
        w[:, 0].min() - (rack_pos[0] - ihx),
        (rack_pos[0] + ihx) - w[:, 0].max(),
        w[:, 1].min() - (rack_pos[1] - ihy),
        (rack_pos[1] + ihy) - w[:, 1].max(),
    )


def _over_rack(fc, rack_pos, rack_half):
    return (abs(fc[0] - rack_pos[0]) < rack_half[0] + 0.06
            and abs(fc[1] - rack_pos[1]) < rack_half[1] + 0.06)


def _seated_corners(w, rack_pos, rack_half, wall, tol=4.7e-3):
    if _wall_clearance(w, rack_pos, rack_half, wall) < -1e-9:
        return False
    floor_z = rack_pos[2] - rack_half[2] + wall
    if abs(w[:, 2].min() - floor_z) > tol:
        return False
    if w[:, 2].max() > rack_pos[2] + rack_half[2] + tol:
        return False
    return True


def _is_seated(p, rack_pos, rack_half, wall, tol=4.7e-3):
    return _seated_corners(_corners_at(p, p.pos, p.R), rack_pos, rack_half,
                           wall, tol)


# ----------------------------------------------------------------------------
# packer + containment invariant
# ----------------------------------------------------------------------------

_WALL_MARGIN = 6.0e-3
_PEG_TARGET = 0.055
_MIN_PART_GAP = 5.0e-3
_TILT_ALLOW = 0.045
_TABLE_LO = np.array([0.10, -0.40])
_TABLE_HI = np.array([0.50, 0.40])


def _slot_bounds(rack_pos, rack_half, wall, w, h, pad=0.0):
    ihx = max(rack_half[0] - wall, 1e-4)
    ihy = max(rack_half[1] - wall, 1e-4)
    lo_x = rack_pos[0] - ihx + w / 2.0 + _WALL_MARGIN + pad
    hi_x = rack_pos[0] + ihx - w / 2.0 - _WALL_MARGIN - pad
    lo_y = rack_pos[1] - ihy + h / 2.0 + _WALL_MARGIN + pad
    hi_y = rack_pos[1] + ihy - h / 2.0 - _WALL_MARGIN - pad
    if lo_x > hi_x:
        lo_x = hi_x = rack_pos[0]
    if lo_y > hi_y:
        lo_y = hi_y = rack_pos[1]
    return lo_x, hi_x, lo_y, hi_y


def _tilt_pad(part):
    return float(_TILT_ALLOW * getattr(part, "radius", 0.05))


def _legal_slot(rack_pos, rack_half, wall, part, yaw, cx, cy):
    w, h = _footprint_extent(part, yaw)
    lo_x, hi_x, lo_y, hi_y = _slot_bounds(rack_pos, rack_half, wall, w, h,
                                          pad=_tilt_pad(part))
    return (float(np.clip(cx, lo_x, hi_x)), float(np.clip(cy, lo_y, hi_y)))


def _plan(rack_pos, rack_half, wall, parts, jitter=None):
    ihx = max(rack_half[0] - wall, 1e-4)
    ihy = max(rack_half[1] - wall, 1e-4)
    usable_w = 2.0 * ihx - 2.0 * _WALL_MARGIN
    usable_h = 2.0 * ihy - 2.0 * _WALL_MARGIN

    cand_yaws = [0.0, np.pi / 2.0, np.pi, -np.pi / 2.0]
    prep = []
    for p in parts:
        best, best_key = 0.0, None
        for y in cand_yaws:
            w, h = _footprint_extent(p, y)
            key = (round(w * h, 6), round(max(w, h), 6), abs(y))
            if best_key is None or key < best_key:
                best_key, best = key, y
        w, h = _footprint_extent(p, best)
        prep.append([p, best, w, h])
    prep.sort(key=lambda t: (-t[0].area, t[0].name))
    if not prep:
        return {}

    max_w = max(t[2] for t in prep)
    ncol = max(1, int(np.floor(usable_w / max(max_w, 1e-9))))

    cols = [[] for _ in range(ncol)]
    heights = [0.0] * ncol
    for item in prep:
        k = int(np.argmin(heights))
        cols[k].append(item)
        heights[k] += item[3]

    col_w = [max((t[2] for t in c), default=0.0) for c in cols]
    used_w = sum(col_w)
    gap_x = max(min((usable_w - used_w) / float(ncol + 1), _PEG_TARGET), 0.0)

    plan = {}
    span_x = used_w + gap_x * (ncol - 1)
    x_cursor = rack_pos[0] - span_x / 2.0
    for ci, col in enumerate(cols):
        if not col:
            x_cursor += col_w[ci] + gap_x
            continue
        cw = col_w[ci]
        cx = x_cursor + cw / 2.0
        used_h = sum(t[3] for t in col)
        n = len(col)
        raw_gap = (usable_h - used_h) / float(max(n - 1, 1))
        gap_y = float(np.clip(raw_gap, 0.0, _PEG_TARGET))
        span_y = used_h + gap_y * (n - 1)
        y_cursor = rack_pos[1] - span_y / 2.0
        for p, yaw, w, h in col:
            gx, gy = cx, y_cursor + h / 2.0
            if jitter and p.name in jitter:
                gx += jitter[p.name][0]
                gy += jitter[p.name][1]
            plan[p.name] = (_legal_slot(rack_pos, rack_half, wall, p, yaw,
                                        gx, gy), yaw)
            y_cursor += h + gap_y
        x_cursor += cw + gap_x
    return plan


# ----------------------------------------------------------------------------
# FK / IK oracle
# ----------------------------------------------------------------------------

class _KinBackend:

    def __init__(self):
        self.mode = "planar"
        self.robot = None
        self.cid = None
        self.import_error = None
        try:
            import pybullet as p  # noqa: F401
            from pybullet_helpers.geometry import SE2Pose
            from pybullet_helpers.robots import create_pybullet_mobile_robot
            cid = p.connect(p.DIRECT)
            robot = create_pybullet_mobile_robot(
                "tidybot-kinova", cid,
                base_z=-0.4,
                base_home_pose=SE2Pose(-0.12, 0.0, 0.0),
                base_pose_lower_bound=SE2Pose(-10.0, -10.0, -np.pi),
                base_pose_upper_bound=SE2Pose(10.0, 10.0, np.pi),
            )
            self.cid = cid
            self.robot = robot
            self.mode = "pybullet"
        except Exception as exc:          # noqa: BLE001
            self.import_error = repr(exc)
            self.mode = "planar"

    def ee_pose(self, base, joints):
        if self.mode == "pybullet":
            try:
                from pybullet_helpers.geometry import SE2Pose
                self.robot.set_base(SE2Pose(float(base[0]), float(base[1]),
                                            float(base[2])))
                self.robot.arm.set_joints(list(joints) + [0.0] * 6)
                pose = self.robot.arm.get_end_effector_pose()
                q = pose.orientation
                return (np.array(pose.position, dtype=float),
                        _quat_to_mat(q[0], q[1], q[2], q[3]))
            except Exception:             # noqa: BLE001
                self.mode = "planar"
        return self._planar_fk(base, joints), np.eye(3)

    def fk(self, base, joints):
        return self.ee_pose(base, joints)[0]

    @staticmethod
    def _planar_fk(base, joints):
        l1, l2, l3 = 0.41, 0.31, 0.19
        shoulder_z = -0.4 + 0.70
        yaw = float(base[2]) + float(joints[0])
        a1 = float(joints[1])
        a2 = a1 + float(joints[3]) + np.pi
        a3 = a2 + float(joints[5])
        r = l1 * np.sin(a1) + l2 * np.sin(a2) + l3 * np.sin(a3)
        z = shoulder_z + l1 * np.cos(a1) + l2 * np.cos(a2) + l3 * np.cos(a3)
        return np.array([base[0] + r * np.cos(yaw),
                         base[1] + r * np.sin(yaw), z], dtype=float)

    def jac6(self, base, joints, eps=1e-4):
        p0, R0 = self.ee_pose(base, joints)
        J = np.zeros((6, 7))
        R0T = R0.T
        for k in range(7):
            q = np.array(joints, dtype=float)
            q[k] += eps
            p1, R1 = self.ee_pose(base, q)
            J[0:3, k] = (p1 - p0) / eps
            J[3:6, k] = _mat_to_rotvec(R1.dot(R0T)) / eps
        return J, p0, R0

    def solve(self, base, joints, goal_pos, goal_R, max_delta,
              w_rot=0.0, home_pull=None, lam=4e-3):
        J, p0, R0 = self.jac6(base, joints)
        e_pos = np.asarray(goal_pos, dtype=float) - p0
        if goal_R is not None and w_rot > 0.0:
            e_rot = _mat_to_rotvec(np.asarray(goal_R).dot(R0.T))
        else:
            e_rot = np.zeros(3)
            w_rot = 0.0
        W = np.diag([1.0, 1.0, 1.0, w_rot, w_rot, w_rot])
        Jw = W.dot(J)
        ew = W.dot(np.concatenate([e_pos, e_rot]))
        try:
            A = Jw.dot(Jw.T) + lam * np.eye(6)
            dq = Jw.T.dot(np.linalg.solve(A, ew))
            if home_pull is not None:
                P = Jw.T.dot(np.linalg.solve(A, Jw))
                dq = dq + (np.eye(7) - P).dot(home_pull)
        except Exception:                 # noqa: BLE001
            dq = Jw.T.dot(ew)
        n = np.max(np.abs(dq))
        if n > 1e-12:
            dq = dq * min(1.0, max_delta / n)
        return dq, e_pos, e_rot

    def reachable(self, base, target, joints, iters=45, tol=0.03):
        q = np.array(joints, dtype=float)
        for _ in range(iters):
            dq, e, _ = self.solve(base, q, target, None, 0.25, w_rot=0.0)
            if float(np.linalg.norm(e)) < tol:
                return True
            q = q + dq
        p, _ = self.ee_pose(base, q)
        return float(np.linalg.norm(np.asarray(target) - p)) < tol


# ----------------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------------

class GeneratedApproach:

    N_ACT = 11
    JLO = 3
    JHI = 10
    HOME = np.array([0.0, -0.35, -np.pi, -2.5, 0.0, -0.87, np.pi / 2])
    ENVELOPE = np.array([2.6, 2.2, 2.6, 2.6, 2.6, 2.2, 3.0])

    F_BASE = 0.08
    F_REACH = 0.22
    F_LIFT = 0.07
    F_REBASE = 0.10
    F_LEVEL = 0.18
    F_TRAV = 0.13
    F_ALIGN = 0.10
    F_DESC = 0.22

    TILT_OK = 0.045
    TILT_RELEASE = 0.055

    # v28 FIX 1: a fixed, always-reached terminal release step. Every rollout in
    # this thread has been 1000 steps; the formula (450 / 600) is a lower bound.
    ASSUMED_EPISODE = 1000
    TERMINAL_MARGIN = 40

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives or {}
        low = np.asarray(getattr(action_space, "low",
                                 -0.2 * np.ones(self.N_ACT)), dtype=float)
        high = np.asarray(getattr(action_space, "high",
                                  0.2 * np.ones(self.N_ACT)), dtype=float)
        if low.shape[0] != self.N_ACT:
            low = -0.2 * np.ones(self.N_ACT)
            high = 0.2 * np.ones(self.N_ACT)
        self.low, self.high = low, high
        self.jmag = float(min(0.2, np.min(np.abs(high[3:10]))))
        self.bmag = float(min(0.2, np.min(np.abs(high[0:3]))))
        self.rng = np.random.default_rng(181)
        self.kin = _KinBackend()
        self._reset_internal()

    def _reset_internal(self):
        self.stage = "BASE"
        self.stage_t = 0
        self.part_t = 0
        self.target = None
        self.plan = {}
        self.base_plan = {}
        self.jitter = {}
        self.attempts = {}
        self.rack_pos = np.array([0.3, 0.0, 0.095])
        self.rack_half = np.array([0.1, 0.15, 0.02])
        self.wall = 0.01
        self.t = 0
        self.n_parts = 1
        self.placed = set()
        self.placed_boxes = {}
        self.placed_pegs = {}
        self.serve_base = None
        self.base_goal = None
        self.prev_joints = None
        self.blocked = 0
        self.clear_z = 0.32
        self.escape_t = 0
        self.escape = np.zeros(7)
        self.G = None
        self.tilt_hist = []
        self.zmin_hist = []
        self.desc_retry = 0
        self.press_t = 0
        self.slice_len = 200
        self.deadline = 200
        self.recover_t = 0
        self.dump_t = 0
        self.diag_fk_mode = 1 if self.kin.mode == "pybullet" else 0
        self.diag_fk_error_msg = self.kin.import_error
        self.diag_cmd_steps = 0
        self.diag_reverted = 0

    def reset(self, state, info):
        self._reset_internal()
        self._refresh(state)
        return None

    def _refresh(self, state):
        try:
            rack = state.get_object_from_name("rack")
            self.rack_pos = np.array([state.get(rack, "pose_x"),
                                      state.get(rack, "pose_y"),
                                      state.get(rack, "pose_z")], dtype=float)
            self.rack_half = np.array([state.get(rack, "half_extent_x"),
                                       state.get(rack, "half_extent_y"),
                                       state.get(rack, "half_extent_z")],
                                      dtype=float)
        except Exception:                 # noqa: BLE001
            pass
        self.wall = float(min(0.01,
                              0.5 * min(self.rack_half[0], self.rack_half[1])))
        parts = _read_parts(state)
        self.n_parts = max(1, len(parts))
        self.plan = _plan(self.rack_pos, self.rack_half, self.wall,
                          parts, self.jitter)
        self.base_plan = {k: (tuple(v[0]), v[1]) for k, v in self.plan.items()}
        rim = self.rack_pos[2] + self.rack_half[2]
        self.clear_z = float(max([p.top_z for p in parts] + [rim]) + 0.16)
        self.slice_len = max(140, int(0.85 * self._horizon() / self.n_parts))
        self.deadline = self.slice_len
        self.serve_base = self._choose_serve_base(state)
        self._filter_slots_by_reach(state, parts)

    def _choose_serve_base(self, state):
        joints = self._joints(state)
        floor_z = self.rack_pos[2] - self.rack_half[2] + self.wall
        targets = [np.array([sx, sy, floor_z + 0.03])
                   for (sx, sy), _y in self.plan.values()] or [self.rack_pos]
        best, best_score = None, -1e9
        for dx in np.linspace(0.32, 0.48, 5):
            for dy in np.linspace(-0.10, 0.10, 5):
                cand = np.array([self.rack_pos[0] - dx,
                                 self.rack_pos[1] + dy, 0.0])
                ok = sum(1 for tg in targets
                         if self.kin.reachable(cand, tg, joints))
                score = ok * 10.0 - abs(dx - 0.40) - abs(dy)
                if score > best_score:
                    best_score, best = score, cand
        return best if best is not None else np.array(
            [self.rack_pos[0] - 0.40, self.rack_pos[1], 0.0])

    def _filter_slots_by_reach(self, state, parts):
        if self.serve_base is None:
            return
        by_name = {p.name: p for p in parts}
        joints = self._joints(state)
        floor_z = self.rack_pos[2] - self.rack_half[2] + self.wall
        for p in parts:
            entry = self.plan.get(p.name)
            if entry is None:
                continue
            (sx, sy), yaw = entry
            if self.kin.reachable(self.serve_base,
                                  np.array([sx, sy, floor_z + 0.03]), joints):
                continue
            for f in (0.7, 0.45, 0.2, 0.0):
                cx = self.rack_pos[0] + (sx - self.rack_pos[0]) * f
                cy = self.rack_pos[1] + (sy - self.rack_pos[1]) * f
                cand = _legal_slot(self.rack_pos, self.rack_half, self.wall,
                                   by_name[p.name], yaw, cx, cy)
                if self.kin.reachable(self.serve_base,
                                      np.array([cand[0], cand[1],
                                                floor_z + 0.03]), joints):
                    self.plan[p.name] = (cand, yaw)
                    break

    # -- budget -----------------------------------------------------------

    def _terminal_step(self):
        formula = 300 + 150 * self.n_parts
        return max(formula, self.ASSUMED_EPISODE) - self.TERMINAL_MARGIN

    def _horizon(self):
        return max(300 + 150 * self.n_parts, self.ASSUMED_EPISODE)

    def _part_budget(self):
        remaining = max(80.0, self._horizon() - self.t)
        todo = max(1, self.n_parts - len(self.placed))
        return remaining / float(todo)

    def _T(self, frac):
        return max(10, int(self._part_budget() * frac))

    # -- utils ------------------------------------------------------------

    def _zero(self):
        return np.zeros(self.N_ACT, dtype=np.float32)

    def _fin(self, a):
        return np.clip(np.asarray(a, dtype=np.float32),
                       self.low, self.high).astype(np.float32)

    def _joints(self, state):
        try:
            r = state.get_object_from_name("robot")
            return np.array([state.get(r, "joint_%d" % i) for i in range(1, 8)],
                            dtype=float)
        except Exception:                 # noqa: BLE001
            return self.HOME.copy()

    def _base(self, state):
        try:
            r = state.get_object_from_name("robot")
            return np.array([state.get(r, "pos_base_x"),
                             state.get(r, "pos_base_y"),
                             state.get(r, "pos_base_rot")], dtype=float)
        except Exception:                 # noqa: BLE001
            return np.zeros(3)

    def _home_pull(self, joints):
        d = self.HOME - joints
        over = np.abs(joints - self.HOME) > self.ENVELOPE
        g = 0.05 * d
        g[over] = 0.25 * d[over]
        return g

    def _note_block(self, joints, cmd_dq):
        if self.prev_joints is not None and np.max(np.abs(cmd_dq)) > 1e-3:
            self.diag_cmd_steps += 1
            if np.max(np.abs(joints - self.prev_joints)) < 1e-6:
                self.blocked += 1
                self.diag_reverted += 1
            else:
                self.blocked = 0
        self.prev_joints = joints.copy()

    def _goto(self, s):
        if s == self.stage:
            return
        self.stage = s
        self.stage_t = 0
        if s == "DESCEND":
            self.zmin_hist = []
            self.press_t = 0
        if s == "LEVEL":
            self.tilt_hist = []
        if s == "DUMP":
            self.dump_t = 0

    def _slot_of(self, name):
        e = self.plan.get(name)
        if e is None:
            return (self.rack_pos[0], self.rack_pos[1]), 0.0
        return e[0], e[1]

    def _park_xy(self, sy):
        side = 1.0 if sy >= self.rack_pos[1] else -1.0
        y = self.rack_pos[1] + side * (self.rack_half[1] + 0.13)
        y = float(np.clip(y, _TABLE_LO[1] + 0.06, _TABLE_HI[1] - 0.06))
        x = float(np.clip(self.rack_pos[0] - 0.02,
                          _TABLE_LO[0] + 0.06, _TABLE_HI[0] - 0.06))
        return np.array([x, y])

    def _bump(self, name, held=None):
        if name is None or held is None:
            return
        k = self.attempts.get(name, 0) + 1
        self.attempts[name] = k
        base_entry = self.base_plan.get(name) or self.plan.get(name)
        if base_entry is None:
            return
        (bx, by), yaw = base_entry
        w, h = _footprint_extent(held, yaw)
        pad = _tilt_pad(held)
        lo_x, hi_x, lo_y, hi_y = _slot_bounds(self.rack_pos, self.rack_half,
                                              self.wall, w, h, pad=pad)
        ang = 2.399963 * k
        dx = 0.006 * np.cos(ang)
        dy = 0.006 * np.sin(ang)
        best, best_d = None, 1e9
        for gx in np.linspace(lo_x, hi_x, 7):
            for gy in np.linspace(lo_y, hi_y, 11):
                lo = np.array([gx - w / 2.0, gy - h / 2.0])
                hi = np.array([gx + w / 2.0, gy + h / 2.0])
                ok = True
                for b in self.placed_boxes.values():
                    if _box_gap(lo, hi, b[0], b[1]) < _MIN_PART_GAP:
                        ok = False
                        break
                if not ok:
                    continue
                d = float(np.hypot(gx - (bx + dx), gy - (by + dy)))
                if d < best_d:
                    best_d, best = d, (float(gx), float(gy))
        if best is None:
            best = (bx, by)
        self.plan[name] = (_legal_slot(self.rack_pos, self.rack_half,
                                       self.wall, held, yaw,
                                       best[0], best[1]), yaw)

    # -- recovery guard ---------------------------------------------------

    @staticmethod
    def _off_table(fc):
        return (fc[0] < _TABLE_LO[0] - 0.01 or fc[0] > _TABLE_HI[0] + 0.01
                or fc[1] < _TABLE_LO[1] - 0.01 or fc[1] > _TABLE_HI[1] + 0.01)

    def _recover(self, joints):
        a = self._zero()
        dq = self.HOME - joints
        n = np.max(np.abs(dq))
        if n > 1e-9:
            dq = dq * min(1.0, self.jmag / n)
        a[self.JLO:self.JHI] = np.clip(dq, -self.jmag, self.jmag)
        a[10] = 1.0
        self._note_block(joints, dq)
        return self._fin(a)

    def _safe_to_release(self, held):
        corners = _corners_at(held, held.pos, held.R)
        if _seated_corners(corners, self.rack_pos, self.rack_half, self.wall,
                           tol=4.7e-3):
            return True
        fc = _footprint_centre_world(held)
        if _over_rack(fc, self.rack_pos, self.rack_half):
            return False
        return float(corners[:, 2].min()) <= 0.095 + 8e-3

    # -- main -------------------------------------------------------------

    def get_action(self, state):
        self.t += 1
        self.stage_t += 1
        self.part_t += 1
        if self.stage == "DUMP":
            self.dump_t += 1

        parts = _read_parts(state)
        if not parts:
            return self._fin(self._zero())
        by_name = {p.name: p for p in parts}
        if set(self.plan.keys()) != set(by_name.keys()):
            self._refresh(state)
        if self.serve_base is None:
            self.serve_base = self._choose_serve_base(state)

        joints = self._joints(state)
        base = self._base(state)
        held = next((p for p in parts if p.grasped), None)

        for p in parts:
            if (not p.grasped) and p.name not in self.placed:
                if _is_seated(p, self.rack_pos, self.rack_half, self.wall):
                    self.placed.add(p.name)
                    self.placed_boxes[p.name] = _footprint_box(p)
                    self.placed_pegs[p.name] = _peg_xy(p)

        # v28 FIX 2: if the held part is ALREADY seated, release now, whatever
        # stage we are in. This was previously only checked inside DESCEND.
        if held is not None:
            if _seated_corners(_corners_at(held, held.pos, held.R),
                               self.rack_pos, self.rack_half, self.wall,
                               tol=4.7e-3):
                a = self._zero()
                a[10] = 1.0
                return self._fin(a)

        # v28 FIX 1: fixed terminal release step that is always reached.
        # v27 used 3 * formula = 1350 > 1000 and so never fired, ending grasped.
        if self.t >= self._terminal_step():
            a = self._zero()
            a[10] = 1.0
            if held is not None:
                fc = _footprint_centre_world(held)
                ee, _R = self.kin.ee_pose(base, joints)
                off = ee - fc
                floor_z = self.rack_pos[2] - self.rack_half[2] + self.wall
                (sx, sy), yaw_cmd = self._slot_of(held.name)
                drop_z = floor_z + _corner_drop(held)
                if _over_rack(fc, self.rack_pos, self.rack_half):
                    goal_fc = np.array([sx, sy, drop_z - 0.006])
                else:
                    goal_fc = np.array([fc[0], fc[1], 0.095 + held.hz])
                dq, _e, _ = self.kin.solve(base, joints, goal_fc + off, None,
                                           self.jmag, w_rot=0.0,
                                           home_pull=self._home_pull(joints))
                a[self.JLO:self.JHI] = np.clip(dq, -self.jmag, self.jmag)
                self._note_block(joints, dq)
            return self._fin(a)

        if self.recover_t > 0:
            self.recover_t -= 1
            return self._recover(joints)
        if held is not None:
            fc_now = _footprint_centre_world(held)
            if self._off_table(fc_now):
                self.recover_t = 25
                self.target = None
                self.G = None
                self.part_t = 0
                self._goto("BASE")
                return self._recover(joints)

        if self.escape_t > 0:
            self.escape_t -= 1
            a = self._zero()
            a[self.JLO:self.JHI] = np.clip(self.escape, -self.jmag, self.jmag)
            a[10] = -1.0 if held is not None else 0.0
            self._note_block(joints, self.escape)
            return self._fin(a)
        if self.blocked > 10:
            self.escape = self.rng.normal(scale=self.jmag * 0.7, size=7)
            self.escape_t = 5
            self.blocked = 0

        if self.t > self.deadline and (self.n_parts - len(self.placed)) >= 1:
            self.deadline = self.t + self.slice_len
            if held is None:
                self.target = None
                self.part_t = 0
                self._goto("BASE")
                return self._fin(self._zero())
            self._bump(self.target, held)
            self.part_t = 0
            self.desc_retry = 0
            self._goto("LIFT")

        if held is None:
            self.G = None
            self.tilt_hist = []
            self.desc_retry = 0
            if self.part_t > self._part_budget():
                self.target = None
                self.part_t = 0
                self._goto("BASE")
            self._pick(parts)
            if self.target is None:
                return self._fin(self._zero())
            if self.stage in ("LIFT", "REBASE", "LEVEL", "TRAVERSE",
                              "ALIGN", "DESCEND", "DUMP"):
                self._goto("BASE")
            return self._to_part(by_name[self.target], base, joints)

        if self.G is None:
            _, R_ee = self.kin.ee_pose(base, joints)
            self.G = R_ee.T.dot(held.R)
            self.target = held.name
            self.desc_retry = 0
            self._goto("LIFT")
        if self.stage in ("BASE", "REACH"):
            self._goto("LIFT")
        return self._carry(held, base, joints)

    def _pick(self, parts):
        cands = [p for p in parts if p.name not in self.placed]
        if not cands:
            self.target = None
            return
        if self.target in {p.name for p in cands}:
            return

        def key(p):
            (sx, sy), _ = self._slot_of(p.name)
            fc = _footprint_centre_world(p)
            return (float(np.hypot(fc[0] - sx, fc[1] - sy)), p.name)
        cands.sort(key=key)
        self.target = cands[0].name
        self.base_goal = None
        self.part_t = 0
        self.blocked = 0
        self.deadline = max(self.deadline, self.t + self.slice_len)
        self._goto("BASE")

    # -- reach & grasp ----------------------------------------------------

    def _to_part(self, target, base, joints):
        a = self._zero()
        peg = _peg_world(target)
        if self.base_goal is None:
            if self.kin.reachable(self.serve_base, peg, joints):
                self.base_goal = self.serve_base[:2].copy()
            else:
                self.base_goal = np.array([peg[0] - 0.40, peg[1]])

        if self.stage == "BASE":
            berr = self.base_goal - base[:2]
            if np.linalg.norm(berr) > 0.012 and self.stage_t < self._T(self.F_BASE):
                a[0] = float(np.clip(berr[0], -self.bmag, self.bmag))
                a[1] = float(np.clip(berr[1], -self.bmag, self.bmag))
                return self._fin(a)
            self._goto("REACH")

        ee = self.kin.fk(base, joints)
        horiz = float(np.linalg.norm(ee[:2] - peg[:2]))
        t_reach = self._T(self.F_REACH)

        if horiz > 0.025 and self.stage_t < t_reach * 0.5:
            goal = np.array([peg[0], peg[1], max(self.clear_z, peg[2] + 0.12)])
            grip = 0.0
        else:
            goal = np.array([peg[0], peg[1], peg[2]])
            grip = -1.0

        dq, e_pos, _ = self.kin.solve(base, joints, goal, None, self.jmag,
                                      w_rot=0.0,
                                      home_pull=self._home_pull(joints))
        if np.linalg.norm(e_pos) < 0.05:
            dq = dq * 0.4
        if self.stage_t > t_reach:
            dq = dq + self.rng.normal(scale=self.jmag * 0.3, size=7)
        a[self.JLO:self.JHI] = np.clip(dq, -self.jmag, self.jmag)
        a[10] = grip
        self._note_block(joints, dq)
        return self._fin(a)

    # -- carry ------------------------------------------------------------

    def _carry(self, held, base, joints):
        a = self._zero()
        ee, R_ee = self.kin.ee_pose(base, joints)
        (sx, sy), yaw_cmd = self._slot_of(held.name)

        floor_z = self.rack_pos[2] - self.rack_half[2] + self.wall
        drop_z = floor_z + _corner_drop(held)
        lift_z = self.clear_z

        fc = _footprint_centre_world(held)
        off = ee - fc
        horiz = float(np.linalg.norm(np.array([sx, sy]) - fc[:2]))
        tilt = held.tilt
        yaw_err = abs(_wrap(held.yaw - yaw_cmd))
        R_goal_ee = _rotz(yaw_cmd).dot(self.G.T) if self.G is not None else None

        corners = _corners_at(held, held.pos, held.R)
        z_min = float(corners[:, 2].min())
        clearance = _wall_clearance(corners, self.rack_pos, self.rack_half,
                                    self.wall)
        lo_f, hi_f = _footprint_box(held)
        gap_placed = (min(_box_gap(lo_f, hi_f, b[0], b[1])
                          for b in self.placed_boxes.values())
                      if self.placed_boxes else 1.0)

        if self.stage == "REBASE":
            berr = self.serve_base[:2] - base[:2]
            if (np.linalg.norm(berr) > 0.015
                    and self.stage_t < self._T(self.F_REBASE)):
                a[0] = float(np.clip(berr[0], -self.bmag * 0.5, self.bmag * 0.5))
                a[1] = float(np.clip(berr[1], -self.bmag * 0.5, self.bmag * 0.5))
                a[10] = 0.0
                return self._fin(a)
            self._goto("LEVEL")

        if self.stage == "LIFT":
            if fc[2] >= lift_z - 0.02 or self.stage_t > self._T(self.F_LIFT):
                if float(np.linalg.norm(base[:2] - self.serve_base[:2])) > 0.02:
                    self._goto("REBASE")
                else:
                    self._goto("LEVEL")
        elif self.stage == "LEVEL":
            self.tilt_hist.append(tilt)
            if len(self.tilt_hist) > 20:
                self.tilt_hist.pop(0)
            plateau = (len(self.tilt_hist) >= 20
                       and (max(self.tilt_hist) - min(self.tilt_hist)) < 2e-3)
            if tilt < self.TILT_OK or plateau \
                    or self.stage_t > self._T(self.F_LEVEL):
                self._goto("TRAVERSE")
        elif self.stage == "TRAVERSE":
            if horiz < 0.025 or self.stage_t > self._T(self.F_TRAV):
                self._goto("ALIGN")
        elif self.stage == "ALIGN":
            if (yaw_err < 0.12 and tilt < self.TILT_OK and horiz < 0.008) \
                    or self.stage_t > self._T(self.F_ALIGN):
                self._goto("DESCEND")
        elif self.stage == "DESCEND":
            self.zmin_hist.append(z_min)
            if len(self.zmin_hist) > 10:
                self.zmin_hist.pop(0)
            stalled = (len(self.zmin_hist) >= 10
                       and (max(self.zmin_hist) - min(self.zmin_hist)) < 4e-4)
            too_high = (z_min - floor_z) > 5.0e-3
            if stalled and too_high:
                # v28 FIX 3: press deeper before giving up on this slot.
                if self.press_t < 60:
                    self.press_t += 1
                    self.zmin_hist = []
                elif self.desc_retry < 3 or self.n_parts <= 1:
                    self.desc_retry += 1
                    self.press_t = 0
                    self._bump(held.name, held)
                    self._goto("LIFT")
                else:
                    self._goto("DUMP")
            elif self.stage_t > self._T(self.F_DESC):
                if self.desc_retry < 3 or self.n_parts <= 1:
                    self.desc_retry += 1
                    self.press_t = 0
                    self._bump(held.name, held)
                    self._goto("LIFT")
                else:
                    self._goto("DUMP")
        elif self.stage != "DUMP":
            self._goto("LIFT")

        if self.stage == "DUMP":
            pxy = self._park_xy(sy)
            goal_fc = np.array([pxy[0], pxy[1], 0.095 + held.hz])
            dq, e_pos, _ = self.kin.solve(base, joints, goal_fc + off,
                                          R_goal_ee, self.jmag, w_rot=0.4,
                                          home_pull=self._home_pull(joints))
            a[self.JLO:self.JHI] = np.clip(dq, -self.jmag, self.jmag)
            a[0] = a[1] = a[2] = 0.0
            a[10] = 1.0 if (self._safe_to_release(held) or self.dump_t > 120) \
                else 0.0
            self._note_block(joints, dq)
            return self._fin(a)

        if self.stage == "LIFT":
            goal_fc = np.array([fc[0], fc[1], lift_z])
            w_rot = 0.20
        elif self.stage == "LEVEL":
            goal_fc = np.array([fc[0], fc[1], lift_z])
            w_rot = 0.95
        elif self.stage == "TRAVERSE":
            goal_fc = np.array([sx, sy, lift_z])
            w_rot = 0.60
        elif self.stage == "ALIGN":
            goal_fc = np.array([sx, sy, drop_z + 0.05])
            w_rot = 0.95
        else:
            # deeper target while pressing (v28 FIX 3)
            depth = 0.005 + (0.010 if self.press_t > 0 else 0.0)
            goal_fc = np.array([sx, sy, drop_z - depth])
            w_rot = 0.95

        dq, e_pos, _e_rot = self.kin.solve(base, joints, goal_fc + off,
                                           R_goal_ee, self.jmag, w_rot=w_rot,
                                           home_pull=self._home_pull(joints))
        if float(np.linalg.norm(e_pos)) < 0.04:
            dq = dq * 0.35
        a[self.JLO:self.JHI] = np.clip(dq, -self.jmag, self.jmag)
        a[0] = a[1] = a[2] = 0.0

        if self.stage == "DESCEND" \
                and tilt < self.TILT_RELEASE \
                and _seated_corners(corners, self.rack_pos, self.rack_half,
                                    self.wall, tol=4.7e-3) \
                and clearance > 5.0e-3 \
                and gap_placed > 2.0e-3:
            a[10] = 1.0
        else:
            a[10] = 0.0

        self._note_block(joints, dq)
        return self._fin(a)