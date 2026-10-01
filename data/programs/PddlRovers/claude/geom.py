"""Geometry model + grid planner for the rovers env (stdlib+numpy only)."""
import numpy as np
from collections import deque

ARENA_HALF = 2.45      # inner wall faces
ROVER_RADIUS = 0.177   # disc radius vs short obstacles (walls, mounds)
ROVER_RADIUS_TALL = 0.24  # conservative footprint vs tall obstacles (pillars, lander)
IMG_RANGE = 1.70        # safe (hard limit 2.0, heading modulates +-0.08)
COMM_RANGE = 3.88       # safe (hard limit 4.0)
SAMPLE_RANGE = 0.25
STEP = 0.2
GRID_RES = 0.05

# middle wall box (cx, cy, hx, hy) -- calibrate empirically
MID_WALL = (0.0, 0.0, 0.05, 2.5)   # splits arena: rover0 east, rover1 west
LANDER_HALF = 0.50
SIGHT_PAD = 0.20        # pillar pad for ray occlusion (footprint 0.05)
WALL_BLOCKS_SIGHT = False          # extra inflation of obstacles for line-of-sight
MOUNDS_BLOCK_SIGHT = False   # 0.1m mounds never block
WALL_BAND = (-1.95, -0.88)    # crossing x=0 in this y range blocks low (lander) rays


def obstacles_from_state(state):
    """Return list of dicts for each obstacle box in the state."""
    out = []
    for name in state.get_object_names():
        if not name.startswith("obstacle"):
            continue
        o = state.get_object_from_name(name)
        out.append(dict(
            name=name,
            cx=float(state.get(o, "x")), cy=float(state.get(o, "y")),
            hx=float(state.get(o, "half_x")), hy=float(state.get(o, "half_y")),
            hz=float(state.get(o, "half_z")),
        ))
    return out


class Model:
    """Collision + visibility model built from an ObjectCentricState."""

    def __init__(self, state, mounds_block_sight=MOUNDS_BLOCK_SIGHT,
                 wall_blocks_sight=WALL_BLOCKS_SIGHT, sight_pad=SIGHT_PAD):
        obs = obstacles_from_state(state)
        o = state.get_object_from_name("lander")
        lx, ly = float(state.get(o, "x")), float(state.get(o, "y"))
        self.lander = np.array([lx, ly])
        # --- motion blockers: (cx, cy, hx, hy, inflate) ---
        mb = []
        sb = []
        for ob in obs:
            tall = ob["hz"] > 0.1
            r = ROVER_RADIUS_TALL if tall else ROVER_RADIUS
            mb.append((ob["cx"], ob["cy"], ob["hx"], ob["hy"], r))
            if tall or mounds_block_sight:
                sb.append((ob["cx"], ob["cy"], ob["hx"] + sight_pad, ob["hy"] + sight_pad))
        mb.append((lx, ly, LANDER_HALF, LANDER_HALF, ROVER_RADIUS_TALL))
        mb.append(MID_WALL + (ROVER_RADIUS,))
        self.motion_boxes = np.array(mb, dtype=float)
        self.sight_boxes = np.array(sb, dtype=float) if sb else np.zeros((0, 4))

    # ---------- collision ----------
    def free_points(self, pts, clearance=0.0):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        ok = np.all(np.abs(pts) <= ARENA_HALF - ROVER_RADIUS - clearance + 1e-9, axis=1)
        c = self.motion_boxes[:, :2]
        h = self.motion_boxes[:, 2:4]
        r = self.motion_boxes[:, 4]
        d = np.abs(pts[:, None, :] - c[None, :, :])
        q = np.maximum(d - h[None, :, :], 0.0)
        dist = np.hypot(q[:, :, 0], q[:, :, 1])
        ok &= np.all(dist > r[None, :] + clearance - 1e-9, axis=1)
        return ok

    # ---------- visibility ----------
    def _band_blocked(self, pts, q):
        """True where segment pts[i]->q crosses x=0 inside the occluding band."""
        pts = np.atleast_2d(np.asarray(pts, float))
        px, py = pts[:, 0], pts[:, 1]
        qx, qy = float(q[0]), float(q[1])
        cross = (px * qx) < 0.0
        out = np.zeros(len(pts), dtype=bool)
        if not np.any(cross):
            return out
        t = px[cross] / (px[cross] - qx)
        yc = py[cross] + t * (qy - py[cross])
        out[cross] = (yc > WALL_BAND[0]) & (yc < WALL_BAND[1])
        return out

    def can_send(self, p):
        p = np.asarray(p, float)[:2]
        if np.hypot(*(p - self.lander)) > COMM_RANGE:
            return False
        return not bool(self._band_blocked(p[None, :], self.lander)[0])

    def can_send_many(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        d = np.hypot(pts[:, 0] - self.lander[0], pts[:, 1] - self.lander[1])
        return (d <= COMM_RANGE) & ~self._band_blocked(pts, self.lander)

    def visible(self, p, q):
        p = np.asarray(p, float)[:2]
        q = np.asarray(q, float)[:2]
        if _seg_hits_boxes(p, q, self.sight_boxes):
            return False
        return not bool(self._band_blocked(p[None, :], q)[0])

    def visible_many(self, pts, q):
        pts = np.atleast_2d(np.asarray(pts, float))
        q = np.asarray(q, float)[:2]
        return (~_seg_hits_boxes_many(pts, q, self.sight_boxes)) & ~self._band_blocked(pts, q)

    # ---------- grid ----------
    def build_grid(self, res=GRID_RES, clearance=0.0):
        n = int(round(2 * 2.5 / res))
        xs = -2.5 + (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        free = self.free_points(pts, clearance=clearance).reshape(n, n)
        self.res = res
        self.n = n
        self.xs = xs
        self.free = free
        return free


def _seg_hits_boxes(p, q, boxes):
    return bool(_seg_hits_boxes_many(p[None, :], q, boxes)[0])


def _seg_hits_boxes_many(pts, q, boxes):
    """Vectorized segment(pts[i] -> q) vs AABB intersection using slab method.
    Boxes that contain q are ignored. Returns bool array (N,)."""
    if len(boxes) == 0:
        return np.zeros(len(pts), dtype=bool)
    c = boxes[:, :2]
    h = boxes[:, 2:]
    lo = c - h
    hi = c + h
    # ignore boxes containing q
    contains_q = np.all((q >= lo) & (q <= hi), axis=1)
    d = q[None, :] - pts                      # (N,2)
    hit = np.zeros(len(pts), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(len(boxes)):
            if contains_q[k]:
                continue
            t0 = np.zeros(len(pts))
            t1 = np.ones(len(pts))
            alive = np.ones(len(pts), dtype=bool)
            for ax in range(2):
                dk = d[:, ax]
                p0 = pts[:, ax]
                par = np.abs(dk) < 1e-12
                # parallel: must be inside slab
                out_par = par & ((p0 < lo[k, ax]) | (p0 > hi[k, ax]))
                alive &= ~out_par
                inv = np.where(par, 1.0, 1.0 / np.where(par, 1.0, dk))
                ta = (lo[k, ax] - p0) * inv
                tb = (hi[k, ax] - p0) * inv
                tlo = np.where(par, -np.inf, np.minimum(ta, tb))
                thi = np.where(par, np.inf, np.maximum(ta, tb))
                t0 = np.maximum(t0, tlo)
                t1 = np.minimum(t1, thi)
            hit |= alive & (t0 <= t1)
    return hit
