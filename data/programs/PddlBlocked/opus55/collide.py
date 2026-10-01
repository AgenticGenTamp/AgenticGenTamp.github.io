"""Approximate arm collision checking (capsules vs boxes)."""
import numpy as np
from kinpy import fk

TABLE_TOP = 0.73
# (segment index pair, radius vs table, radius vs objects)
R_UPPER = 0.075
R_FORE = 0.06
R_GRIP_T = 0.06   # gripper vs table (empirically ~0.07 below tool axis at contact)
R_GRIP_O = 0.03


class Obstacles:
    def __init__(self, table, blocks, walls):
        """table: (cx,cy,hx,hy); blocks/walls: list of (cx,cy,cz,hx,hy,hz,yaw)."""
        self.table = table
        self.boxes = list(blocks) + list(walls)
        self.nblocks = len(blocks)


def link_points(b, q):
    p, (c0, c1, c2), axes, origins, elbow = fk(b, q)
    sh = np.array(origins[1])     # shoulder lift
    el = np.array(elbow)
    wr = np.array(origins[5])     # wrist flex
    tool = np.array(p)
    return sh, el, wr, tool, np.array(c0)


def _seg_samples(a, b, n):
    return [a + (b - a) * (i / (n - 1)) for i in range(n)]


def point_box_dist(p, box):
    cx, cy, cz, hx, hy, hz, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    ex = max(abs(lx) - hx, 0.0); ey = max(abs(ly) - hy, 0.0); ez = max(abs(dz) - hz, 0.0)
    return (ex * ex + ey * ey + ez * ez) ** 0.5


def over_table(p, table, r):
    cx, cy, hx, hy = table
    return abs(p[0] - cx) < hx + r and abs(p[1] - cy) < hy + r


def table_clear(p, table, r):
    """distance-like clearance of point p (with radius r) above table top; positive=clear."""
    cx, cy, hx, hy = table
    ex = max(abs(p[0] - cx) - hx, 0.0); ey = max(abs(p[1] - cy) - hy, 0.0)
    ez = max(p[2] - TABLE_TOP, 0.0)
    return (ex * ex + ey * ey + ez * ez) ** 0.5 - r


def collisions(b, q, obs, skip_blocks=(), held=None, verbose=False):
    """Return list of collision descriptions (empty if clear)."""
    sh, el, wr, tool, ax = link_points(b, q)
    grip_end = tool - 0.02 * ax
    segs = [("upper", sh, el, R_UPPER, R_UPPER, 5), ("fore", el, wr, R_FORE, R_FORE, 5),
            ("grip", wr, grip_end, R_GRIP_T, R_GRIP_O, 4)]
    out = []
    for name, a, c, rt, ro, n in segs:
        for p in _seg_samples(a, c, n):
            if table_clear(p, obs.table, rt) < 0:
                out.append((name, 'table'))
                break
            for i, box in enumerate(obs.boxes):
                if i in skip_blocks:
                    continue
                if point_box_dist(p, box) < ro:
                    out.append((name, 'box%d' % i))
                    break
    return out


# ---------------- global obstacle context used by the planner ----------------
CTX = {'boxes': {}, 'skip': set(), 'on': True}
R_LINK = {'upper': 0.085, 'fore': 0.07, 'grip': 0.035}


def set_obstacles(named_boxes):
    CTX['boxes'] = dict(named_boxes)


def pen_boxes(g0x, g0y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)

    def bx(u, v, hu, hv):
        return (g0x + c * u - s * v, g0y + s * u + c * v, 0.8, hu, hv, 0.07, yaw)
    return {'penL': bx(0.08, 0.15, 0.085, 0.012), 'penR': bx(0.08, -0.15, 0.085, 0.012),
            'penB': bx(0.16, 0.0, 0.012, 0.165)}


def arm_hits(b, q):
    if not CTX['on'] or not CTX['boxes']:
        return None
    sh, el, wr, tool, ax = link_points(b, q)
    segs = (('upper', sh, el, 5), ('fore', el, wr, 5), ('grip', wr, tool - 0.03 * ax, 4))
    for name, a, c, n in segs:
        r = R_LINK[name]
        for p in _seg_samples(a, c, n):
            for bn, box in CTX['boxes'].items():
                if bn in CTX['skip']:
                    continue
                if abs(p[0] - box[0]) > 0.4 or abs(p[1] - box[1]) > 0.4:
                    continue
                if point_box_dist(p, box) < r:
                    return (name, bn)
    return None
