"""Small scalar collision helpers; rectangle = (cx, cy, hx, hy, cos, sin).

lower_left_rect rotates local width/height axes about the supplied lower-left
corner. Contacts count as collisions (subject to a small numeric tolerance).
"""
import math

_EPS = 1e-10


def center_rect(cx, cy, width, height, theta=0.0):
    return (cx, cy, width * 0.5, height * 0.5,
            math.cos(theta), math.sin(theta))


def lower_left_rect(x, y, width, height, theta=0.0):
    c, s = math.cos(theta), math.sin(theta)
    hx, hy = width * 0.5, height * 0.5
    return (x + c * hx - s * hy, y + s * hx + c * hy, hx, hy, c, s)


def inflate_rect(rect, margin):
    x, y, hx, hy, c, s = rect
    return (x, y, max(0.0, hx + margin), max(0.0, hy + margin), c, s)


def rect_corners(rect):
    x, y, hx, hy, c, s = rect
    ux, uy, vx, vy = c * hx, s * hx, -s * hy, c * hy
    return ((x - ux - vx, y - uy - vy),
            (x + ux - vx, y + uy - vy),
            (x + ux + vx, y + uy + vy),
            (x - ux + vx, y - uy + vy))


def point_rect_distance_sq(x, y, rect):
    cx, cy, hx, hy, c, s = rect
    dx, dy = x - cx, y - cy
    qx = max(0.0, abs(c * dx + s * dy) - hx)
    qy = max(0.0, abs(-s * dx + c * dy) - hy)
    return qx * qx + qy * qy


def circle_rect_collision(x, y, radius, rect, margin=0.0):
    radius = max(0.0, radius + margin)
    return point_rect_distance_sq(x, y, rect) <= radius * radius + _EPS


def rect_rect_collision(a, b, margin=0.0):
    """Separating-axis test. Margin expands a by margin on its local axes."""
    ax, ay, ahx, ahy, ac, ass = a
    bx, by, bhx, bhy, bc, bs = b
    ahx, ahy = max(0.0, ahx + margin), max(0.0, ahy + margin)
    dx, dy = bx - ax, by - ay
    # Absolute pairwise projections between each rectangle's basis vectors.
    cc = abs(ac * bc + ass * bs)
    cs = abs(ac * bs - ass * bc)
    if abs(dx * ac + dy * ass) > ahx + bhx * cc + bhy * cs + _EPS:
        return False
    if abs(-dx * ass + dy * ac) > ahy + bhx * cs + bhy * cc + _EPS:
        return False
    if abs(dx * bc + dy * bs) > bhx + ahx * cc + ahy * cs + _EPS:
        return False
    if abs(-dx * bs + dy * bc) > bhy + ahx * cs + ahy * cc + _EPS:
        return False
    return True


def _point_segment_distance_sq(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= _EPS * _EPS:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return (px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2


def segment_rect_collision(ax, ay, bx, by, rect, thickness=0.0):
    """True if a segment/capsule intersects rect; thickness is capsule radius."""
    cx, cy, hx, hy, c, s = rect
    adx, ady, bdx, bdy = ax - cx, ay - cy, bx - cx, by - cy
    ax, ay = c * adx + s * ady, -s * adx + c * ady
    bx, by = c * bdx + s * bdy, -s * bdx + c * bdy
    dx, dy = bx - ax, by - ay
    lo, hi = 0.0, 1.0
    intersects = True
    for start, delta, half in ((ax, dx, hx), (ay, dy, hy)):
        if abs(delta) <= _EPS:
            if abs(start) > half + _EPS:
                intersects = False
                break
        else:
            t0, t1 = (-half - start) / delta, (half - start) / delta
            if t0 > t1:
                t0, t1 = t1, t0
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi + _EPS:
                intersects = False
                break
    if intersects:
        return True
    if thickness <= 0.0:
        return False
    rr = thickness * thickness + _EPS
    # Disjoint segment/box minimum distance occurs at a segment endpoint
    # or a box corner. These tests include rounded capsule end caps.
    for px, py in ((ax, ay), (bx, by)):
        qx, qy = max(0.0, abs(px) - hx), max(0.0, abs(py) - hy)
        if qx * qx + qy * qy <= rr:
            return True
    for px, py in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        if _point_segment_distance_sq(px, py, ax, ay, bx, by) <= rr:
            return True
    return False
