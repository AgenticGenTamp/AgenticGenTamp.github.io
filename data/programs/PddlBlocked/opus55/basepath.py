"""Base path around an axis-aligned table rectangle."""
import numpy as np


def seg_hits_rect(a, b, rect):
    """Does segment a->b intersect open rect (xmin,xmax,ymin,ymax)? Liang-Barsky."""
    x0, y0 = a; x1, y1 = b
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - rect[0]), (dx, rect[1] - x0), (-dy, y0 - rect[2]), (dy, rect[3] - y0)):
        if abs(p) < 1e-12:
            if q <= 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    return t1 - t0 > 1e-9


def base_path(a, b, table, clear=0.60):
    """Shortest 2D path from a to b avoiding table expanded by clear. Returns list of xy points (excluding a)."""
    cx, cy, hx, hy = table
    rect = (cx - hx - clear + 1e-3, cx + hx + clear - 1e-3, cy - hy - clear + 1e-3, cy + hy + clear - 1e-3)
    m = clear + 0.02
    corners = [np.array([cx + sx * (hx + m), cy + sy * (hy + m)]) for sx in (-1, 1) for sy in (-1, 1)]
    corners = [c for c in corners if abs(c[0]) < 4.97]
    nodes = [np.asarray(a[:2], float), np.asarray(b[:2], float)] + corners
    n = len(nodes)
    INF = 1e9
    dist = [INF] * n; prev = [-1] * n; dist[0] = 0.0; done = [False] * n
    for _ in range(n):
        u = min((i for i in range(n) if not done[i]), key=lambda i: dist[i])
        if dist[u] >= INF:
            break
        done[u] = True
        for v in range(n):
            if done[v] or v == u:
                continue
            if seg_hits_rect(nodes[u], nodes[v], rect):
                continue
            w = np.linalg.norm(nodes[u] - nodes[v])
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w; prev[v] = u
    path = []
    v = 1
    if dist[1] >= INF:
        return [nodes[1]]
    while v != 0:
        path.append(nodes[v]); v = prev[v]
    return path[::-1]


def push_out(p, table, clear):
    cx, cy, hx, hy = table
    dx, dy = p[0] - cx, p[1] - cy
    ox = hx + clear - abs(dx); oy = hy + clear - abs(dy)
    if ox <= 0 or oy <= 0:
        return np.asarray(p[:2], float), False
    q = np.array(p[:2], float)
    if ox < oy:
        q[0] = cx + np.sign(dx or 1) * (hx + clear + 0.01)
    else:
        q[1] = cy + np.sign(dy or 1) * (hy + clear + 0.01)
    return q, True


def base_route(a, b, table, clear=0.60):
    """List of (x,y,yaw) base poses from a to b (excluding a)."""
    a_out, a_in = push_out(a, table, clear)
    b_out, b_in = push_out(b, table, clear)
    pts = []
    if a_in:
        pts.append(a_out)
    pts += base_path(a_out, b_out, table, clear)
    poses = [np.array([p[0], p[1], b[2]]) for p in pts]
    if b_in:
        poses.append(np.array(b, float))
    else:
        poses[-1] = np.array(b, float)
    # drop near-duplicate points
    out = []
    prev = np.asarray(a, float)
    for p in poses:
        if np.linalg.norm(p[:2] - prev[:2]) > 1e-3 or abs(p[2] - prev[2]) > 1e-3:
            out.append(p)
        prev = p
    return out
