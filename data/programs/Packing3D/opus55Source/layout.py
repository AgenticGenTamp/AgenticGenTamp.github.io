"""2D layout optimizer: place part footprints inside the rack interior."""
from __future__ import annotations

import math

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box


def local_footprint(kind, params):
    """Footprint polygon in the part's local (origin) frame."""
    if kind == "cuboid":
        hx, hy = params
        return Polygon([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])
    ttype, a, b = params
    if ttype == "equilateral":
        h = math.sqrt(3) / 2 * a
        return Polygon([(-a / 2, -h / 3), (a / 2, -h / 3), (0, 2 * h / 3)])
    return Polygon([(0, 0), (a, 0), (0, b)])


def place_poly(poly, x, y, yaw):
    p = affinity.rotate(poly, yaw, origin=(0, 0), use_radians=True)
    return affinity.translate(p, x, y)


def clearance(polys, inner):
    """Min clearance: pairwise distances and distance to inner boundary.
    Negative if overlapping / outside."""
    best = np.inf
    for i, p in enumerate(polys):
        if not inner.contains(p):
            out = p.difference(inner).area
            best = min(best, -1e3 * out - 1e-3)
        else:
            best = min(best, inner.exterior.distance(p))
        for q in polys[i + 1:]:
            if p.intersects(q):
                best = min(best, -1e3 * p.intersection(q).area - 1e-3)
            else:
                best = min(best, p.distance(q))
    return best


def optimize_layout(footprints, inner_bounds, rng=None, iters=1500,
                    restarts=12, time_budget=8.0):
    """footprints: list of local polygons. inner_bounds: (xmin,ymin,xmax,ymax).
    Returns (list of (x,y,yaw), clearance)."""
    import time
    t0 = time.time()
    if rng is None:
        rng = np.random.default_rng(0)
    xmin, ymin, xmax, ymax = inner_bounds
    inner = box(xmin, ymin, xmax, ymax)
    n = len(footprints)
    # centroid offsets so that we can parametrize by centroid
    cents = [np.array(fp.centroid.coords[0]) for fp in footprints]

    def build(params):
        polys = []
        for k in range(n):
            cx, cy, yaw = params[k]
            c, s = math.cos(yaw), math.sin(yaw)
            off = cents[k]
            ox = cx - (c * off[0] - s * off[1])
            oy = cy - (s * off[0] + c * off[1])
            polys.append(place_poly(footprints[k], ox, oy, yaw))
        return polys

    def score(params):
        return clearance(build(params), inner)

    best_p, best_s = None, -np.inf
    yaw_choices = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    for r in range(restarts):
        if time.time() - t0 > time_budget and best_p is not None:
            break
        p = np.zeros((n, 3))
        p[:, 0] = rng.uniform(xmin, xmax, n)
        p[:, 1] = rng.uniform(ymin, ymax, n)
        p[:, 2] = rng.choice(yaw_choices, n)
        s = score(p)
        step = np.array([0.03, 0.03, 0.5])
        for it in range(iters):
            q = p.copy()
            k = rng.integers(n)
            q[k] += rng.normal(0, 1, 3) * step
            sq = score(q)
            if sq >= s:
                p, s = q, sq
            if it % 300 == 299:
                step *= 0.6
        if s > best_s:
            best_p, best_s = p, s
    # convert centroid params to origin params
    out = []
    for k in range(n):
        cx, cy, yaw = best_p[k]
        c, s_ = math.cos(yaw), math.sin(yaw)
        off = cents[k]
        out.append((cx - (c * off[0] - s_ * off[1]),
                    cy - (s_ * off[0] + c * off[1]), float(yaw)))
    return out, best_s


def refine_layout(footprints, inner_bounds, init_centroid_params, iters=3000,
                  rng=None, step0=(0.01, 0.01, 0.1)):
    """Local hill-climb starting from given centroid params (cx, cy, yaw)."""
    if rng is None:
        rng = np.random.default_rng(0)
    xmin, ymin, xmax, ymax = inner_bounds
    inner = box(xmin, ymin, xmax, ymax)
    n = len(footprints)
    cents = [np.array(fp.centroid.coords[0]) for fp in footprints]

    def to_origin(params):
        out = []
        for k in range(n):
            cx, cy, yaw = params[k]
            c, s = math.cos(yaw), math.sin(yaw)
            off = cents[k]
            out.append((cx - (c * off[0] - s * off[1]),
                        cy - (s * off[0] + c * off[1]), float(yaw)))
        return out

    def score(params):
        polys = [place_poly(footprints[k], *o) for k, o in enumerate(to_origin(params))]
        return clearance(polys, inner)

    p = np.array(init_centroid_params, dtype=float)
    s = score(p)
    step = np.array(step0)
    for it in range(iters):
        q = p.copy()
        if rng.random() < 0.3:
            q += rng.normal(0, 1, q.shape) * step * 0.5
        else:
            k = rng.integers(n)
            q[k] += rng.normal(0, 1, 3) * step
        sq = score(q)
        if sq >= s:
            p, s = q, sq
        if it % 500 == 499:
            step *= 0.6
    return to_origin(p), s, p


RACK_INNER = (0.21, -0.14, 0.39, 0.14)


def compute_layout(parts, inner=RACK_INNER, rng=None):
    """parts: list of (kind, params). Returns list of origin (x, y, yaw), clr."""
    fps = [local_footprint(k, p) for k, p in parts]
    xmin, ymin, xmax, ymax = inner
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    W, H = xmax - xmin, ymax - ymin
    cub = [i for i, (k, _) in enumerate(parts) if k == "cuboid"]
    tri = [i for i, (k, _) in enumerate(parts) if k != "cuboid"]
    standard = (len(cub) <= 2 and len(tri) <= 2 and abs(W - 0.18) < 1e-6
                and abs(H - 0.28) < 1e-6
                and all(abs(p[0] - 0.05) < 1e-6 and abs(p[1] - 0.05) < 1e-6
                        for k, p in parts if k == "cuboid")
                and all(abs(p[1] - 0.1) < 1e-6 and abs(p[2] - 0.1) < 1e-6
                        for k, p in parts if k != "cuboid"))
    if standard:
        h = math.sqrt(3) / 2 * 0.1
        sq_slots = [(cx - 0.04, cy - 0.09, 0.0), (cx + 0.04, cy + 0.09, 0.0)]
        eq_slots = [(xmax - h / 3, cy - 0.01, math.pi / 2),
                    (xmin + h / 3, cy + 0.01, -math.pi / 2)]
        rt_slots = [(xmax - 0.1 / 3, cy + 0.04 - 0.1 / 3, math.pi),
                    (xmin + 0.1 / 3, cy - 0.04 + 0.1 / 3, 0.0)]
        init = [None] * len(parts)
        for j, i in enumerate(cub):
            init[i] = sq_slots[j]
        for j, i in enumerate(tri):
            ttype = parts[i][1][0]
            init[i] = (eq_slots if ttype == "equilateral" else rt_slots)[j]
        best = None
        for it_seed in range(2):
            out, c, _ = refine_layout(fps, inner, init, iters=3000 if len(parts) >= 4 else 1500,
                                      rng=np.random.default_rng(it_seed))
            if best is None or c > best[1]:
                best = (out, c)
        if best[1] > 0.003:
            return best
    out, c = optimize_layout(fps, inner, rng=rng)
    return out, c
