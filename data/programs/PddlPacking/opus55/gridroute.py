"""Step-optimal base routing around the table on a fine grid.

Base moves are Chebyshev-limited (|dx|,|dy| <= STEP per step) and collisions are
only checked at step endpoints, so the minimum number of steps is found by
repeated dilation of the reachable set intersected with the free mask.
"""
import math
import numpy as np
from scipy.ndimage import maximum_filter

RES = 0.025
W = 8  # STEP / RES
LIM = 1.7
MARGIN = 0.012


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _free_mask(X, Y, t):
    c, s = abs(math.cos(t)), abs(math.sin(t))
    hx, hy = 0.335, 0.365
    ex = hx * c + hy * s
    ey = hx * s + hy * c
    sep = (np.abs(X) > 0.3 + ex + MARGIN) | (np.abs(Y) > 0.6 + ey + MARGIN)
    ct, st = math.cos(t), math.sin(t)
    tu = 0.3 * c + 0.6 * s
    tv = 0.3 * s + 0.6 * c
    sep |= np.abs(X * ct + Y * st) > hx + tu + MARGIN
    sep |= np.abs(-X * st + Y * ct) > hy + tv + MARGIN
    tw = _wrap(t)
    e = 0.36 * abs(math.sin(t))
    if abs(tw) < 0.16:
        sep |= (X >= -0.70 + e) & (X <= -0.455 - e) & (np.abs(Y) <= 0.23 - e)
    if abs(_wrap(t - math.pi)) < 0.16:
        sep |= (X >= 0.455 + e) & (X <= 0.70 - e) & (np.abs(Y) <= 0.23 - e)
    return sep


def grid_path(b0, b1, base_ok, step=0.2, max_layers=40):
    """Return list of base poses (excluding b0, ending at b1) or None."""
    x0, y0, t0 = float(b0[0]), float(b0[1]), float(b0[2])
    x1, y1, t1 = float(b1[0]), float(b1[1]), float(b1[2])
    dyaw = _wrap(t1 - t0)
    # grid aligned with start
    i_lo = -int((x0 + LIM) / RES); i_hi = int((LIM - x0) / RES)
    j_lo = -int((y0 + LIM) / RES); j_hi = int((LIM - y0) / RES)
    xs = x0 + RES * np.arange(i_lo, i_hi + 1)
    ys = y0 + RES * np.arange(j_lo, j_hi + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    mask = _free_mask(X, Y, t0)
    if abs(dyaw) > 1e-6:
        nseg = max(1, int(math.ceil(abs(dyaw) / 0.3)))
        for q in range(1, nseg + 1):
            mask &= _free_mask(X, Y, t0 + dyaw * q / nseg)
    si, sj = -i_lo, -j_lo
    R = np.zeros_like(mask)
    R[si, sj] = True
    layers = [R]
    # goal window: cells within step of goal
    gi = (x1 - x0) / RES - i_lo
    gj = (y1 - y0) / RES - j_lo
    wi0 = max(0, int(math.ceil(gi - step / RES - 1e-6))); wi1 = min(len(xs) - 1, int(math.floor(gi + step / RES + 1e-6)))
    wj0 = max(0, int(math.ceil(gj - step / RES - 1e-6))); wj1 = min(len(ys) - 1, int(math.floor(gj + step / RES + 1e-6)))
    if wi0 > wi1 or wj0 > wj1:
        return None
    k = None
    for L in range(1, max_layers + 1):
        Rp = layers[-1]
        if abs(dyaw) <= step * L + 1e-9 and Rp[wi0:wi1 + 1, wj0:wj1 + 1].any():
            k = L
            break
        Rn = maximum_filter(Rp, size=2 * W + 1, mode='constant', cval=False) & mask
        layers.append(Rn)
    if k is None:
        return None
    # backtrack
    pts = [(x1, y1)]
    ti, tj = gi, gj
    for L in range(k - 1, 0, -1):
        R = layers[L]
        a0 = max(0, int(math.ceil(ti - W - 1e-6))); a1 = min(len(xs) - 1, int(math.floor(ti + W + 1e-6)))
        c0 = max(0, int(math.ceil(tj - W - 1e-6))); c1 = min(len(ys) - 1, int(math.floor(tj + W + 1e-6)))
        sub = R[a0:a1 + 1, c0:c1 + 1]
        idx = np.argwhere(sub)
        if len(idx) == 0:
            return None
        # prefer cell nearest to the previous-layer frontier direction: closest to start-side
        ii = idx[:, 0] + a0; jj = idx[:, 1] + c0
        best = np.argmin(np.maximum(np.abs(ii - ti), np.abs(jj - tj)) * 1000
                         + (ii - ti) ** 2 + (jj - tj) ** 2)
        ti, tj = float(ii[best]), float(jj[best])
        pts.append((x0 + RES * (ti + i_lo), y0 + RES * (tj + j_lo)))
    pts = pts[::-1]
    out = []
    for n, (px, py) in enumerate(pts, start=1):
        yaw = t0 + dyaw * n / k if n < k else t1
        b = (px, py, yaw)
        if not base_ok(b):
            return None
        out.append(b)
    return out
