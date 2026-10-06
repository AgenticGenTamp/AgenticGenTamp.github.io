"""Exact (x, y, heading-bin) time-optimal refinement of a fixed op sequence.

Heading bins are anchored at the rover's current heading th0:
bin k <-> th0 + k * BIN_W, so +-5 bin moves (0.3927 rad) are exact.
"""
import math

import numpy as np
from scipy.ndimage import minimum_filter1d

NB3 = 80
BW3 = 2 * math.pi / NB3
HB = 5  # heading bins per step
KX = 4
INF = 10**6


def minf3(E):
    a = minimum_filter1d(E, 2 * KX + 1, axis=0, mode="constant", cval=INF)
    a = minimum_filter1d(a, 2 * KX + 1, axis=1, mode="constant", cval=INF)
    return minimum_filter1d(a, 2 * HB + 1, axis=2, mode="wrap")


def prop3(D, free3, limit=None):
    E = D
    it = 0
    while True:
        N = np.minimum(E, minf3(E) + 1)
        N[~free3] = INF
        if np.array_equal(N, E):
            return E
        E = N
        it += 1
        if limit is not None and it >= limit:
            return E


def _window(arr, c, b):
    """Return values and offsets of the 9x9x11 neighbourhood."""
    i, j = c
    i0, i1 = max(0, i - KX), min(arr.shape[0], i + KX + 1)
    j0, j1 = max(0, j - KX), min(arr.shape[1], j + KX + 1)
    bins = (b + np.arange(-HB, HB + 1)) % NB3
    sub = arr[i0:i1, j0:j1][:, :, bins]
    return sub, i0, j0, bins


def _pick(arr, c, b, v):
    sub, i0, j0, bins = _window(arr, c, b)
    idx = np.argwhere(sub == v)
    if len(idx) == 0:
        return None
    di = idx[:, 0] + i0 - c[0]
    dj = idx[:, 1] + j0 - c[1]
    db = idx[:, 2] - HB
    score = np.maximum(np.abs(di), np.abs(dj)) * 100 + (np.abs(di) + np.abs(dj)) * 10 \
        + np.abs(db)
    k = int(np.argmin(score))
    return (int(idx[k, 0] + i0), int(idx[k, 1] + j0)), int(bins[idx[k, 2]])


def solve(free2, start_cell, masks, home3, need_end=True):
    """masks: list of 3D bool arrays (op feasibility at arrival pose).

    Returns (cost, steps) with steps a list of (cell, bin, op_index or None),
    or None if infeasible.
    """
    shape = free2.shape + (NB3,)
    free3 = np.broadcast_to(free2[:, :, None], shape)
    D = np.full(shape, INF, dtype=np.int32)
    D[start_cell[0], start_cell[1], 0] = 0
    F = [prop3(D, free3)]
    A = [None]
    for m in masks:
        Ak = np.where(m & free3, minf3(F[-1]) + 1, INF).astype(np.int32)
        if Ak.min() >= INF:
            return None
        A.append(Ak)
        F.append(prop3(Ak, free3))
    H = np.where(home3 & free3, F[-1], INF)
    flat = int(np.argmin(H))
    cost = int(H.flat[flat])
    if cost >= INF:
        return None
    i, j, b = np.unravel_index(flat, shape)
    c, b, v = (int(i), int(j)), int(b), cost
    k = len(masks)
    steps = []
    while not (k == 0 and v == 0):
        # move within stage k
        p = _pick(F[k], c, b, v - 1) if v > 0 else None
        if p is not None:
            steps.append((c, b, None))
            c, b = p
            v -= 1
            continue
        if k == 0 or A[k][c[0], c[1], b] != v:
            raise RuntimeError("3d backtrack failed")
        p = _pick(F[k - 1], c, b, v - 1)
        if p is None:
            raise RuntimeError("3d backtrack failed (op)")
        steps.append((c, b, k - 1))
        c, b = p
        v -= 1
        k -= 1
    steps.reverse()
    return cost, steps
