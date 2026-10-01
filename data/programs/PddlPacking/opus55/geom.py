"""Simple collision geometry: oriented boxes + SAT test."""
import numpy as np


class OBB:
    __slots__ = ("c", "R", "h")

    def __init__(self, c, R, h):
        self.c = np.asarray(c, float); self.R = np.asarray(R, float); self.h = np.asarray(h, float)


def obb_overlap(a, b, margin=0.0):
    """SAT test, True if boxes overlap (with margin added to separation)."""
    Ra, Rb = a.R, b.R
    T = b.c - a.c
    ha, hb = a.h, b.h
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    A = np.concatenate([Ra.T, Rb.T, np.cross(np.repeat(Ra.T, 3, 0), np.tile(Rb.T, (3, 1)))], 0)
    n = np.sqrt((A * A).sum(1))
    A = A[n > 1e-6] / n[n > 1e-6][:, None]
    ra = np.abs(A @ Ra) @ ha
    rb = np.abs(A @ Rb) @ hb
    return not np.any(np.abs(A @ T) > ra + rb + margin)


def seg_box(p0, p1, r):
    """OBB around segment p0->p1 with square cross-section half-width r."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    x = d / L if L > 1e-9 else np.array([1., 0, 0])
    tmp = np.array([0, 0, 1.]) if abs(x[2]) < 0.9 else np.array([1., 0, 0])
    y = np.cross(tmp, x); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return OBB((p0 + p1) / 2, np.stack([x, y, z], 1), [L / 2 + r, r, r])


def rect_overlap_2d(c1, yaw1, h1, c2, yaw2, h2, margin=0.0):
    """2D OBB overlap of rectangles."""
    def ax(y):
        return np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)])
    a1 = ax(yaw1); a2 = ax(yaw2)
    T = np.asarray(c2, float) - np.asarray(c1, float)
    for L in (a1[0], a1[1], a2[0], a2[1]):
        r1 = h1[0] * abs(a1[0] @ L) + h1[1] * abs(a1[1] @ L)
        r2 = h2[0] * abs(a2[0] @ L) + h2[1] * abs(a2[1] @ L)
        if abs(T @ L) > r1 + r2 + margin:
            return False
    return True
