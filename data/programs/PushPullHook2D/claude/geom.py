"""Geometry helpers shared by approach.py (stdlib+numpy only)."""
import numpy as np

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])

def perp(v):
    return np.array([-v[1], v[0]])

def dist_pt_rect(p, c, u, hl, hw):
    """Distance from point p to rectangle centered c, axis u (unit), half-length hl,
    half-width hw. 0 if inside."""
    d = np.asarray(p, dtype=float) - c
    a = d[0] * u[0] + d[1] * u[1]
    b = -d[0] * u[1] + d[1] * u[0]
    dx = max(abs(a) - hl, 0.0)
    dy = max(abs(b) - hw, 0.0)
    return np.hypot(dx, dy)

def rect_corners(c, u, hl, hw):
    w = np.array([-u[1], u[0]])
    return np.array([c + u * hl + w * hw, c + u * hl - w * hw,
                     c - u * hl + w * hw, c - u * hl - w * hw])
