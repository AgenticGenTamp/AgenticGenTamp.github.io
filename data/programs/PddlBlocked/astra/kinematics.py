"""Empirically calibrated PR2 left-arm forward and inverse kinematics."""
import math
import numpy as np
from scipy.optimize import least_squares

_LOWER = np.array([-2*math.pi, -.7146018147, -.5236, -.8, -2.3213, -2*math.pi, -2.094, -2*math.pi])
_UPPER = np.array([2*math.pi, 2.285398245, 1.3963, 3.9, 0., 2*math.pi, 0., 2*math.pi])
_AXES = (2, 2, 1, 0, 1, 0, 1, 0)
_TRANSLATIONS = (np.zeros(3), np.array([-.05,.188,.990675]), np.array([.1,0.,0.]), np.zeros(3), np.array([.4,0.,0.]), np.zeros(3), np.array([.321,0.,0.]), np.zeros(3))

def _rot(axis, theta):
    c, s = math.cos(theta), math.sin(theta)
    if axis == 0:
        return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    if axis == 1:
        return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

def _chain(x, base_xy, derivatives=False):
    p = np.array([base_xy[0],base_xy[1],0.], dtype=float)
    R = np.eye(3)
    origins, axes = [], []
    for theta, axis, offset in zip(x, _AXES, _TRANSLATIONS):
        p = p + R @ offset
        if derivatives:
            origins.append(p.copy())
            axes.append(R[:,axis].copy())
        R = R @ _rot(axis, theta)
    p = p + .18*R[:,0]
    if derivatives:
        return p, R, np.array(origins), np.array(axes)
    return p, R

def fk(config10):
    """Return world tool position and orientation for base XY, yaw, seven joints."""
    c = np.asarray(config10, dtype=float)
    return _chain(c[2:10], c[:2])

def _canonical(x):
    x = np.asarray(x, dtype=float).copy()
    for i in (0,5,7):
        x[i] = (x[i]+math.pi)%(2*math.pi)-math.pi
    return x

def solve(target_xyz, yaw, base_xy, initial=None):
    """Solve a horizontal tool pose with fixed base XY; return config and pose error.

    Error is max(position distance in metres, orientation Frobenius distance).
    An initial 10-coordinate configuration provides the first optimization seed.
    """
    target = np.asarray(target_xyz, dtype=float)
    base_xy = np.asarray(base_xy, dtype=float)
    goal_R = _rot(2, yaw)
    yaw = (float(yaw)+math.pi)%(2*math.pi)-math.pi
    seeds = []
    if initial is not None:
        initial = np.asarray(initial, dtype=float)
        seeds.append(_canonical(initial[2:10] if initial.size == 10 else initial))
    # Straight and folded configurations avoid the wrist-flex upper bound by
    # using a half-turn forearm roll for positive effective wrist bending.
    seeds.extend([
        np.array([yaw, 0., .45, 0., -.4, 0., -.05, 0.]),
        np.array([yaw, 0., .9, 0., -1.2, math.pi, -.3, math.pi]),
        np.array([yaw-.7, .7, .8, .5, -1.1, math.pi, -.3, math.pi]),
    ])
    best, best_error = None, float('inf')
    cache_x = None
    cache_value = None
    def evaluate(x):
        nonlocal cache_x, cache_value
        if cache_x is None or not np.array_equal(cache_x, x):
            p, R, origins, axes = _chain(x, base_xy, True)
            residual = np.concatenate((p-target, .35*(R-goal_R).ravel()))
            jac = np.empty((12,8))
            jac[:3,:] = np.cross(axes,p-origins).T
            for j, axis in enumerate(axes):
                # Each column of R rotates about this joint's world axis.
                jac[3:,j] = .35*np.cross(axis, R.T).T.ravel()
            cache_x, cache_value = x.copy(), (residual,jac,p,R)
        return cache_value
    for seed in seeds:
        seed = np.clip(seed, _LOWER+1e-7, _UPPER-1e-7)
        result = least_squares(lambda x:evaluate(x)[0], seed,
                               jac=lambda x:evaluate(x)[1], bounds=(_LOWER,_UPPER),
                               max_nfev=65, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        _, _, p, R = evaluate(result.x)
        error = max(float(np.linalg.norm(p-target)),float(np.linalg.norm(R-goal_R)))
        if error < best_error:
            best, best_error = result.x.copy(), error
        if error < 2e-4:
            break
    return np.concatenate((base_xy,_canonical(best))), best_error
