"""Numerical kinematics for the Kinova Gen3 seven-axis arm."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_ORIGINS = np.array([
    (0., 0., .15643), (0., .005375, -.12838),
    (0., -.21038, -.006375), (0., .006375, -.21038),
    (0., -.20843, -.006375), (0., 0., -.10593),
    (0., -.10593, 0.), (0., 0., -.061525),
])
_ROLLS = np.array([np.pi, np.pi/2, -np.pi/2, np.pi/2,
                   -np.pi/2, np.pi/2, -np.pi/2, np.pi])
_FIXED = np.repeat(np.eye(4)[None], 8, axis=0)
_FIXED[:, :3, 3] = _ORIGINS
_FIXED[:, :3, :3] = Rotation.from_euler('x', _ROLLS).as_matrix()


def fk(q, tool_length=.12):
    """Return arm mount-to-tool transform for seven joint angles."""
    t = np.eye(4)
    for i in range(7):
        t = t @ _FIXED[i]
        c, s = np.cos(q[i]), np.sin(q[i])
        # Multiply rotation about local z without allocating a 4x4 matrix.
        x, y = t[:, 0].copy(), t[:, 1].copy()
        t[:, 0] = c*x + s*y
        t[:, 1] = -s*x + c*y
    t = t @ _FIXED[7]
    t[:3, 3] += tool_length*t[:3, 2]
    return t


def ik(position, orientation, seed, tool_length=.12, max_nfev=40,
       bounds=None, orientation_weight=.25, return_result=False):
    """Fit local tool pose; orientation is a 3x3 matrix or scipy xyzw quat.

    Set orientation=None to solve position only.  The seed selects the nearest
    practical branch. Optional bounds are a pair of length-seven arrays.
    """
    position = np.asarray(position, dtype=float)
    seed = np.asarray(seed, dtype=float)[:7]
    if orientation is not None:
        orientation = np.asarray(orientation, dtype=float)
        if orientation.shape == (4,):
            orientation = Rotation.from_quat(orientation).as_matrix()
    if bounds is None:
        bounds = (-np.ones(7)*2*np.pi, np.ones(7)*2*np.pi)
    lower, upper = (np.asarray(x, dtype=float) for x in bounds)
    seed = np.clip(seed, lower + 1e-9, upper - 1e-9)

    def residual(q):
        transform = fk(q, tool_length)
        error = transform[:3, 3] - position
        if orientation is not None:
            angular = Rotation.from_matrix(orientation @ transform[:3, :3].T).as_rotvec()
            error = np.concatenate((error, orientation_weight*angular))
        return error

    result = least_squares(residual, seed, bounds=(lower, upper),
                           max_nfev=max_nfev, ftol=1e-7, xtol=1e-7,
                           gtol=1e-7)
    return result if return_result else result.x


def planar_ik(x, z, pitch=0., seed=None):
    """Continuous elbow-forward pose respecting the arm's limited joints."""
    q=np.array([0.,-.34906585,np.pi,-2.5481807,0.,-.8726646,np.pi/2])
    target_rot=Rotation.from_euler('y',pitch).as_matrix()@fk(q)[:3,:3]
    def error(v):
        q[[1,3,5]]=v
        t=fk(q)
        return np.r_[t[0,3]-x,t[2,3]-z,.3*Rotation.from_matrix(target_rot@t[:3,:3].T).as_rotvec()]
    start=[.8,-1.7,-.5] if seed is None else np.asarray(seed)[[1,3,5]]
    r=least_squares(error,start,bounds=([-2.24,-2.58,-2.1],[2.24,2.58,2.1]),max_nfev=50)
    q[[1,3,5]]=r.x
    return q
