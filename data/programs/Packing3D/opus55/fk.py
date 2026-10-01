"""Forward kinematics for Kinova Gen3 7DOF (kortex URDF parameters)."""
import numpy as np

def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def T(xyz, R):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = xyz; return M

PI = np.pi
JOINTS = [
    ((0, 0, 0.15643), (PI, 0, 0)),
    ((0, 0.005375, -0.12838), (PI / 2, 0, 0)),
    ((0, -0.21038, -0.006375), (-PI / 2, 0, 0)),
    ((0, 0.006375, -0.21038), (PI / 2, 0, 0)),
    ((0, -0.20843, -0.006375), (-PI / 2, 0, 0)),
    ((0, 0.00017505, -0.10593), (PI / 2, 0, 0)),
    ((0, -0.10593, -0.00017505), (-PI / 2, 0, 0)),
]
JT = [T(np.array(x), rpy_to_R(*r)) for x, r in JOINTS]
EE_T = T(np.array([0, 0, -0.061525]), rpy_to_R(PI, 0, 0))

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# Calibrated (fit to grasped-part poses, rms 1.2e-6 m over 88 configs incl. base motion):
#   arm mount sits at (MOUNT_X, MOUNT_Y, BASE_Z) in the mobile-base frame; TCP is TOOL along flange z.
#   With these defaults fk() returns the TCP frame used by grasp_tf:
#   part_pose = fk(q)[0] @ pose(grasp_tf).
BASE_Z = -0.0052
MOUNT_X = 0.1199
MOUNT_Y = 0.0
TOOL = 0.12

def fk(q, base=(0, 0, 0), base_z=BASE_Z, tool=TOOL, mount=(MOUNT_X, MOUNT_Y)):
    Rb = Rz(base[2])
    off = Rb @ np.array([mount[0], mount[1], 0.0])
    M = T(np.array([base[0] + off[0], base[1] + off[1], base_z]), Rb)
    frames = [M.copy()]
    for i in range(7):
        M = M @ JT[i] @ T(np.zeros(3), Rz(q[i]))
        frames.append(M.copy())
    M = M @ EE_T @ T(np.array([0, 0, tool]), np.eye(3))
    frames.append(M)
    return M, frames
