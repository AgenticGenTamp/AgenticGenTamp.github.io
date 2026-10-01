"""Calibrated kinematics for the Kinova Gen3 in this environment."""
import numpy as np

def rpy_to_mat(r, p, y):
    cr, sr = np.cos(r), np.sin(r); cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

def _tf(xyz, rpy):
    T = np.eye(4); T[:3,:3] = rpy_to_mat(*rpy); T[:3,3] = xyz
    return T

_CHAIN = [
    ((0.0, 0.0, 0.15643), (-np.pi, 0.0, 0.0)),
    ((0.0, 0.005375, -0.12838), (np.pi/2, 0.0, 0.0)),
    ((0.0, -0.21038, -0.006375), (-np.pi/2, 0.0, 0.0)),
    ((0.0, 0.006375, -0.21038), (np.pi/2, 0.0, 0.0)),
    ((0.0, -0.20843, -0.006375), (-np.pi/2, 0.0, 0.0)),
    ((0.0, 0.00017505, -0.10593), (np.pi/2, 0.0, 0.0)),
    ((0.0, -0.10593, -0.00017505), (-np.pi/2, 0.0, 0.0)),
]
_OFFS = np.array([
    [1.19896065e-01, -4.43352117e-06, -1.73048726e-03],
    [-5.87556482e-07, -5.19333152e-02, 1.73048234e-03],
    [-2.50377658e-06, -8.15827673e-02, -5.19285600e-02],
    [-5.13267278e-06, 1.34557845e-01, 8.15983413e-02],
    [1.10713216e-05, -1.59888057e-01, 1.34550219e-01],
    [-6.86351502e-07, 1.98276304e-01, 1.59878446e-01],
    [-7.23624298e-06, 2.23342929e-02, 1.98280467e-01],
])
BASE_Z = -1.73055475e-03
# tool: gripper frame relative to modelled ee frame (pure translation along ee z)
TOOL_Z = 0.1423

_FIXED = []
for i, c in enumerate(_CHAIN):
    T = _tf(*c); T[:3,3] = T[:3,3] + _OFFS[i]
    _FIXED.append(T)
_EE = _tf((0.0, 0.0, -0.06153), (np.pi, 0.0, 0.0))
_TOOL = _tf((0.0, 0.0, TOOL_Z), (0.0, 0.0, 0.0))

def rotz(t):
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4); T[0,0]=c; T[0,1]=-s; T[1,0]=s; T[1,1]=c
    return T

def fk_frames(q):
    T = np.eye(4); frames = []
    for i in range(7):
        T = T @ _FIXED[i] @ rotz(q[i]); frames.append(T.copy())
    frames.append(T @ _EE @ _TOOL)
    return frames

def fk_tool(q):
    """Gripper frame in the robot base frame."""
    return fk_frames(q)[-1]

def base_tf(bx, by, brot):
    T = np.eye(4); c, s = np.cos(brot), np.sin(brot)
    T[:3,:3] = np.array([[c,-s,0],[s,c,0],[0,0,1]]); T[:3,3] = (bx, by, BASE_Z)
    return T

def jacobian(q):
    frames = fk_frames(q)
    p_ee = frames[-1][:3,3]
    J = np.zeros((6,7))
    for i in range(7):
        z = frames[i][:3,2]; p = frames[i][:3,3]
        J[:3,i] = np.cross(z, p_ee - p); J[3:,i] = z
    return J

def rot_log(Rm):
    c = np.clip((np.trace(Rm)-1)/2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    s = np.sin(ang)
    if abs(s) > 1e-6:
        return ang/(2*s)*np.array([Rm[2,1]-Rm[1,2], Rm[0,2]-Rm[2,0], Rm[1,0]-Rm[0,1]])
    # angle close to pi: use the symmetric part
    A = (Rm + np.eye(3))/2.0
    d = np.clip(np.diag(A), 0.0, 1.0)
    axis = np.sqrt(d)
    k = int(np.argmax(axis))
    if axis[k] < 1e-8:
        return np.zeros(3)
    axis = A[:, k]/axis[k]
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.zeros(3)
    return ang*axis/n

def ik_step(q, p_des, R_des, damp=0.03, max_delta=0.2, w_rot=1.0):
    T = fk_tool(q)
    ep = p_des - T[:3,3]
    ew = rot_log(R_des @ T[:3,:3].T)
    e = np.concatenate([ep, w_rot*ew])
    J = jacobian(q); J = np.vstack([J[:3], w_rot*J[3:]])
    dq = J.T @ np.linalg.solve(J @ J.T + damp**2*np.eye(6), e)
    n = np.max(np.abs(dq))
    if n > max_delta: dq = dq*(max_delta/n)
    return dq, np.linalg.norm(ep), np.linalg.norm(ew)

def solve_ik(q0, p_des, R_des, iters=200, tol=1e-5):
    q = np.array(q0, float)
    for _ in range(iters):
        dq, ep, ew = ik_step(q, p_des, R_des, damp=0.02, max_delta=0.3)
        q = q + dq
        if ep < tol and ew < 1e-4: break
    return q, ep, ew

def quat_to_mat(qx, qy, qz, qw):
    n = np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
    if n < 1e-12: return np.eye(3)
    x,y,z,w = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def down_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c,-s,0],[s,c,0],[0,0,1.0]])
    return Rz @ np.array([[1,0,0],[0,-1,0],[0,0,-1.0]])
