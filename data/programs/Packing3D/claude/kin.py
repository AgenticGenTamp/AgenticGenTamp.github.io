"""Calibrated kinematics for the black-box Kinova Gen3 in this env.

fk.py's ROTATIONS are exact; its link TRANSLATIONS are wrong. Fitting
p = sum_i R_i(q) a_i (linear, rotations known) to measured grasped-part
poses gives an exact model (residual 0.0).
"""
import numpy as np
from fk import fk
from ik import fk_full

# a[0..7]: translation before joint i expressed in frame i-1 (a[7] uses frame 7).
A = np.array([[ 0.1199 ,  0.0    ,  0.13981],
              [ 0.0    ,  0.00588, -0.13981],
              [ 0.0    , -0.21038, -0.00588],
              [ 0.0    ,  0.00638, -0.21038],
              [ 0.0    , -0.15718, -0.00638],
              [ 0.0    ,  0.00017, -0.15718],
              [ 0.0    , -0.18362, -0.00017],
              [-0.01989,  0.0    , -0.18362]])
# offset of the part center in the fk tool frame at the calibration grasp
G0 = np.array([-0.0199, 0.0, 0.0798])

def linpos(q):
    """Position (robot-base frame) where a part grasped like the calibration
    grasp sits.  Exact."""
    M, Ms = fk_full(q)
    Rs = [np.eye(3)] + [m[:3, :3] for m in Ms]      # R_0..R_7
    return sum(Rs[i] @ A[i] for i in range(8))

def tool(q):
    """(position, rotation) of the true gripper frame, base frame. Exact."""
    R = fk(q)[:3, :3]
    return linpos(q) - R @ G0, R

def pose(q):
    """4x4 pose whose translation is linpos and rotation is the tool rotation."""
    M = np.eye(4); M[:3, :3] = fk(q)[:3, :3]; M[:3, 3] = linpos(q); return M

def jac(q, eps=1e-6):
    J = np.zeros((3, 7)); p0 = linpos(q)
    for i in range(7):
        qq = np.array(q, float); qq[i] += eps
        J[:, i] = (linpos(qq) - p0) / eps
    return J

def roterr(Rc, Rt):
    Re = Rt @ Rc.T
    w = np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]])
    s = np.linalg.norm(w); c = (np.trace(Re)-1)/2
    ang = np.arctan2(s/2, c)
    return w/s*ang if s > 1e-9 else np.zeros(3)

def ikin(q0, target_pos, target_R, iters=300, tol=1e-5):
    """IK on the calibrated model: linpos(q)=target_pos, R(q)=target_R."""
    q = np.array(q0, float)
    for _ in range(iters):
        M, Ms = fk_full(q)
        Rs = [np.eye(3)] + [m[:3,:3] for m in Ms]
        p = sum(Rs[i] @ A[i] for i in range(8))
        R = M[:3,:3]
        Jp = np.zeros((3,7)); Jw = np.zeros((3,7))
        for i in range(7):
            z = Ms[i][:3,2]; pj = Ms[i][:3,3]
            # position jacobian via numeric (chain differs from fk translations)
            Jw[:,i] = z
        eps = 1e-6
        for i in range(7):
            qq = q.copy(); qq[i] += eps
            Jp[:,i] = (linpos(qq)-p)/eps
        e = np.zeros(6); e[:3] = target_pos - p; e[3:] = roterr(R, target_R)*0.5
        J = np.vstack([Jp, Jw])
        lam = 0.05
        dq = J.T @ np.linalg.solve(J@J.T + lam**2*np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.3: dq = dq/n*0.3
        q = q + dq
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(roterr(R,target_R)) < 1e-4:
            break
    return q, linpos(q), fk(q)[:3,:3]
