"""Numerical IK for Kinova Gen3 on mobile base (uses fk.py)."""
import numpy as np
from scipy.optimize import least_squares
from fk import fk, Rz

LO = np.full(7, -np.inf); HI = np.full(7, np.inf)

def down_R(yaw):
    """EE rotation with z-axis pointing down, x-axis at angle yaw in world xy."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])

def ik(target_pos, q0, base=(0, 0, 0), yaw=None, R_target=None, w_rot=0.3,
       w_reg=1e-3, **fkkw):
    """Solve for q so EE is at target_pos with z-axis down (and x-axis at `yaw`
    if given; else yaw free). Returns (q, pos_err, rot_err)."""
    q0 = np.asarray(q0, float)
    tp = np.asarray(target_pos, float)
    def res(q):
        M, _ = fk(q, base=base, **fkkw)
        r = [M[:3, 3] - tp]
        if R_target is not None:
            r.append(w_rot * (M[:3, :3] - R_target).ravel())
        else:
            r.append(w_rot * (M[:3, 2] - np.array([0, 0, -1.0])))
            if yaw is not None:
                r.append(w_rot * (M[:3, 0] - np.array([np.cos(yaw), np.sin(yaw), 0])))
        r.append(w_reg * (q - q0))
        return np.concatenate(r)
    sol = least_squares(res, q0, method='lm', xtol=1e-12, ftol=1e-12, max_nfev=2000)
    q = sol.x
    M, _ = fk(q, base=base, **fkkw)
    perr = np.linalg.norm(M[:3, 3] - tp)
    rerr = np.linalg.norm(M[:3, 2] - np.array([0, 0, -1.0]))
    return q, perr, rerr

def wrap_near(q, qref):
    """Wrap each joint of q to within pi of qref."""
    q = np.asarray(q, float); qref = np.asarray(qref, float)
    return qref + (q - qref + np.pi) % (2 * np.pi) - np.pi

def interp_steps(q_from, q_to, max_step=0.19):
    d = np.asarray(q_to) - np.asarray(q_from)
    n = max(1, int(np.ceil(np.max(np.abs(d)) / max_step)))
    return [d / n] * n


# ---------------------------------------------------------------------------
# Grasp helpers.  Semantics (validated): part_pose = fk(q)[0] @ pose(grasp_tf),
# i.e. grasp_tf is the part pose expressed in the TCP frame returned by fk().
def quat_to_R(qx, qy, qz, qw):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

def pose_mat(xyz, quat):
    M = np.eye(4); M[:3, :3] = quat_to_R(*quat); M[:3, 3] = xyz; return M

def tcp_for_part_pose(part_xyz, part_quat, grasp_tf):
    """TCP 4x4 pose needed so that the held part lands at (part_xyz, part_quat).
    grasp_tf = [x,y,z,qx,qy,qz,qw] from the robot observation."""
    P = pose_mat(part_xyz, part_quat)
    G = pose_mat(grasp_tf[:3], grasp_tf[3:7])
    return P @ np.linalg.inv(G)

def ik_pose(Mt, q0, base=(0, 0, 0), w_rot=0.3, **kw):
    """IK to a full 4x4 TCP pose."""
    return ik(Mt[:3, 3], q0, base=base, R_target=Mt[:3, :3], w_rot=w_rot, **kw)

def handle_offset(part_type_name, triangle_type=None):
    """xy offset (world, for unrotated parts) from part pose to its grasp handle.
    Cuboids and triangle_type 0: at the pose origin; triangle_type 1: ~(+0.033,+0.033)
    (measured on one un-rotated instance; rotate by part yaw if parts are rotated)."""
    if triangle_type is not None and int(round(triangle_type)) == 1:
        return np.array([0.0333, 0.0333])
    return np.zeros(2)
