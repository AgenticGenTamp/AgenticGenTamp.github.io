import numpy as np

DH = [
    (0.0,     -0.2848, 0.0,  np.pi/2),
    (np.pi,   -0.0118, 0.0,  np.pi/2),
    (np.pi,   -0.4208, 0.0,  np.pi/2),
    (np.pi,   -0.0128, 0.0,  np.pi/2),
    (np.pi,   -0.3143, 0.0,  np.pi/2),
    (np.pi,    0.0,    0.0,  np.pi/2),
    (np.pi,   -0.1674, 0.0,  np.pi),
]
QLO = np.array([-1e9, -2.20, -1e9, -2.55, -1e9, -2.05, -1e9])
QHI = np.array([ 1e9,  1.80,  1e9,  2.55,  1e9,  2.05,  1e9])
QNOM = np.array([0.0, 0.35, np.pi, -2.0, 0.0, 0.9, np.pi/2])

def dh_T(th, d, a, al):
    ct, st = np.cos(th), np.sin(th)
    ca, sa = np.cos(al), np.sin(al)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,     ca,    d],
        [0.0,   0.0,    0.0,  1.0],
    ])

_T0 = np.array([[1.,0,0,0],[0,-1.,0,0],[0,0,-1.,0],[0,0,0,1.]])

def fk_chain(q, tool_offset=0.0):
    T = _T0.copy(); out=[]
    for i in range(7):
        tho, d, a, al = DH[i]
        T = T @ dh_T(q[i] + tho, d, a, al)
        out.append(T.copy())
    if tool_offset:
        Tt = np.eye(4); Tt[2,3] = tool_offset
        out.append(T @ Tt)
    return out

def fk(q, tool_offset=0.0):
    return fk_chain(q, tool_offset)[-1]

def jacobian(q, tool_offset=0.0):
    Ts = fk_chain(q, tool_offset)
    pe = Ts[-1][:3,3]
    J = np.zeros((6,7))
    Tprev = _T0.copy()
    for i in range(7):
        z = Tprev[:3,2]; p = Tprev[:3,3]
        J[:3,i] = np.cross(z, pe - p)
        J[3:,i] = z
        Tprev = Ts[i]
    return J

def ik_step(q, p_des, z_des, tool_offset=0.0, yaw_des=None,
            pos_w=1.0, rot_w=0.5, damp=0.06, null_w=0.02):
    """Resolved-rate step. z_des: desired tool z-axis (unit) in arm-base frame.
    yaw_des: optional desired tool x-axis direction (unit, weak)."""
    T = fk(q, tool_offset)
    ep = (p_des - T[:3,3]) * pos_w
    zc = T[:3,2]
    er = np.cross(zc, z_des)
    s = np.linalg.norm(er); c = float(np.dot(zc, z_des))
    ang = np.arctan2(s, c)
    if s > 1e-9:
        er = er / s * ang
    else:
        er = np.zeros(3) if c > 0 else np.array([1.0,0,0])*np.pi
    er = er * rot_w
    if yaw_des is not None:
        xc = T[:3,0]
        ey = np.cross(xc, yaw_des)
        er = er + 0.25*rot_w*ey
    e = np.concatenate([ep, er])
    J = jacobian(q, tool_offset)
    JT = J.T
    Jinv = JT @ np.linalg.inv(J @ JT + (damp**2)*np.eye(6))
    dq = Jinv @ e
    # null-space: pull toward nominal / away from limits
    lo = np.maximum(QLO, -1e6); hi = np.minimum(QHI, 1e6)
    dn = np.zeros(7)
    for i in (1,3,5):
        mid = 0.5*(QLO[i]+QHI[i])
        dn[i] = -(q[i]-mid)
    dn = dn * null_w
    dq = dq + (np.eye(7) - Jinv @ J) @ dn
    # respect limits
    for i in (1,3,5):
        if q[i] + dq[i] < QLO[i]: dq[i] = QLO[i]-q[i]
        if q[i] + dq[i] > QHI[i]: dq[i] = QHI[i]-q[i]
    return dq
