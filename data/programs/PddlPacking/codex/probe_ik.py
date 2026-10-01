import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from env_client import make_env

Q0 = np.array([.677170, -.343132, 1.2, -1.466884, 1.242232, -1.954428, 2.222541])
AXES = "zyxyxyx"
LENS = [0, .4, 0, .321, 0, .1, .18]


def fk(q):
    m = np.eye(3)
    p = np.array([-.05, .188, 1.0])
    for axis, angle, length in zip(AXES, q, [.1, .4, 0, .321, 0, .1, .18]):
        m = m @ Rotation.from_euler(axis, angle).as_matrix()
        p += m @ np.array([length, 0., 0.])
    return p, m


_, R0 = fk(Q0)


def ik(xyz, seed=Q0):
    def err(q):
        p, r = fk(q)
        re = Rotation.from_matrix(R0.T @ r).as_rotvec()
        return np.r_[8 * (p - xyz), 2 * re, .015 * (q - Q0)]
    lo = [-.564, -.353, -.65, -2.12, -20, -2.0, -20]
    hi = [2.135, 1.296, 3.75, -.15, 20, -.001, 20]
    return least_squares(err, seed, bounds=(lo, hi), max_nfev=1000).x


def get(s, name, f):
    return s.get(s.get_object_from_name(name), f)


for z in [.86, .83, .80, .78, .76, .74, .72, .70]:
    q = ik(np.array([.755, .19, z]))
    print("IK", z, np.round(q, 3), np.round(fk(q)[0], 3))
    e = make_env(); s, info = e.reset(seed=0, options={"object_count": 3})
    b = s.get_object_from_name("block1")
    bx, by = s.get(b, "pose_x"), s.get(b, "pose_y")
    # Set arm at the initial base, in <=.2 increments.
    for k in range(20):
        cur = np.array([get(s, "robot", "joint_" + str(i)) for i in range(1, 8)])
        a = np.zeros(11, np.float32); a[3:10] = np.clip(q-cur, -.2, .2)
        s, *_ = e.step(a)
        if np.max(np.abs(q-cur)) < .005: break
    # Align target with desired tool relative xy while staying outside x boundary.
    tx, ty = bx-.755, by-.19
    for k in range(10):
        a = np.zeros(11, np.float32)
        a[0] = np.clip(tx-get(s,"robot","base_x"),-.2,.2)
        a[1] = np.clip(ty-get(s,"robot","base_y"),-.2,.2)
        s, *_ = e.step(a)
        if max(abs(a[0]), abs(a[1])) < .005: break
    a=np.zeros(11,np.float32); a[10]=-1
    s,*_=e.step(a)
    print("RES", z, get(s,"robot","base_x"),get(s,"robot","base_y"),
          get(s,"robot","grasp_active"),
          np.round([get(s,"robot","joint_"+str(i)) for i in range(1,8)],3))
    if get(s, "robot", "grasp_active"):
        yaw0 = 2*np.arctan2(get(s,"block1","pose_qz"), get(s,"block1","pose_qw"))
        a=np.zeros(11,np.float32); a[9]=.2
        s,*_=e.step(a)
        yaw1 = 2*np.arctan2(get(s,"block1","pose_qz"), get(s,"block1","pose_qw"))
        print("WRIST_YAW", yaw0, yaw1, "delta", ((yaw1-yaw0+np.pi)%(2*np.pi)-np.pi))
    e.close()
