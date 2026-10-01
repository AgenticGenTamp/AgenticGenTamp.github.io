import numpy as np
from fk import ik, fk

K_BASE = 0.87
K_JOINT = 0.249
JF = [f'pos_arm_joint{i}' for i in range(1,8)]
BF = ['pos_base_x','pos_base_y','pos_base_rot']
MOUNT = np.array([0.0, 0.0, 0.386])   # arm base in robot-base frame
TH = np.pi                             # fixed base yaw
R2 = np.array([[np.cos(TH), -np.sin(TH)],[np.sin(TH), np.cos(TH)]])

def robot_state(s):
    o = s.get_object_from_name('robot')
    b = np.array([float(s.get(o,f)) for f in BF])
    q = np.array([float(s.get(o,f)) for f in JF])
    g = float(s.get(o,'pos_gripper'))
    return b,q,g

def cube_dict(s):
    out={}
    for n in s.get_object_names():
        if n.startswith('cube'):
            o=s.get_object_from_name(n)
            out[n]=np.array([float(s.get(o,f)) for f in 'xyz'])
    return out

def tool_R(yaw=0.0):
    c,sn=np.cos(yaw),np.sin(yaw)
    Rz=np.array([[c,-sn,0],[sn,c,0],[0,0,1.0]])
    return Rz@np.array([[1.,0,0],[0,-1,0],[0,0,-1]])

def gripper_world(b, q, mount=MOUNT):
    p = fk(q)[:3,3]
    th = b[2]
    c,s = np.cos(th), np.sin(th)
    R = np.array([[c,-s],[s,c]])
    xy = b[:2] + R@(mount[:2]+p[:2])
    return np.array([xy[0], xy[1], mount[2]+p[2]])

def base_for(world_xy, q, mount=MOUNT, th=TH):
    p = fk(q)[:3,3]
    c,s = np.cos(th), np.sin(th)
    R = np.array([[c,-s],[s,c]])
    return np.array(world_xy) - R@(mount[:2]+p[:2])

def make_action(b, q, b_tgt, q_tgt, grip, gain=0.7):
    a=np.zeros(11,dtype=np.float32)
    db = np.array(b_tgt)-b
    db[2] = (db[2]+np.pi)%(2*np.pi)-np.pi
    a[0:3]=np.clip(db/K_BASE, -0.1, 0.1)
    dq = np.array(q_tgt)-q
    dq = (dq+np.pi)%(2*np.pi)-np.pi
    a[3:10]=np.clip(gain*dq/K_JOINT, -0.1, 0.1)
    a[10]=grip
    return a
