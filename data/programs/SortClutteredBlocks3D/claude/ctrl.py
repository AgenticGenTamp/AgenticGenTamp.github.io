import numpy as np
K_BASE = 0.87
K_JOINT = 0.249
JF = [f'pos_arm_joint{i}' for i in range(1,8)]
BF = ['pos_base_x','pos_base_y','pos_base_rot']

def robot_state(s):
    o = s.get_object_from_name('robot')
    b = np.array([float(s.get(o,f)) for f in BF])
    q = np.array([float(s.get(o,f)) for f in JF])
    g = float(s.get(o,'pos_gripper'))
    return b,q,g

def cubes(s):
    out={}
    for n in s.get_object_names():
        if n.startswith('cube'):
            o=s.get_object_from_name(n)
            out[n]=np.array([float(s.get(o,f)) for f in 'xyz'])
    return out

def action(b, q, b_tgt, q_tgt, grip):
    a=np.zeros(11,dtype=np.float32)
    db = np.array(b_tgt)-b
    db[2] = (db[2]+np.pi)%(2*np.pi)-np.pi
    a[0:3]=np.clip(db/K_BASE, -0.1, 0.1)
    dq = np.array(q_tgt)-q
    dq = (dq+np.pi)%(2*np.pi)-np.pi
    a[3:10]=np.clip(0.9*dq/K_JOINT, -0.1, 0.1)
    a[10]=grip
    return a
