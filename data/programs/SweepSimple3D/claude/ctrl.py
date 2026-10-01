import numpy as np, fk

BASE_GAIN = 0.87
ROT_GAIN = 0.994
ARM_GAIN = 0.25
CONT = [0,2,4,6]  # continuous joints (1,3,5,7)
Rdown = np.array([[1.0,0,0],[0,-1,0],[0,0,-1]])

def wrap_to(q, ref):
    q = np.array(q,dtype=float)
    for i in CONT:
        q[i] = ref[i] + np.remainder(q[i]-ref[i]+np.pi, 2*np.pi) - np.pi
    return q

def action(cur_j, tgt_j, cur_b=None, tgt_b=None, grip=0.0):
    a = np.zeros(11,dtype=np.float32)
    if tgt_b is not None:
        d = np.array(tgt_b,dtype=float)-np.array(cur_b,dtype=float)
        d[2] = np.remainder(d[2]+np.pi,2*np.pi)-np.pi
        a[0]=np.clip(d[0]/BASE_GAIN,-0.1,0.1)
        a[1]=np.clip(d[1]/BASE_GAIN,-0.1,0.1)
        a[2]=np.clip(d[2]/ROT_GAIN,-0.1,0.1)
    a[3:10]=np.clip((np.array(tgt_j)-np.array(cur_j))/ARM_GAIN,-0.1,0.1)
    a[10]=grip
    return a

def ik_local(pos_local, q_seed, R=None):
    """IK for tip position in the base(arm-mount) frame; returns wrapped joints."""
    q = fk.ik(np.asarray(pos_local,dtype=float), Rdown if R is None else R, q_seed, (0,0,0))
    return wrap_to(q, q_seed)
