import numpy as np, fk

RF=["base_x","base_y","base_rot","joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7","gripper_opening","grasp_active"]
GTF=["grasp_tf_x","grasp_tf_y","grasp_tf_z","grasp_tf_qx","grasp_tf_qy","grasp_tf_qz","grasp_tf_qw"]
BF=["pose_x","pose_y","pose_z","pose_qx","pose_qy","pose_qz","pose_qw","grasp_active"]

def rob(s):
    r=s.get_object_from_name("robot"); return np.array([s.get(r,f) for f in RF])
def gtf(s):
    r=s.get_object_from_name("robot"); return np.array([s.get(r,f) for f in GTF])
def blocks(s):
    out={}
    for n in s.get_object_names():
        if n.startswith("block"):
            b=s.get_object_from_name(n)
            out[n]=np.array([s.get(b,f) for f in BF])
    return out
def targR(a):
    x=np.array([0,0,-1.]); y=np.array([np.cos(a),np.sin(a),0]); z=np.cross(x,y)
    return np.column_stack([x,y,z])
def quat_yaw(q):  # q = qx,qy,qz,qw
    return 2*np.arctan2(q[2],q[3])
def align_angle(yaw, ref=np.pi/2):
    best=None
    for k in range(-4,5):
        c=yaw+k*np.pi/2
        d=abs(((c-ref+np.pi)%(2*np.pi))-np.pi)
        if best is None or d<best[0]: best=(d,c)
    return best[1]
def wrapd(d, idxs=(4,6)):
    d=d.copy()
    for j in idxs: d[j]=(d[j]+np.pi)%(2*np.pi)-np.pi
    return d
