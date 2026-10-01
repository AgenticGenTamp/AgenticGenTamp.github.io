import numpy as np
from env_client import make_env

RF = ["pos_base_x","pos_base_y","pos_base_rot","joint_1","joint_2","joint_3","joint_4",
      "joint_5","joint_6","joint_7","finger_state","grasp_active"]

def rs(obs):
    R = obs.get_object_from_name("robot")
    return np.array([float(obs.get(R,f)) for f in RF])

def fmt(v, nd=4):
    return "[" + " ".join(f"{x:+.{nd}f}" for x in v) + "]"

def act(**kw):
    a = np.zeros(11, dtype=np.float32)
    idx = dict(bx=0,by=1,br=2,j1=3,j2=4,j3=5,j4=6,j5=7,j6=8,j7=9,g=10)
    for k,v in kw.items(): a[idx[k]] = v
    return a

def objs(obs):
    out={}
    for n in obs.get_object_names():
        o=obs.get_object_from_name(n)
        if n=="robot": continue
        try:
            out[n]=(float(obs.get(o,"pose_x")),float(obs.get(o,"pose_y")),float(obs.get(o,"pose_z")))
        except Exception: pass
    return out
