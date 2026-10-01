from env_client import make_env
import numpy as np


def val(s, o, f): return float(s.get(o, f))


e = make_env()
for seed in range(3):
    s, info = e.reset(seed=seed)
    print("SEED", seed, "info", info, "names", s.get_object_names())
    for name in s.get_object_names():
        o = s.get_object_from_name(name)
        try:
            fs = ['base_x','base_y','base_rot','joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7','gripper_opening','grasp_active','grasp_tf_x','grasp_tf_y','grasp_tf_z']
            print(name, {f:round(val(s,o,f),4) for f in fs})
        except Exception:
            fs = ['pose_x','pose_y','pose_z','half_extent_x','half_extent_y','half_extent_z','grasp_active']
            out = {}
            for f in fs:
                try: out[f] = round(val(s,o,f),4)
                except Exception: pass
            print(name, out)
e.close()
