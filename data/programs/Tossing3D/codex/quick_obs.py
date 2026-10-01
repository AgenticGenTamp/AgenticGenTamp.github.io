import numpy as np
from env_client import make_env

env = make_env()
s, info = env.reset(seed=0, options={"object_count": 1})
for name in s.get_object_names():
    o = s.get_object_from_name(name)
    vals = {}
    for f in ("x", "y", "z", "bb_x", "bb_y", "bb_z", "pos_base_x", "pos_base_y", "pos_base_rot", *[f"pos_arm_joint{i}" for i in range(1,8)], "pos_gripper"):
        try: vals[f] = round(float(s.get(o,f)),4)
        except (KeyError, ValueError): pass
    print(name, vals)
env.close()
