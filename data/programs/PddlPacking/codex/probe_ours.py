import numpy as np
from env_client import make_env


def val(s, name, feature):
    return s.get(s.get_object_from_name(name), feature)


e = make_env()
s, info = e.reset(seed=0, options={"object_count": 3})
target = s.get_object_from_name("block0")
tx = s.get(target, "pose_x") - 0.602
ty = s.get(target, "pose_y") - 0.250
for k in range(5):
    a = np.zeros(11, np.float32)
    a[0] = np.clip(tx - val(s, "robot", "base_x"), -.2, .2)
    a[1] = np.clip(ty - val(s, "robot", "base_y"), -.2, .2)
    s, r, term, trunc, inf = e.step(a)
    print(k, val(s, "robot", "base_x"), val(s, "robot", "base_y"), val(s, "robot", "grasp_active"))
    if abs(a[0]) + abs(a[1]) < 1e-5:
        break
a = np.zeros(11, np.float32)
a[10] = -1
s, r, term, trunc, inf = e.step(a)
print("close", val(s, "robot", "grasp_active"), val(s, "robot", "gripper_opening"),
      [val(s, "block" + str(i), "grasp_active") for i in range(3)],
      [val(s, "robot", "grasp_tf_" + f) for f in ("x", "y", "z")])
e.close()
