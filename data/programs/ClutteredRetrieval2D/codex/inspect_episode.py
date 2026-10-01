"""Small state/action probe for environment reverse engineering."""

import sys
import numpy as np
from env_client import make_env


def vals(s, obj, names):
    return [round(float(s.get(obj, n)), 4) for n in names]


env = make_env()
s, info = env.reset(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
types = {t.name: t for t in env.observation_space.types}
print("info", info, "max", env.max_steps)
for name in s.get_object_names():
    o = s.get_object_from_name(name)
    if name == "robot":
        fs = ["x", "y", "theta", "base_radius", "arm_joint", "arm_length",
              "vacuum", "gripper_height", "gripper_width"]
    else:
        fs = ["x", "y", "theta", "static", "z_order", "width", "height"]
    print(name, vals(s, o, fs))

robot = list(s.get_objects(types["crv_robot"]))[0]
for label, a in [("rotate", [0, 0, .196, 0, 0]),
                 ("extend", [0, 0, 0, .1, 0]),
                 ("translate", [.05, .05, 0, 0, 0])]:
    before = vals(s, robot, ["x", "y", "theta", "arm_joint", "arm_length", "vacuum"])
    s, r, term, trunc, inf = env.step(np.array(a, dtype=np.float32))
    robot = list(s.get_objects(types["crv_robot"]))[0]
    after = vals(s, robot, ["x", "y", "theta", "arm_joint", "arm_length", "vacuum"])
    print(label, before, "->", after, r, term, inf)
env.close()
