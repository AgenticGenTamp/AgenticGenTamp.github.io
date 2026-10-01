"""Validate the lowest visually observed staged shoulder/elbow pose."""
import numpy as np
from env_client import make_env


def g(s, name, feat):
    return float(s.get(s.get_object_from_name(name), feat))


env = make_env()
state, _ = env.reset(seed=0, options={"object_count": 1})
cube0 = np.array([g(state, "cube1", f) for f in ("x", "y", "z")])
for step in range(260):
    a = np.zeros(11, np.float32)
    q2 = g(state, "robot", "pos_arm_joint2")
    q4 = g(state, "robot", "pos_arm_joint4")
    a[4] = np.clip(2 * (2.2 - q2), -.1, .1)
    if step > 100:
        a[6] = np.clip(2 * (1.0 - q4), -.1, .1)
    # Image/world calibration places the tucked tool about +0.18 m base-x.
    a[0] = np.clip(2 * (cube0[0] - .18 - g(state, "robot", "pos_base_x")), -.1, .1)
    a[1] = np.clip(2 * (cube0[1] - g(state, "robot", "pos_base_y")), -.1, .1)
    state, _, _, _, _ = env.step(a)
print("pose", [round(g(state, "robot", f"pos_arm_joint{i}"), 3) for i in range(1, 8)],
      "base", round(g(state, "robot", "pos_base_x"), 3))
for step in range(30):
    a = np.zeros(11, np.float32)
    a[4] = np.clip(2 * (2.2 - g(state, "robot", "pos_arm_joint2")), -.1, .1)
    a[6] = np.clip(2 * (1.0 - g(state, "robot", "pos_arm_joint4")), -.1, .1)
    a[10] = 1.0
    state, _, _, _, _ = env.step(a)
    cube = np.array([g(state, "cube1", f) for f in ("x", "y", "z")])
    if np.linalg.norm(cube - cube0) > .002:
        print("CONTACT", step, cube.tolist())
        break
else:
    print("NO CONTACT", cube.tolist())
env.close()
