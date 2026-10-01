import numpy as np
from env_client import make_env


def xyz(s, name):
    o = s.get_object_from_name(name)
    return np.array([float(s.get(o, f)) for f in ("x", "y", "z")])


def joints(s):
    o = s.get_object_from_name("robot")
    return np.array([float(s.get(o, f"pos_arm_joint{i}")) for i in range(1, 8)])


for seed in range(3):
    env = make_env()
    s, _ = env.reset(seed=seed, options={"object_count": 1})
    p0 = xyz(s, "cube_0")
    # Close at the task's default arm pose, then sweep lift joints separately.
    a = np.zeros(18, np.float32)
    a[10] = 1.0
    for _ in range(20):
        s, r, t, tr, i = env.step(a)
    pc = xyz(s, "cube_0")
    q0 = joints(s)
    for j in (1, 3, 5):
        a[:] = 0
        a[10] = 1.0
        a[3 + j] = -0.1
        for _ in range(15):
            s, r, t, tr, i = env.step(a)
    print(seed, "cube", p0.round(3), pc.round(3), xyz(s,"cube_0").round(3), "q", q0.round(2), joints(s).round(2), "r", r)
    env.close()
