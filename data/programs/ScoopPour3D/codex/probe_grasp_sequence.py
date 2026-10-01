"""Try both gripper polarities while approaching and retreating from the scoop."""

import numpy as np
from env_client import make_env


def vec(s, name):
    o = s.get_object_from_name(name)
    if name == "robot":
        fs = ("pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper")
    else:
        fs = ("x", "y", "z", "qw", "qx", "qy", "qz")
    return np.array([s.get(o, f) for f in fs])


def phase(env, s, n, x=0.0, y=0.0, grip=0.0, joint=None):
    for _ in range(n):
        a = np.zeros(11, np.float32)
        a[0], a[1], a[10] = x, y, grip
        if joint is not None:
            a[joint] = 0.1
        s, r, done, trunc, _ = env.step(a)
    return s


def trial(mode):
    env = make_env()
    s, _ = env.reset(seed=0)
    p0 = vec(s, "scoop_0")
    open_cmd, close_cmd = mode
    s = phase(env, s, 5, grip=open_cmd)
    s = phase(env, s, 6, x=0.1, grip=open_cmd)
    s = phase(env, s, 8, grip=close_cmd)
    caught = vec(s, "scoop_0") - p0
    s = phase(env, s, 8, x=-0.1, grip=close_cmd)
    retreat = vec(s, "scoop_0") - p0
    s = phase(env, s, 10, y=0.1, grip=close_cmd)
    lateral = vec(s, "scoop_0") - p0
    print("open,close", mode, "caught", np.round(caught, 4),
          "retreat", np.round(retreat, 4), "lateral", np.round(lateral, 4),
          "robot", np.round(vec(s, "robot"), 4))
    env.close()


if __name__ == "__main__":
    trial((0.0, 1.0))
    trial((1.0, 0.0))
