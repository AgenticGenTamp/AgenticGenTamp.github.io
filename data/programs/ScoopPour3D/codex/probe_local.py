"""Probe each control dimension against identical resets."""

import numpy as np
from env_client import make_env


RFS = ["pos_base_x", "pos_base_y", "pos_base_rot"] + [f"pos_arm_joint{i}" for i in range(1, 8)] + ["pos_gripper"]
OFS = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def vals(state, obj, fs):
    return np.array([state.get(obj, f) for f in fs], dtype=float)


def run(dim, value, steps=10):
    env = make_env()
    s0, info = env.reset(seed=123)
    rt = env.observation_space.get_type("mujoco_tidybot_robot")
    r0 = vals(s0, s0.get_objects(rt)[0], RFS)
    named0 = {n: vals(s0, s0.get_object_from_name(n), OFS) for n in ("scoop_0", "bin_green_0", "cube_0")}
    total = 0.0
    s = s0
    for _ in range(steps):
        a = np.zeros(11, np.float32)
        a[dim] = value
        a[10] = 1.0
        s, rew, term, trunc, _ = env.step(a)
        total += rew
    r1 = vals(s, s.get_objects(rt)[0], RFS)
    moved = {n: vals(s, s.get_object_from_name(n), OFS) - v for n, v in named0.items()}
    print(dim, value, "robot_delta", np.round(r1-r0, 4), "obj_xyz", {n: np.round(d[:3], 4).tolist() for n, d in moved.items()}, "R", round(total, 3), term, trunc)
    env.close()


if __name__ == "__main__":
    for d in range(11):
        run(d, 0.1 if d < 10 else 0.0)
