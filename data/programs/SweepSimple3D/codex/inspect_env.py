"""Concise black-box inspection utility (not imported by approach.py)."""

import numpy as np
from env_client import make_env


def dump(seed=0):
    env = make_env()
    state, info = env.reset(seed=seed)
    print("seed", seed, "info", info, "max_steps", env.max_steps)
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        vals = {}
        for feat in ("x", "y", "z", "bb_x", "bb_y", "bb_z",
                     "pos_base_x", "pos_base_y", "pos_base_rot",
                     "pos_gripper"):
            try:
                vals[feat] = round(float(state.get(obj, feat)), 4)
            except Exception:
                pass
        print(name, vals)
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    state, reward, term, trunc, info = env.step(action)
    print("zero step", reward, term, trunc, info)
    env.close()


if __name__ == "__main__":
    for s in range(3):
        dump(s)
