"""Sweep the lowered open gripper laterally to detect cube contact."""

import numpy as np

from env_client import make_env
from probe_grasp_grid import LOWER
from probe_grasp_sequence import drive, pose, value


def shift(env, state, dx, dy):
    remaining = np.array([dx, dy], dtype=float)
    for _ in range(10):
        if np.max(np.abs(remaining)) < 0.004:
            break
        action = np.zeros(11, dtype=np.float32)
        action[:2] = np.clip(remaining / 0.87, -0.1, 0.1)
        old = np.array([value(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
        state, *_ = env.step(action)
        new = np.array([value(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
        remaining -= new - old
    return state


for approach_x in (0.10, 0.20, 0.30):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube = next(n for n in state.get_object_names() if n.startswith("cube"))
    state = shift(env, state, approach_x, -0.15)
    state = drive(env, state, LOWER, 0.0)
    start = pose(state, cube)
    touched = False
    for _ in range(16):
        state = shift(env, state, 0.0, 0.02)
        if np.linalg.norm(pose(state, cube) - start) > 0.005:
            touched = True
            break
    print("x", approach_x, "cube", np.round(start, 4).tolist(), "->",
          np.round(pose(state, cube), 4).tolist(), "touched", touched)
    env.close()
