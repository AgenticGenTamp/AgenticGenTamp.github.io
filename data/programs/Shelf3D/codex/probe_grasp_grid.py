"""Grid-search base alignment around a low, vertical grasp posture."""

import numpy as np

from env_client import make_env
from probe_grasp_sequence import HIGH, drive, pose, value


LOWER = np.array([-0.216, 2.363, 3.140, -0.577, 0.006, -0.202, 1.571])


def trial(dx, dy):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube = next(n for n in state.get_object_names() if n.startswith("cube"))
    # A full base command produces about 0.87 times its nominal displacement.
    remaining = np.array([0.30 + dx, dy])
    while np.max(np.abs(remaining)) > 0.005:
        action = np.zeros(11, dtype=np.float32)
        action[:2] = np.clip(remaining / 0.87, -0.1, 0.1)
        old = np.array([value(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
        state, *_ = env.step(action)
        new = np.array([value(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
        remaining -= new - old
    state = drive(env, state, LOWER, 0.0)
    before = pose(state, cube)
    if dx == 0.0 and dy == 0.0:
        print("render", env.render_state(state=state, label="deep_close_grasp"))
    for _ in range(8):
        action = np.zeros(11, dtype=np.float32)
        action[10] = 1.0
        state, *_ = env.step(action)
    state = drive(env, state, HIGH, 1.0)
    after = pose(state, cube)
    env.close()
    print(round(dx, 3), round(dy, 3), np.round(before, 4).tolist(),
          "->", np.round(after, 4).tolist())


if __name__ == "__main__":
    for dx in (0.0,):
        for dy in (-0.04, 0.0, 0.04):
            trial(dx, dy)
