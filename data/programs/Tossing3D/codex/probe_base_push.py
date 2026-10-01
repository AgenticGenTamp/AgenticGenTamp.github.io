"""Probe whether the mobile base can contact and push the floor cube."""

import math

import numpy as np

from env_client import make_env


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.asarray([float(state.get(obj, f)) for f in ("x", "y", "z")])


def base_xy(state):
    obj = state.get_object_from_name("robot")
    return np.asarray([float(state.get(obj, "pos_base_x")),
                       float(state.get(obj, "pos_base_y"))])


def run(label, commands, grip):
    env = make_env()
    state, _ = env.reset(seed=0)
    cube0 = xyz(state, "cube_0")
    base0 = base_xy(state)
    min_dist = math.inf
    total = 0.0
    max_cube_shift = 0.0
    contact_step = None
    step_num = 0
    for dx, dy, count in commands:
        action = np.zeros(18, dtype=np.float32)
        action[0] = dx
        action[1] = dy
        action[10] = grip
        for _ in range(count):
            step_num += 1
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            cube = xyz(state, "cube_0")
            shift = float(np.linalg.norm(cube[:2] - cube0[:2]))
            distance = float(np.linalg.norm(base_xy(state) - cube[:2]))
            min_dist = min(min_dist, distance)
            max_cube_shift = max(max_cube_shift, shift)
            if contact_step is None and shift > 0.01:
                contact_step = step_num
            if terminated or truncated:
                break
    cubef = xyz(state, "cube_0")
    basef = base_xy(state)
    print(label, "grip", grip, "steps", step_num,
          "base", np.round(base0, 3), "->", np.round(basef, 3),
          "cube", np.round(cube0, 3), "->", np.round(cubef, 3),
          "cube_shift", round(float(np.linalg.norm(cubef[:2] - cube0[:2])), 3),
          "max_shift", round(max_cube_shift, 3), "min_dist", round(min_dist, 3),
          "first_shift_step", contact_step, "reward", round(total, 3))
    env.close()


def main():
    # Seed 0: base=(0.036,-0.047), cube=(0.703,0.206). Commands are
    # incremental world-frame base targets. Slow commands make impacts gentler.
    paths = {
        "direct_slow": [(0.035, 0.013, 25)],
        "align_y_then_x": [(0.0, 0.035, 8), (0.035, 0.0, 25)],
        "x_only": [(0.035, 0.0, 25)],
        "direct_fast": [(0.08, 0.03, 12)],
        # Drive into the cube and then continue toward the barrier.
        "direct_then_x": [(0.035, 0.013, 20), (0.04, 0.0, 20)],
    }
    for label, commands in paths.items():
        for grip in (0.0, 1.0):
            run(label, commands, grip)


if __name__ == "__main__":
    main()
