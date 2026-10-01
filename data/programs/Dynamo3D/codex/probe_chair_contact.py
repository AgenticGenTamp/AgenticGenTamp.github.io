"""Drive the base into the nearest chair and measure contact displacement."""

import math
import numpy as np

from env_client import make_env


def read(state):
    robot = state.get_object_from_name("robot")
    base = np.array([state.get(robot, "pos_base_x"), state.get(robot, "pos_base_y")])
    chairs = {}
    for name in state.get_object_names():
        if name.startswith("obstacle_chair"):
            obj = state.get_object_from_name(name)
            chairs[name] = np.array([state.get(obj, "x"), state.get(obj, "y")])
    return base, chairs


def main():
    env = make_env()
    state, info = env.reset(seed=17)
    base0, chairs0 = read(state)
    target_name = min(chairs0, key=lambda n: np.linalg.norm(chairs0[n] - base0))
    target = chairs0[target_name]
    print("start", base0, target_name, target, "distance", np.linalg.norm(target-base0), flush=True)
    for t in range(60):
        base, chairs = read(state)
        delta = target - base
        # Drive through the chair along the original approach ray.
        if np.linalg.norm(delta) < 0.3:
            delta = target - base0
        action = np.zeros(11, dtype=np.float32)
        action[:2] = np.clip(delta / max(np.linalg.norm(delta), 1e-9) * 0.1, -0.1, 0.1)
        state, reward, term, trunc, _ = env.step(action)
        if t % 5 == 0 or term:
            base, chairs = read(state)
            print(t + 1, "base", np.round(base, 3), "chair", np.round(chairs[target_name], 3),
                  "chairmove", round(float(np.linalg.norm(chairs[target_name]-target)), 3),
                  "reward", reward, "term", term, flush=True)
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    main()
