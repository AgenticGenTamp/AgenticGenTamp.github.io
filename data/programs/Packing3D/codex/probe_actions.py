"""Measure one-step action effects from identical Packing3DEnv resets."""

import numpy as np

from env_client import make_env


def snapshot(state):
    result = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        result[name] = {
            feature: state.get(obj, feature)
            for feature in state.type_features[obj.type]
        }
    return result


def changes(before, after, tolerance=1e-7):
    result = []
    for name in sorted(before):
        for key, old in before[name].items():
            new = after[name][key]
            if abs(new - old) > tolerance:
                result.append(f"{name}.{key}: {old:.6g} -> {new:.6g} ({new-old:+.6g})")
    return result


def main():
    probes = [("zero", 0, 0.0), ("base_x_pos", 0, 0.2),
              ("base_x_neg", 0, -0.2),
              ("base_y", 1, 0.2), ("base_rot", 2, 0.2),
              ("joint1", 3, 0.2), ("joint4", 6, 0.2),
              ("joint7", 9, 0.2), ("close", 10, -1.0),
              ("open", 10, 1.0)]
    for label, index, value in probes:
        env = make_env()
        try:
            state, _ = env.reset(seed=0, options={"object_count": 1})
            before = snapshot(state)
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            action[index] = value
            state, reward, terminated, truncated, info = env.step(action)
            print(f"{label}: reward={reward} term={terminated} trunc={truncated} info={info}")
            for line in changes(before, snapshot(state)):
                print("  " + line)
        finally:
            env.close()


if __name__ == "__main__":
    main()
