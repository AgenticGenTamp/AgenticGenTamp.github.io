"""Short controlled rollouts for discovering action effects."""

import numpy as np
from env_client import make_env


def value(state, name, feat):
    return float(state.get(state.get_object_from_name(name), feat))


def positions(state):
    out = {}
    for name in state.get_object_names():
        if name == "robot" or name == "wiper_0" or name.startswith("cube_"):
            obj = state.get_object_from_name(name)
            feats = ("pos_base_x", "pos_base_y", "pos_base_rot") if name == "robot" else ("x", "y", "z")
            out[name] = tuple(round(float(state.get(obj, f)), 3) for f in feats)
    return out


def run(index, amount, steps=20, seed=0):
    env = make_env()
    state, _ = env.reset(seed=seed)
    start = positions(state)
    rewards = []
    for _ in range(steps):
        action = np.zeros(11, dtype=np.float32)
        action[index] = amount
        state, reward, term, trunc, _ = env.step(action)
        rewards.append(round(float(reward), 3))
        if term or trunc:
            break
    print("control", index, amount, "n", len(rewards), "reward", rewards[-5:])
    print(" start", start)
    print(" end  ", positions(state))
    env.close()


if __name__ == "__main__":
    for i in range(11):
        run(i, 0.1 if i < 10 else 1.0)
