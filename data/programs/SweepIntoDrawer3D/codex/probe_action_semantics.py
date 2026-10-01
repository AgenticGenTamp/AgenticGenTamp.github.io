"""Empirically probe each action coordinate on independent seed-0 resets."""

import numpy as np

from env_client import make_env


WATCH = list(range(0, 80)) + list(range(87, 89)) + list(range(103, 109)) + list(range(125, 163))


def run(dim, value, steps=8, seed=0):
    env = make_env()
    try:
        initial, _ = env.reset(seed=seed)
        state = initial
        rewards = []
        action = np.zeros(11, dtype=np.float32)
        action[dim] = value
        for _ in range(steps):
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        return initial, state, rewards
    finally:
        env.close()


def summary(dim, sign):
    a, b, rewards = run(dim, sign)
    delta = b - a
    changed = [(i, float(delta[i])) for i in WATCH if abs(float(delta[i])) > 1e-4]
    changed.sort(key=lambda x: abs(x[1]), reverse=True)
    print(
        f"d={dim:2d} v={sign:+.1f} reward={sum(rewards):+.3f} "
        f"rseq={rewards} robot_d={np.round(delta[125:147], 4).tolist()} "
        f"wiper_xyz_d={np.round(delta[147:150], 4).tolist()} "
        f"drawers_d={np.round(delta[[87,88,103,104,105,106,107,108]],4).tolist()} "
        f"top_changes={changed[:12]}"
    )


if __name__ == "__main__":
    for d in range(11):
        for s in (0.1, -0.1) if d < 10 else (1.0, 0.0):
            summary(d, s)
