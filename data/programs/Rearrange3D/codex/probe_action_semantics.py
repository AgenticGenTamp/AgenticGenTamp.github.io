"""Concise black-box probes for Rearrange3D action semantics."""

import numpy as np

from env_client import make_env


def summary(obs):
    return np.r_[obs[0:3], obs[16:19], obs[32:35], obs[93:104]]


def run(label, actions, seed=0):
    env = make_env()
    obs, info = env.reset(seed=seed)
    start = obs.copy()
    rewards = []
    for action in actions:
        obs, reward, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        rewards.append(reward)
        if term or trunc:
            break
    delta = summary(obs) - summary(start)
    print(label, "n", len(rewards), "rew", np.round(rewards, 3).tolist())
    print("  start xyz objs/base+q+g", np.round(summary(start), 3).tolist())
    print("  delta                    ", np.round(delta, 4).tolist())
    print("  final vel base/joints/g  ", np.round(obs[104:115], 4).tolist())
    env.close()


def action(index=None, value=0.0, grip=0.0):
    a = np.zeros(11, dtype=np.float32)
    a[10] = grip
    if index is not None:
        a[index] = value
    return a


if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset(seed=0)
    print("spaces", env.action_space, "max_steps", env.max_steps)
    print("seed0 obs selected", np.round(summary(obs), 3).tolist(), "info", info)
    env.close()
    run("zero x1", [action(grip=obs[103])])
    run("zero x10", [action(grip=obs[103])] * 10)
    for i in range(11):
        grip = float(obs[103]) if i != 10 else 1.0
        run("dim%d + x1" % i, [action(i if i != 10 else None, 0.1, grip)])
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        run("dim%d + x10" % i, [action(i, 0.1, float(obs[103]))] * 10)
