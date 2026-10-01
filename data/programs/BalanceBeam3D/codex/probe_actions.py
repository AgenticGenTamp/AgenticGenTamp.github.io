"""Small black-box probes for BalanceBeam3D action semantics."""

import numpy as np

from env_client import make_env


POS = np.arange(16, 27)
VEL = np.arange(27, 38)


def one_step(seed, action):
    env = make_env()
    try:
        before, info = env.reset(seed=seed)
        after, reward, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
        return before, after, reward, term, trunc, info, env.max_steps
    finally:
        env.close()


def repeated(seed, action, count=8):
    env = make_env()
    try:
        obs, _ = env.reset(seed=seed)
        rows = [obs[POS].copy()]
        rewards = []
        for _ in range(count):
            obs, reward, term, trunc, _ = env.step(np.asarray(action, dtype=np.float32))
            rows.append(obs[POS].copy())
            rewards.append(reward)
            if term or trunc:
                break
        return np.asarray(rows), rewards
    finally:
        env.close()


def main():
    zero = np.zeros(11, dtype=np.float32)
    b0, z1, rew, term, trunc, info, max_steps = one_step(0, zero)
    print("max_steps", max_steps, "reset_info", info)
    print("initial robot pos", np.round(b0[POS], 6))
    print("zero step dpos", np.round(z1[POS] - b0[POS], 6), "vel", np.round(z1[VEL], 6), "reward", rew)
    for j in range(11):
        a = zero.copy()
        a[j] = 1.0 if j == 10 else 0.1
        before, after, reward, term, trunc, info, _ = one_step(0, a)
        print("axis", j, "dpos", np.round(after[POS] - before[POS], 6),
              "vel", np.round(after[VEL], 6), "r", reward)

    for j in (0, 2, 3, 6, 9):
        a = zero.copy(); a[j] = 0.1
        rows, _ = repeated(1, a)
        print("repeat axis", j, "delta-from-reset")
        print(np.round(rows - rows[0], 5))

    for grip in (0.0, 1.0):
        a = zero.copy(); a[10] = grip
        rows, _ = repeated(2, a, 12)
        print("gripper command", grip, "positions", np.round(rows[:, 10], 6))


if __name__ == "__main__":
    main()
