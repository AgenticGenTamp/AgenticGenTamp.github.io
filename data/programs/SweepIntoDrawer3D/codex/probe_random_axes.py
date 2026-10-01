import numpy as np
from env_client import make_env


def run(axis, value, steps=10, seed=0):
    env = make_env()
    o0, _ = env.reset(seed=seed)
    o = o0
    total = 0.0
    rewards = []
    a = np.zeros(11, dtype=np.float32)
    a[axis] = value
    if axis != 10:
        a[10] = o0[135]
    for _ in range(steps):
        o, r, term, trunc, info = env.step(a)
        total += r
        rewards.append(round(float(r), 4))
        if term or trunc:
            break
    inds = list(range(125, 136)) + [147, 148, 149, 150, 151, 152, 153]
    print("axis", axis, "val", value, "n", len(rewards), "rew", total,
          "rseq", rewards, "robot_delta", np.round(o[125:136]-o0[125:136], 4).tolist(),
          "wiper_delta", np.round(o[147:154]-o0[147:154], 4).tolist(),
          "cube_delta", np.round(o[:80].reshape(5,16)[:,:3]-o0[:80].reshape(5,16)[:,:3],3).tolist(),
          "info", info)
    env.close()


for axis in range(11):
    run(axis, 0.1 if axis < 10 else 1.0)
for axis in range(10):
    run(axis, -0.1)
