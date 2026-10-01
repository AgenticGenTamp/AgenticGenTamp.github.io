import numpy as np
from env_client import make_env


def trial(seed, offset):
    env = make_env(); obs, _ = env.reset(seed=seed)
    target = obs[19:21].copy()
    steps = 0
    while np.any(np.abs(target - np.asarray(offset) - obs[:2]) > .400001):
        a = np.zeros(11, np.float32)
        a[:2] = np.clip(target - np.asarray(offset) - obs[:2], -.4, .4)
        obs, _, term, trunc, _ = env.step(a); steps += 1
        if term or trunc: break
    if not (term or trunc):
        a = np.zeros(11, np.float32)
        a[:2] = np.clip(target - np.asarray(offset) - obs[:2], -.4, .4)
        obs, _, term, trunc, _ = env.step(a); steps += 1
    env.close()
    return offset, term, steps, float(np.linalg.norm(obs[:2]-target))


if __name__ == '__main__':
    for off in [(x, 0) for x in (.2,.15,.11,.1,.09,.05,0)]:
        print(trial(0, off), flush=True)
