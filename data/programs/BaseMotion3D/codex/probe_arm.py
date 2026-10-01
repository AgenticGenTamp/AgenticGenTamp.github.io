import numpy as np
import sys
from env_client import make_env


def compact(x):
    return np.array2string(np.asarray(x), precision=3, suppress_small=True)


def initial_seeds():
    for seed in range(5):
        env = make_env()
        obs, info = env.reset(seed=seed)
        print("RESET", seed, "base", compact(obs[:3]), "joints", compact(obs[3:10]),
              "finger/grasp", compact(obs[10:12]), "tf", compact(obs[12:19]),
              "target", compact(obs[19:22]), "info", info)
        env.close()


def one_step_deltas(seed=0):
    env0 = make_env()
    base, _ = env0.reset(seed=seed)
    env0.close()
    for idx in range(11):
        for value in ([0.4] if idx < 10 else [-1.0, 1.0]):
            env = make_env()
            obs, _ = env.reset(seed=seed)
            a = np.zeros(11, np.float32)
            a[idx] = value
            nxt, rew, term, trunc, info = env.step(a)
            print("STEP", idx, value, "delta", compact(nxt-base), "rtt", rew, term, trunc, info)
            env.close()


def navigate(seed):
    env = make_env()
    obs, _ = env.reset(seed=seed)
    print("NAV reset", seed, compact(obs))
    for step in range(20):
        delta = obs[19:21] - obs[:2]
        a = np.zeros(11, np.float32)
        a[:2] = np.clip(delta, -0.4, 0.4)
        obs, rew, term, trunc, info = env.step(a)
        print("NAV", step + 1, "a", compact(a), "base,target,dist", compact(obs[:2]), compact(obs[19:21]), np.linalg.norm(obs[19:21]-obs[:2]), rew, term, trunc, info)
        if term or trunc:
            break
    env.close()


def gripper_sequence(seed=0):
    env = make_env()
    obs, _ = env.reset(seed=seed)
    for value in (-1, -1, 0, 1, 1, 0):
        a = np.zeros(11, np.float32)
        a[10] = value
        obs, rew, term, trunc, info = env.step(a)
        print("GRIP", value, "finger/grasp/tf", compact(obs[10:19]), rew, term, trunc, info)
    env.close()


def threshold(seed=0):
    env = make_env()
    for residual in (0.2, 0.15, 0.11, 0.101, 0.1, 0.099, 0.075, 0.051, 0.05, 0.049, 0.01):
        obs, _ = env.reset(seed=seed)
        target = obs[19:21].copy()
        term = trunc = False
        for _ in range(10):
            desired = target.copy()
            desired[0] -= residual
            a = np.zeros(11, np.float32)
            a[:2] = np.clip(desired - obs[:2], -0.4, 0.4)
            obs, rew, term, trunc, info = env.step(a)
            if term or trunc or np.allclose(obs[:2], desired, atol=1e-6):
                break
        print("THRESH", residual, "pos", compact(obs[:2]), "diff", compact(target-obs[:2]), "term", term, "steps", _+1)
    env.close()


def threshold_2d(seed=0):
    env = make_env()
    for rx, ry in ((.04,.04),(.03,.04),(.029,.04),(.049,0),(.035,.035),(.036,.036)):
        obs, _ = env.reset(seed=seed)
        target = obs[19:21].copy()
        desired = target - np.array([rx, ry], np.float32)
        for step in range(10):
            a = np.zeros(11, np.float32)
            a[:2] = np.clip(desired - obs[:2], -.4, .4)
            obs, rew, term, trunc, info = env.step(a)
            if term or trunc or np.allclose(obs[:2], desired, atol=1e-6):
                break
        print("THRESH2", rx, ry, "norm", np.hypot(rx,ry), "actual", compact(target-obs[:2]), "term", term)
    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "navigate":
        for seed in map(int, sys.argv[2:] or [0]):
            navigate(seed)
    elif mode == "gripper":
        gripper_sequence(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif mode == "threshold":
        threshold(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif mode == "threshold2":
        threshold_2d(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        initial_seeds()
        one_step_deltas()
