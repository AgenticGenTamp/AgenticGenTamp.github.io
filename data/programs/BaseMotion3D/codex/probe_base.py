import numpy as np

from env_client import make_env


def brief(obs):
    return np.round(np.asarray(obs)[[0, 1, 2, 19, 20, 21]], 4).tolist()


def reset_survey():
    env = make_env()
    print("max_steps", env.max_steps)
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        print("R", seed, brief(obs), info)
    env.close()


def impulse_survey():
    tests = [
        ("+x", [0.4, 0, 0]), ("-x", [-0.4, 0, 0]),
        ("+y", [0, 0.4, 0]), ("-y", [0, -0.4, 0]),
        ("+r", [0, 0, 0.4]), ("-r", [0, 0, -0.4]),
    ]
    for seed in [0, 1]:
        for name, vals in tests:
            env = make_env()
            obs, _ = env.reset(seed=seed)
            before = np.asarray(obs).copy()
            act = np.zeros(11, dtype=np.float32)
            act[:3] = vals
            obs, reward, term, trunc, info = env.step(act)
            delta = np.asarray(obs)[:3] - before[:3]
            print("I", seed, name, "pre", brief(before), "delta", np.round(delta, 5).tolist(),
                  "rtt", reward, term, trunc, info)
            env.close()


def greedy_survey():
    for seed in range(20):
        env = make_env()
        obs, _ = env.reset(seed=seed)
        initial = brief(obs)
        done = False
        for step in range(20):
            act = np.zeros(11, dtype=np.float32)
            # Test whether commanded base displacement is in world coordinates.
            act[0] = np.clip(obs[19] - obs[0], -0.4, 0.4)
            act[1] = np.clip(obs[20] - obs[1], -0.4, 0.4)
            obs, reward, term, trunc, info = env.step(act)
            if term or trunc:
                done = True
                break
        dist = float(np.linalg.norm(np.asarray(obs)[19:21] - np.asarray(obs)[:2]))
        print("G", seed, "init", initial, "steps", step + 1, "done", done,
              "final", brief(obs), "xy_dist", round(dist, 5), "info", info)
        env.close()


def threshold_survey():
    # Seed 16's target is within one base action of the origin, allowing exact
    # placement at controlled offsets on the very first transition.
    env = make_env()
    for dx, dy in [(0, 0), (0.049, 0), (0.05, 0), (0.051, 0),
                   (0.035, 0.035), (0.036, 0.036), (0.04, 0.04),
                   (-0.049, 0), (0, -0.049)]:
        obs, _ = env.reset(seed=16)
        act = np.zeros(11, dtype=np.float32)
        act[0] = obs[19] - obs[0] + dx
        act[1] = obs[20] - obs[1] + dy
        obs, reward, term, trunc, info = env.step(act)
        d = float(np.linalg.norm(np.asarray(obs)[19:21] - np.asarray(obs)[:2]))
        print("T", dx, dy, "actual_dist", round(d, 7), "term", term,
              "pos", np.round(obs[:2], 6).tolist())
    env.close()


def rotated_frame_survey():
    env = make_env()
    obs, _ = env.reset(seed=0)
    for vals in [(0, 0, 0.4), (0, 0, 0.4), (0, 0, 0.4),
                 (0.4, 0, 0), (0, 0.4, 0), (0.4, -0.4, -0.4)]:
        before = np.asarray(obs).copy()
        act = np.zeros(11, dtype=np.float32)
        act[:3] = vals
        obs, reward, term, trunc, info = env.step(act)
        print("F", vals, "before", np.round(before[:3], 5).tolist(),
              "after", np.round(obs[:3], 5).tolist(), "term", term)
    env.close()


if __name__ == "__main__":
    reset_survey()
    impulse_survey()
    greedy_survey()
