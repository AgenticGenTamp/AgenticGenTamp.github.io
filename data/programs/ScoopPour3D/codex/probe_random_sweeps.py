"""Search coarse constant arm motions for contacts with scoop, bins, or cubes."""

import argparse
import numpy as np
from env_client import make_env


def xyz(s, name):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in ("x", "y", "z")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    env = make_env()
    for trial in range(args.trials):
        s, _ = env.reset(seed=0)
        names = sorted(n for n in s.get_object_names() if n.startswith("cube_"))
        before = {n: xyz(s, n) for n in names + ["scoop_0", "bin_green_0", "bin_yellow_0"]}
        a = np.zeros(11, np.float32)
        # Coarse velocity direction, biased toward proximal joints.
        a[3:10] = rng.choice([-0.1, 0.0, 0.1], 7, p=[0.35, 0.3, 0.35])
        a[10] = float(rng.integers(2))
        best_reward = -1.0
        done = False
        for _ in range(args.steps):
            s, reward, term, trunc, _ = env.step(a)
            best_reward = max(best_reward, reward)
            if term or trunc:
                done = True
                break
        delta = {n: float(np.linalg.norm(xyz(s, n) - p)) for n, p in before.items()}
        max_cube = max(delta[n] for n in names)
        if max(max_cube, delta["scoop_0"], delta["bin_green_0"],
               delta["bin_yellow_0"]) > 0.002 or best_reward != -1.0 or done:
            print(trial, "a", a.tolist(), "scoop", round(delta["scoop_0"], 4),
                  "green", round(delta["bin_green_0"], 4),
                  "yellow", round(delta["bin_yellow_0"], 4),
                  "cube", round(max_cube, 4), "reward", best_reward, "done", done)
    env.close()


if __name__ == "__main__":
    main()
