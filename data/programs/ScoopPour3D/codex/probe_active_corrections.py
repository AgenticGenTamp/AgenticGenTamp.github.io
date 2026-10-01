"""Compare small corrections applied while the seed-1 sweep is active."""

import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def xy(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y")], float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start", type=int, default=194)
    parser.add_argument("--stop", type=int, default=270)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--yscale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=270)
    args = parser.parse_args()
    env = make_env()
    state, info = env.reset(seed=args.seed, options={"object_count": args.count})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube_"))
    initial = np.array([xy(state, n) for n in names])
    source = xy(state, "bin_yellow_0")
    target = xy(state, "bin_green_0")
    desired = initial + target - source
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    for step in range(1, args.steps + 1):
        action = policy.get_action(state)
        if args.start <= step <= args.stop:
            action[0] = np.clip(action[0] + args.x, -.1, .1)
            action[1] = np.clip(action[1] * args.yscale, -.1, .1)
            action[2] = np.clip(action[2] + args.yaw, -.1, .1)
        state, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    final = np.array([xy(state, n) for n in names])
    delta = final - desired
    error = np.linalg.norm(delta, axis=1)
    print("x", args.x, "yaw", args.yaw, "yscale", args.yscale,
          "window", args.start, args.stop,
          "mean_delta", np.round(delta.mean(0), 4),
          "errors", np.round([error.min(), error.mean(), error.max()], 4),
          "within", int((error < .05).sum()), "reward", reward,
          "done", term)
    env.close()


if __name__ == "__main__":
    main()
