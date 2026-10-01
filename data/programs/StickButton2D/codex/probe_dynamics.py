"""Small black-box probes for StickButton2DEnv observations and dynamics."""

import argparse
import math

import numpy as np

from env_client import make_env


def snapshot(state, env):
    out = {}
    for typ in env.observation_space.types:
        for obj in state.get_objects(typ):
            # Skip inherited/base-type views, which otherwise overwrite the
            # richer concrete object entry (e.g. circle with kinematic2d).
            if str(obj.type) != str(typ):
                continue
            features = env.observation_space.type_features[typ]
            out[obj.name] = {feature: float(state.get(obj, feature)) for feature in features}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--action", type=float, nargs=5, default=[0, 0, 0, 0, 0])
    args = parser.parse_args()

    env = make_env()
    reset_options = {} if args.count is None else {"object_count": args.count}
    state, info = env.reset(seed=args.seed, options=reset_options)
    print("max_steps", env.max_steps, "info", info)
    print("initial", snapshot(state, env))
    action = np.asarray(args.action, dtype=np.float32)
    for step in range(args.steps):
        state, reward, terminated, truncated, info = env.step(action)
        print(step + 1, reward, terminated, truncated, info, snapshot(state, env))
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
