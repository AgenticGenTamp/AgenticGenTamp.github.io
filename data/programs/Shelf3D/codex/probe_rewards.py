"""Empirical metadata/reward probes for Shelf3DEnv."""

import argparse
import json
import numpy as np

from env_client import make_env


def state_dict(state):
    out = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        vals = {}
        # Query only known relevant features; unsupported accesses can fail remotely.
        for feature in (
            "x", "y", "z", "qw", "qx", "qy", "qz",
            "vx", "vy", "vz", "bb_x", "bb_y", "bb_z",
            "pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper",
        ):
            try:
                vals[feature] = float(state.get(obj, feature))
            except Exception:
                pass
        out[name] = vals
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--seeds", type=int, default=10)
parser.add_argument("--steps", type=int, default=5)
args = parser.parse_args()

for seed in range(args.start, args.start + args.seeds):
    env = make_env()
    state, info = env.reset(seed=seed)
    sd = state_dict(state)
    cubes = sorted(n for n in sd if n.startswith("cube"))
    print("RESET", seed, "max", env.max_steps, "names", list(sd),
          "count", len(cubes), "info", json.dumps(info, default=str),
          "objects", json.dumps(sd, sort_keys=True))
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    action[-1] = 1.0
    rewards = []
    last = None
    for step in range(args.steps):
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        last = (terminated, truncated, info)
        if terminated or truncated:
            break
    print("STEPS", seed, "n", len(rewards), "first", rewards[:5],
          "last_rewards", rewards[-5:], "last", json.dumps(last, default=str))
    env.close()
