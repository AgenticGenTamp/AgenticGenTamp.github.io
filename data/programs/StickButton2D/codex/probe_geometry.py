"""Small black-box probes for StickButton2DEnv (never imported by approach.py)."""

import argparse
import math

import numpy as np

from env_client import make_env


def snapshot(state):
    rows = []
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        vals = {}
        for feature in (
            "x", "y", "theta", "radius", "width", "height", "base_radius",
            "arm_joint", "arm_length", "vacuum", "gripper_height", "gripper_width",
            "static", "color_r", "color_g", "color_b",
        ):
            try:
                vals[feature] = float(state.get(obj, feature))
            except Exception:
                pass
        rows.append((name, vals))
    return rows


def print_snapshot(step, state, reward=None, terminated=None):
    print("STEP", step, "reward", reward, "terminated", terminated)
    for name, vals in snapshot(state):
        print(name, " ".join(f"{k}={v:.5f}" for k, v in vals.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actions", default="")
    args = parser.parse_args()
    env = make_env()
    state, info = env.reset(seed=args.seed)
    print("max_steps", env.max_steps, "info", info)
    print_snapshot(0, state)
    # Format: dx,dy,dtheta,darm,vac;... repeated.
    for i, token in enumerate(filter(None, args.actions.split(";")), 1):
        action = np.asarray([float(x) for x in token.split(",")], dtype=np.float32)
        state, reward, terminated, truncated, info = env.step(action)
        print_snapshot(i, state, reward, terminated)
        print("truncated", truncated, "info", info)
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
