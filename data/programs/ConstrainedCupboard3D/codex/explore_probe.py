"""Compact geometry/reward probes for the remote cupboard environment."""

import argparse
import numpy as np

from env_client import make_env


def val(state, obj, feature):
    return float(state.get(obj, feature))


def summarize(seed, count):
    env = make_env()
    try:
        state, info = env.reset(seed=seed, options={"object_count": count})
        print("SEED", seed, "COUNT", count, "INFO", info)
        for type_name in ("mujoco_fixture", "mujoco_movable_object", "mujoco_tidybot_robot"):
            objects = state.get_objects(env.observation_space.get_type(type_name))
            for obj in objects:
                if type_name == "mujoco_fixture":
                    fs = ("x", "y", "z", "qw", "qx", "qy", "qz")
                elif type_name == "mujoco_movable_object":
                    fs = ("x", "y", "z", "qw", "qx", "qy", "qz", "bb_x", "bb_y", "bb_z")
                else:
                    fs = ("pos_base_x", "pos_base_y", "pos_base_rot", "pos_gripper")
                print(obj.name, *(f"{f}={val(state,obj,f):.5f}" for f in fs))
        for label, action in (
            ("zero_open", np.r_[np.zeros(10), 1.0]),
            ("zero_closed", np.zeros(11)),
        ):
            _, rew, term, trunc, step_info = env.step(action)
            print("STEP", label, rew, term, trunc, step_info)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    summarize(args.seed, args.count)
