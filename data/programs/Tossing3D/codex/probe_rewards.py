"""Small black-box probes for Tossing3DEnv metadata, layouts, and rewards."""

import numpy as np
import sys

from env_client import make_env


def scalar(state, obj, feature):
    return float(state.get(obj, feature))


def summarize(seed, steps=0, action_kind="hold"):
    env = make_env()
    try:
        obs, info = env.reset(seed=seed)
        movable = obs.get_objects(env.observation_space.get_type("mujoco_movable_object"))
        rows = []
        for obj in movable:
            name = getattr(obj, "name", str(obj))
            rows.append(
                (
                    name,
                    tuple(round(scalar(obs, obj, q), 4) for q in ("x", "y", "z")),
                    tuple(round(scalar(obs, obj, q), 4) for q in ("bb_x", "bb_y", "bb_z")),
                )
            )
        print("RESET", seed, "max_steps", env.max_steps, "info", info)
        print(" action", env.action_space.shape, env.action_space.dtype)
        print(" low", np.asarray(env.action_space.low).round(3).tolist())
        print(" high", np.asarray(env.action_space.high).round(3).tolist())
        print(" objects", obs.get_object_names())
        print(" movable", rows)
        robot = obs.get_object_from_name("robot")
        robot_fields = [
            "pos_base_x", "pos_base_y", "pos_base_rot", *
            [f"pos_arm_joint{i}" for i in range(1, 8)], "pos_gripper"
        ]
        robot_q = np.array([scalar(obs, robot, q) for q in robot_fields])
        print(" robot", np.round(robot_q, 4).tolist())
        rewards = []
        for i in range(steps):
            if action_kind == "zero":
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            elif action_kind == "hold":
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                action[:3] = robot_q[:3]
                action[3:10] = robot_q[3:10]
                action[10] = robot_q[10]
            elif action_kind == "random":
                action = env.action_space.sample()
            else:
                raise ValueError(action_kind)
            obs, reward, terminated, truncated, step_info = env.step(action)
            rewards.append(float(reward))
            if i < 3 or terminated or truncated or i + 1 == steps:
                print(" step", i + 1, "r", reward, "term", terminated,
                      "trunc", truncated, "info", step_info)
            if terminated or truncated:
                break
        if rewards:
            print(" rewards unique", sorted(set(round(x, 6) for x in rewards)),
                  "sum", round(sum(rewards), 6), "n", len(rewards))
    finally:
        env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "counts":
        env = make_env()
        try:
            for count in (1, 2):
              for count_seed in range(5):
                obs, info = env.reset(seed=count_seed, options={"object_count": count})
                positions = []
                for name in sorted(obs.get_object_names()):
                    if name.startswith("cube_"):
                        obj = obs.get_object_from_name(name)
                        positions.append((name,) + tuple(round(scalar(obs, obj, f), 3)
                                                         for f in ("x", "y", "z")))
                bin_obj = obs.get_object_from_name("bin_0")
                barrier = obs.get_object_from_name("cuboid_barrier")
                print("COUNT", count, "seed", count_seed, info, positions,
                      "bin", tuple(round(scalar(obs, bin_obj, f), 3) for f in ("x", "y", "z")),
                      "barrier", tuple(round(scalar(obs, barrier, f), 3) for f in ("x", "y", "z")))
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                _, reward, terminated, truncated, _ = env.step(action)
                print(" COUNT_REWARD", reward, terminated, truncated)
        finally:
            env.close()
    elif len(sys.argv) > 1 and sys.argv[1] == "limit":
        env = make_env()
        try:
            obs, info = env.reset(seed=0)
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            for i in range(env.max_steps + 1):
                obs, reward, terminated, truncated, info = env.step(action)
                if i in (0, env.max_steps - 2, env.max_steps - 1) or terminated or truncated:
                    print("LIMIT", i + 1, reward, terminated, truncated, info)
                if terminated or truncated:
                    break
        finally:
            env.close()
    else:
        for probe_seed in range(8):
            summarize(probe_seed)
        summarize(0, 8, "hold")
        summarize(0, 8, "zero")
