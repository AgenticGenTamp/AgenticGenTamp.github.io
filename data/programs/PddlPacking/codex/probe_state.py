"""Bounded reset-state probe for the black-box PR2Packed environment."""

from collections import Counter

import numpy as np

from env_client import make_env


def values(state, obj, names):
    return np.array([state.get(obj, name) for name in names], dtype=float)


def main():
    env = make_env()
    print("max_steps", env.max_steps)
    print("action", env.action_space.shape, env.action_space.low, env.action_space.high)
    types = {typ.name: typ for typ in env.observation_space.types}
    rows = []
    robots = []
    tables = []
    plates = []
    blocks = []
    infos = []
    try:
        for seed in range(30):
            state, info = env.reset(seed=seed)
            robot = state.get_objects(types["robot"])[0]
            surfaces = {obj.name: obj for obj in state.get_objects(types["surface"])}
            block_objs = state.get_objects(types["block"])
            robot_vec = values(
                state,
                robot,
                [
                    "base_x", "base_y", "base_rot", "joint_1", "joint_2",
                    "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
                    "gripper_opening", "grasp_active",
                ],
            )
            table_vec = values(
                state,
                surfaces["table"],
                ["pose_x", "pose_y", "pose_z", "half_extent_x", "half_extent_y", "half_extent_z"],
            )
            plate_vec = values(
                state,
                surfaces["plate"],
                ["pose_x", "pose_y", "pose_z", "half_extent_x", "half_extent_y", "half_extent_z"],
            )
            block_vecs = [
                values(
                    state,
                    obj,
                    ["pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw", "grasp_active", "half_extent_x", "half_extent_y", "half_extent_z"],
                )
                for obj in block_objs
            ]
            robots.append(robot_vec)
            tables.append(table_vec)
            plates.append(plate_vec)
            blocks.extend(block_vecs)
            infos.append(repr(info))
            rows.append((seed, len(block_objs), robot_vec[:3], plate_vec[:3]))
        for row in rows:
            print("seed", row[0], "n", row[1], "base", row[2], "plate", row[3])
        for label, arr in (
            ("robot", np.array(robots)),
            ("table", np.array(tables)),
            ("plate", np.array(plates)),
            ("block", np.array(blocks)),
        ):
            print(label, "min", arr.min(axis=0), "max", arr.max(axis=0))
            print(label, "unique_rows", len(np.unique(np.round(arr, 7), axis=0)))
        print("count_hist", Counter(row[1] for row in rows))
        print("info_unique", sorted(set(infos)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
