"""Print compact reset-state summaries for several PR2Blocked seeds."""

import argparse
import math
from env_client import make_env


def vals(state, obj, features):
    return [float(state.get(obj, feature)) for feature in features]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    env = make_env()
    robot_features = [
        "base_x", "base_y", "base_rot",
        "joint_1", "joint_2", "joint_3", "joint_4",
        "joint_5", "joint_6", "joint_7", "gripper_opening",
        "grasp_active", "grasp_tf_x", "grasp_tf_y", "grasp_tf_z",
        "grasp_tf_qx", "grasp_tf_qy", "grasp_tf_qz", "grasp_tf_qw",
    ]
    pose_features = ["pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw"]
    extent_features = ["half_extent_x", "half_extent_y", "half_extent_z"]
    records = []
    for seed in range(args.seeds):
        state, info = env.reset(seed=seed)
        names = state.get_object_names()
        green0 = state.get_object_from_name("green0")
        blocker = state.get_object_from_name("blocker")
        gx, gy = vals(state, green0, ["pose_x", "pose_y"])
        bx, by = vals(state, blocker, ["pose_x", "pose_y"])
        qz, qw = vals(state, green0, ["pose_qz", "pose_qw"])
        yaw = 2.0 * math.atan2(qz, qw)
        spares = []
        for name in names:
            if name.startswith("green") and name != "green0":
                obj = state.get_object_from_name(name)
                spares.append(vals(state, obj, ["pose_x", "pose_y", "pose_z"]))
        records.append((int(info["object_count"]), gx, gy, bx, by, yaw, spares))
        if args.summary:
            continue
        print(f"SEED {seed} info={info} names={names}")
        rob = state.get_object_from_name("robot")
        print(" robot", " ".join(f"{x:.6f}" for x in vals(state, rob, robot_features)))
        for name in names:
            if name == "robot":
                continue
            obj = state.get_object_from_name(name)
            extra = extent_features
            if name.startswith("green") or name == "blocker":
                extra = ["grasp_active"] + extent_features
            print(f" {name}", " ".join(f"{x:.6f}" for x in vals(state, obj, pose_features + extra)))
    if args.summary:
        counts = {}
        spare_xyz = []
        errors = []
        for count, gx, gy, bx, by, yaw, spares in records:
            counts[count] = counts.get(count, 0) + 1
            spare_xyz.extend(spares)
            # The blocker is 0.15 m from green0 along negative local x.
            errors.append(math.hypot(bx - gx + 0.15 * math.cos(yaw),
                                     by - gy + 0.15 * math.sin(yaw)))
        print("count histogram", counts)
        print("green0 x/y ranges",
              (min(r[1] for r in records), max(r[1] for r in records)),
              (min(r[2] for r in records), max(r[2] for r in records)))
        print("blocker x/y ranges",
              (min(r[3] for r in records), max(r[3] for r in records)),
              (min(r[4] for r in records), max(r[4] for r in records)))
        print("pen yaw range", (min(r[5] for r in records), max(r[5] for r in records)))
        print("blocker geometry max error", max(errors))
        if spare_xyz:
            print("spare x/y/z ranges",
                  (min(p[0] for p in spare_xyz), max(p[0] for p in spare_xyz)),
                  (min(p[1] for p in spare_xyz), max(p[1] for p in spare_xyz)),
                  (min(p[2] for p in spare_xyz), max(p[2] for p in spare_xyz)))
    env.close()


if __name__ == "__main__":
    main()
