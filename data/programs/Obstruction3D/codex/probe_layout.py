"""Probe reset-only geometry for Obstruction3DEnv.

This is intentionally separate from approach.py and uses only the public black-box
client API.  Run with /opt/robocode-strict/bin/python probe_layout.py.
"""

from collections import Counter

import numpy as np

from env_client import make_env


FEATURES = (
    "pose_x", "pose_y", "pose_z", "half_extent_x", "half_extent_y",
    "half_extent_z", "object_type", "grasp_active",
)


def read(state, name):
    obj = state.get_object_from_name(name)
    return {feature: state.get(obj, feature) for feature in FEATURES}


def main():
    env = make_env()
    print("max_steps", env.max_steps)
    print("action", env.action_space.shape, env.action_space.low.tolist(), env.action_space.high.tolist())
    rows = []
    for seed in range(100):
        state, info = env.reset(seed=seed)
        names = sorted(state.get_object_names())
        obs_names = [name for name in names if name.startswith("obstruction")]
        target = read(state, "target_region")
        block = read(state, "target_block")
        robot = state.get_object_from_name("robot")
        rob = {f: state.get(robot, f) for f in (
            "pos_base_x", "pos_base_y", "pos_base_rot",
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
            "joint_6", "joint_7", "finger_state", "grasp_active",
        )}
        obstruction_rows = []
        for name in obs_names:
            ob = read(state, name)
            overlap = (
                abs(ob["pose_x"] - target["pose_x"]) <= ob["half_extent_x"] + target["half_extent_x"]
                and abs(ob["pose_y"] - target["pose_y"]) <= ob["half_extent_y"] + target["half_extent_y"]
            )
            obstruction_rows.append((name, ob, overlap))
        rows.append((seed, target, block, rob, obstruction_rows, info))

    print("count_hist", dict(sorted(Counter(len(row[4]) for row in rows).items())))
    for label, getter in (
        ("target xyz", lambda r: [r[1]["pose_x"], r[1]["pose_y"], r[1]["pose_z"]]),
        ("target half", lambda r: [r[1]["half_extent_x"], r[1]["half_extent_y"], r[1]["half_extent_z"]]),
        ("block xyz", lambda r: [r[2]["pose_x"], r[2]["pose_y"], r[2]["pose_z"]]),
        ("block half", lambda r: [r[2]["half_extent_x"], r[2]["half_extent_y"], r[2]["half_extent_z"]]),
        ("robot base", lambda r: [r[3]["pos_base_x"], r[3]["pos_base_y"], r[3]["pos_base_rot"]]),
        ("robot joints", lambda r: [r[3][f"joint_{i}"] for i in range(1, 8)]),
    ):
        arr = np.asarray([getter(row) for row in rows])
        print(label, "min", np.round(arr.min(axis=0), 5), "max", np.round(arr.max(axis=0), 5),
              "unique", [len(np.unique(arr[:, i])) for i in range(arr.shape[1])])
    block_offset = np.asarray([
        [r[2]["pose_x"] - r[1]["pose_x"], r[2]["pose_y"] - r[1]["pose_y"],
         r[2]["pose_z"] - r[1]["pose_z"]] for r in rows
    ])
    half_ratio = np.asarray([
        [r[2]["half_extent_x"] / r[1]["half_extent_x"],
         r[2]["half_extent_y"] / r[1]["half_extent_y"]] for r in rows
    ])
    print("block-target offset min/max", np.round(block_offset.min(axis=0), 5),
          np.round(block_offset.max(axis=0), 5))
    print("block/region xy half ratio", np.round(half_ratio.min(axis=0), 5),
          np.round(half_ratio.max(axis=0), 5))

    all_obs = [(r, name, ob, overlap) for r in rows for name, ob, overlap in r[4]]
    obs_xyz = np.asarray([[ob["pose_x"], ob["pose_y"], ob["pose_z"]] for _, _, ob, _ in all_obs])
    obs_half = np.asarray([[ob["half_extent_x"], ob["half_extent_y"], ob["half_extent_z"]] for _, _, ob, _ in all_obs])
    print("obstruction xyz min/max", np.round(obs_xyz.min(axis=0), 5), np.round(obs_xyz.max(axis=0), 5))
    print("obstruction half min/max", np.round(obs_half.min(axis=0), 5), np.round(obs_half.max(axis=0), 5))
    print("obstruction target-overlap", sum(x[3] for x in all_obs), "/", len(all_obs))
    print("types", sorted({(name, data["object_type"]) for r in rows for name, data in
                           [("region", r[1]), ("block", r[2])] } |
                          {("obstruction", ob["object_type"]) for r in rows for _, ob, _ in r[4]}))
    print("robot unique", {key: sorted({round(row[3][key], 6) for row in rows})
                           for key in rows[0][3]})
    print("samples")
    for seed, target, block, rob, obstructions, info in rows[:12]:
        obs_text = [
            (name, tuple(round(ob[k], 4) for k in ("pose_x", "pose_y", "pose_z")),
             tuple(round(ob[k], 4) for k in ("half_extent_x", "half_extent_y", "half_extent_z")), overlap)
            for name, ob, overlap in obstructions
        ]
        print(seed, "target", tuple(round(target[k], 4) for k in ("pose_x", "pose_y", "pose_z")),
              "block", tuple(round(block[k], 4) for k in ("pose_x", "pose_y", "pose_z")), "obs", obs_text,
              "info", info)
    env.close()


if __name__ == "__main__":
    main()
