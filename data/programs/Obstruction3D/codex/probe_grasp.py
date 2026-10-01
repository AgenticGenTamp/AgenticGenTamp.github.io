"""Empirical probes for Obstruction3D gripper/attachment behavior."""

import argparse
import numpy as np

from env_client import make_env


def val(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def snapshot(state):
    out = {}
    for name in state.get_object_names():
        if name == "robot":
            fs = ["pos_base_x", "pos_base_y", "pos_base_rot"]
            fs += [f"joint_{i}" for i in range(1, 8)]
            fs += ["finger_state", "grasp_active", "grasp_tf_x", "grasp_tf_y", "grasp_tf_z"]
        else:
            fs = ["pose_x", "pose_y", "pose_z", "grasp_active", "object_type",
                  "half_extent_x", "half_extent_y", "half_extent_z"]
        out[name] = {f: val(state, name, f) for f in fs}
    return out


def summarize(state):
    s = snapshot(state)
    r = s.pop("robot")
    print("robot base", [round(r[x], 4) for x in ("pos_base_x", "pos_base_y", "pos_base_rot")],
          "joints", [round(r[f"joint_{i}"], 4) for i in range(1, 8)],
          "finger/grasp", round(r["finger_state"], 4), round(r["grasp_active"], 4),
          "tf", [round(r[x], 4) for x in ("grasp_tf_x", "grasp_tf_y", "grasp_tf_z")])
    for name, d in s.items():
        print(name, "pos", [round(d[x], 4) for x in ("pose_x", "pose_y", "pose_z")],
              "half", [round(d[x], 4) for x in ("half_extent_x", "half_extent_y", "half_extent_z")],
              "type/grasp", round(d["object_type"], 3), round(d["grasp_active"], 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("idle", "random", "scan"), default="idle")
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    env = make_env()
    state, info = env.reset(seed=args.seed)
    print("INITIAL", info)
    summarize(state)
    rng = np.random.default_rng(args.seed + 12345)
    for t in range(args.steps):
        a = np.zeros(11, dtype=np.float32)
        if args.mode == "random":
            a[:10] = rng.uniform(-0.2, 0.2, size=10)
        elif args.mode == "scan":
            # Serpentine grid of base positions, retaining the reset arm pose.
            # One 0.2m move gets to a new row/column; interpolation is performed
            # by the server's relative-action dynamics.
            row = t // 8
            col = t % 8
            target_x = -0.25 + 0.1 * (col if row % 2 == 0 else 7 - col)
            target_y = -0.40 + 0.1 * row
            a[0] = np.clip(target_x - val(state, "robot", "pos_base_x"), -0.2, 0.2)
            a[1] = np.clip(target_y - val(state, "robot", "pos_base_y"), -0.2, 0.2)
        a[10] = -1 if args.mode == "scan" or t < args.steps // 2 else 1
        state, reward, term, trunc, info = env.step(a)
        print("STEP", t + 1, "action", np.round(a, 3).tolist(), "r/done", reward, term, trunc)
        summarize(state)
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    main()
