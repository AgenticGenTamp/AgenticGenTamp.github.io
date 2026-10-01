"""Focused black-box probes for ScoopPour3D action/control semantics."""

import argparse
import numpy as np

from env_client import make_env


def val(state, obj, feat):
    return float(state.get(obj, feat))


def snapshot(state):
    robot = state.get_object_from_name("robot")
    names = state.get_object_names()
    mov = []
    for name in names:
        if name.startswith(("bin_", "scoop_", "cube_")):
            obj = state.get_object_from_name(name)
            mov.append((name, *(val(state, obj, f) for f in ("x", "y", "z"))))
    r = tuple(val(state, robot, f) for f in (
        "pos_base_x", "pos_base_y", "pos_base_rot",
        "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3", "pos_arm_joint4",
        "pos_arm_joint5", "pos_arm_joint6", "pos_arm_joint7", "pos_gripper"))
    return np.array(r), mov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="info", choices=("info", "impulse", "hold"))
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--value", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    env = make_env()
    state, info = env.reset(seed=args.seed)
    r0, m0 = snapshot(state)
    print("info", info)
    print("robot0", np.round(r0, 5).tolist())
    print("objects0", [(n, round(x, 4), round(y, 4), round(z, 4)) for n, x, y, z in m0])
    if args.mode != "info":
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[args.dim] = args.value
        if args.mode == "impulse":
            actions = [action] + [np.zeros_like(action)] * (args.steps - 1)
        else:
            actions = [action] * args.steps
        total = 0.0
        for i, a in enumerate(actions):
            state, rew, term, trunc, info = env.step(a)
            total += rew
            r, mov = snapshot(state)
            if i < 4 or i == args.steps - 1:
                print("step", i + 1, "rew", round(rew, 5), "robot", np.round(r, 5).tolist())
        print("total", total, "term", term, "trunc", trunc)
        print("objectsN", [(n, round(x, 4), round(y, 4), round(z, 4)) for n, x, y, z in mov])
    env.close()


if __name__ == "__main__":
    main()
