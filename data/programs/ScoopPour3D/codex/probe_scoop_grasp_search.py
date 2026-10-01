"""Probe a true scoop grasp, including lift and lateral follow tests.

This is deliberately separate from approach.py.  Offsets are expressed in
the scoop's observed long/short axes, so the same trial can be used on seeds
with different scoop yaw.
"""

import argparse
import math

import numpy as np

from env_client import make_env


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([state.get(obj, f) for f in features], dtype=float)


def scoop_pose(state):
    return read(state, "scoop_0", ("x", "y", "z", "qw", "qx", "qy", "qz"))


def robot_pose(state):
    fs = ("pos_base_x", "pos_base_y", "pos_base_rot")
    fs += tuple("pos_arm_joint%d" % i for i in range(1, 8))
    fs += ("pos_gripper",)
    return read(state, "robot", fs)


def wrap(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def step(env, state, action, count, samples, label):
    for _ in range(count):
        state, reward, term, trunc, _ = env.step(action)
        samples.append((label, scoop_pose(state).copy(), robot_pose(state).copy()))
        if term or trunc:
            break
    return state


def trial(args):
    env = make_env()
    state, _ = env.reset(seed=args.seed)
    initial_scoop = scoop_pose(state)
    initial_robot = robot_pose(state)
    q = initial_scoop[3:]
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                     1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
    long_axis = np.array([math.cos(yaw), math.sin(yaw)])
    short_axis = np.array([-long_axis[1], long_axis[0]])
    xy_offset = args.long * long_axis + args.short * short_axis
    if args.canonical:
        desired_base = initial_scoop[:2] - np.array([0.417, 0.0]) + xy_offset
        desired_yaw = math.atan2(initial_scoop[1] - desired_base[1],
                                 initial_scoop[0] - desired_base[0])
        relative_yaw = wrap(yaw - desired_yaw)
        wrist_delta = wrap(relative_yaw + 0.557)
        if wrist_delta > math.pi / 2.0:
            wrist_delta -= math.pi
        elif wrist_delta < -math.pi / 2.0:
            wrist_delta += math.pi
    else:
        desired_base = initial_robot[:2] + xy_offset
        desired_yaw = initial_robot[2]
        wrist_delta = 0.0
    desired_wrist = initial_robot[9] + wrist_delta + args.wrist
    samples = []

    # Fully open before contact, while translating to the selected scoop end.
    for _ in range(args.align_steps):
        rp = robot_pose(state)
        action = np.zeros(11, np.float32)
        action[:2] = np.clip(0.8 * (desired_base - rp[:2]), -0.1, 0.1)
        action[2] = np.clip(0.8 * wrap(desired_yaw - rp[2]), -0.1, 0.1)
        action[9] = np.clip(0.8 * (desired_wrist - rp[9]), -0.1, 0.1)
        action[10] = args.open_cmd
        state = step(env, state, action, 1, samples, "align")

    # Descend; optionally start closing during the final approach steps.
    for i in range(args.depth):
        action = np.zeros(11, np.float32)
        action[4], action[6] = 0.1, args.elbow
        action[10] = (args.close_cmd if i >= args.depth - args.close_early
                      else args.open_cmd)
        state = step(env, state, action, 1, samples, "descend")

    action = np.zeros(11, np.float32)
    action[10] = args.close_cmd
    state = step(env, state, action, args.close_hold, samples, "close")

    # A genuine grasp must raise the scoop, then retain it under translation.
    action = np.zeros(11, np.float32)
    action[4], action[6], action[10] = -0.1, -args.elbow, args.close_cmd
    state = step(env, state, action, args.lift_steps, samples, "lift")
    action = np.zeros(11, np.float32)
    action[1], action[10] = args.verify_y, args.close_cmd
    state = step(env, state, action, args.verify_steps, samples, "verify")

    by_phase = {}
    for label, sp, rp in samples:
        by_phase.setdefault(label, []).append((sp, rp))
    print("params", vars(args))
    print("initial_robot", np.round(initial_robot, 4),
          "initial_scoop", np.round(initial_scoop, 4))
    print("yaw", round(yaw, 4), "axis", np.round(long_axis, 4),
          "offset", np.round(xy_offset, 4), "target_base",
          np.round(desired_base, 4), "target_yaw", round(desired_yaw, 4),
          "wrist_delta", round(wrist_delta + args.wrist, 4))
    for label, values in by_phase.items():
        sp = np.asarray([v[0] for v in values])
        rp = np.asarray([v[1] for v in values])
        print(label, "scoop_delta", np.round(sp[-1] - initial_scoop, 4),
              "z_range", np.round([sp[:, 2].min(), sp[:, 2].max()], 4),
              "base_delta", np.round(rp[-1, :2] - initial_robot[:2], 4),
              "grip", round(float(rp[-1, -1]), 4))
    final = scoop_pose(state)
    lifted = final[2] - initial_scoop[2]
    followed = np.linalg.norm(final[:2] - initial_scoop[:2])
    print("RESULT", "lift", round(float(lifted), 4),
          "xy_follow", round(float(followed), 4),
          "final", np.round(final[:3], 4))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--long", type=float, default=0.0)
    parser.add_argument("--short", type=float, default=0.0)
    parser.add_argument("--wrist", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--elbow", type=float, default=0.06)
    parser.add_argument("--close-early", type=int, default=2)
    parser.add_argument("--open-cmd", type=float, default=1.0)
    parser.add_argument("--close-cmd", type=float, default=0.0)
    parser.add_argument("--align-steps", type=int, default=12)
    parser.add_argument("--close-hold", type=int, default=8)
    parser.add_argument("--lift-steps", type=int, default=18)
    parser.add_argument("--verify-y", type=float, default=0.06)
    parser.add_argument("--verify-steps", type=int, default=10)
    trial(parser.parse_args())
