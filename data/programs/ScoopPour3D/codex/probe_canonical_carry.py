"""Diagnose canonical scoop grasp retention and downstream cube motion.

This intentionally duplicates only the short experimental sequence.  It does
not import or modify the submitted policy.
"""

import argparse
import math

import numpy as np

from env_client import make_env


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([state.get(obj, f) for f in features], dtype=float)


def wrap(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def advance(env, state, action, count):
    for _ in range(count):
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return state


def cube_xy(state, names):
    return np.asarray([read(state, name, ("x", "y")) for name in names])


def run(args):
    env = make_env()
    state, _ = env.reset(seed=args.seed)
    cube_names = sorted(name for name in state.get_object_names()
                        if name.startswith("cube_"))
    cubes0 = cube_xy(state, cube_names)
    scoop0 = read(state, "scoop_0", ("x", "y", "z", "qw", "qx", "qy", "qz"))
    base0 = read(state, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
    joints0 = read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))
    q = scoop0[3:]
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                     1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
    axis = np.asarray([math.cos(yaw), math.sin(yaw)])
    target_xy = scoop0[:2] - np.asarray([0.417, 0.0]) + args.long * axis
    target_yaw = math.atan2(scoop0[1] - target_xy[1],
                            scoop0[0] - target_xy[0])
    wrist_delta = wrap(yaw - target_yaw + 0.557)
    if wrist_delta > math.pi / 2.0:
        wrist_delta -= math.pi
    elif wrist_delta < -math.pi / 2.0:
        wrist_delta += math.pi
    target_wrist = joints0[6] + wrist_delta

    # Exact canonical grasp: 12 feedback alignment increments, 32 descent
    # increments, two-step close, eight-step hold, and 18-step inverse lift.
    for _ in range(12):
        base = read(state, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
        wrist = read(state, "robot", ("pos_arm_joint7",))[0]
        action = np.zeros(11, np.float32)
        action[:2] = np.clip(0.8 * (target_xy - base[:2]), -0.1, 0.1)
        action[2] = np.clip(0.8 * wrap(target_yaw - base[2]), -0.1, 0.1)
        action[9] = np.clip(0.8 * (target_wrist - wrist), -0.1, 0.1)
        action[10] = 1.0
        state = advance(env, state, action, 1)
    for index in range(32):
        action = np.zeros(11, np.float32)
        action[4], action[6] = 0.1, 0.06
        action[10] = (0.0 if index >= 32 - args.close_early else 1.0)
        state = advance(env, state, action, 1)
    action = np.zeros(11, np.float32)
    action[10] = 0.0
    state = advance(env, state, action, 8)
    action[4], action[6] = -0.1, -0.06
    state = advance(env, state, action, 18)
    after_lift = read(state, "scoop_0", ("x", "y", "z"))
    lift_joints = read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))

    # Carry slowly toward source.  Zero joint actions hold the compact lift.
    for _ in range(args.carry_steps):
        action = np.zeros(11, np.float32)
        action[1], action[10] = args.speed, 0.0
        if args.stabilize:
            joints = read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))
            action[3:10] = np.clip(0.5 * (lift_joints - joints), -0.03, 0.03)
        state = advance(env, state, action, 1)
    after_carry = read(state, "scoop_0", ("x", "y", "z"))
    base_carry = read(state, "robot", ("pos_base_x", "pos_base_y"))

    if args.downstream:
        action = np.zeros(11, np.float32)
        action[6], action[10] = 0.1, 0.0
        state = advance(env, state, action, 30)
        for _ in range(53):
            action = np.zeros(11, np.float32)
            z = read(state, "scoop_0", ("z",))[0]
            action[4], action[10] = (0.06 if z > 0.466 else 0.0), 0.0
            state = advance(env, state, action, 1)
        for _ in range(25):
            action = np.zeros(11, np.float32)
            z = read(state, "scoop_0", ("z",))[0]
            action[4] = 0.1 if z > 0.466 else 0.0
            action[6], action[10] = 0.1, 0.0
            state = advance(env, state, action, 1)
        action = np.zeros(11, np.float32)
        action[1], action[5], action[10] = 0.012, -0.1, 0.0
        state = advance(env, state, action, 127)

    cubes1 = cube_xy(state, cube_names)
    disp = cubes1 - cubes0
    print("seed", args.seed, "count", len(cube_names), "long", args.long,
          "speed", args.speed, "carry_steps", args.carry_steps)
    print("lift", np.round(after_lift - scoop0[:3], 4),
          "carry", np.round(after_carry - after_lift, 4),
          "base", np.round(base_carry - target_xy, 4))
    print("cube_mean_delta", np.round(disp.mean(axis=0), 4),
          "cube_y_range", np.round([disp[:, 1].min(), disp[:, 1].max()], 4))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--long", type=float, default=0.06)
    parser.add_argument("--speed", type=float, default=-0.01)
    parser.add_argument("--carry-steps", type=int, default=24)
    parser.add_argument("--close-early", type=int, default=1)
    parser.add_argument("--stabilize", action="store_true")
    parser.add_argument("--downstream", action="store_true")
    run(parser.parse_args())
