"""Measure how much first-grasp lowering time is actually required.

This intentionally duplicates only the grasp prefix from ``approach.py`` so
the production policy is never modified while timing candidates are tested.
"""

import argparse
import math

import numpy as np

from env_client import make_env


HOME = np.array([0.0, -0.3491, math.pi, -2.5482, 0.0, -0.8727,
                 math.pi / 2])
GROUND = np.array([0.0, 2.24, 2.945, -1.0, -0.982, 0.20, 1.30])
LIFT = np.array([0.0, 1.45, 2.945, -1.0, -0.982, 0.20, 1.30])


def read(state):
    robot = state.get_object_from_name("robot")
    base = np.array([state.get(robot, "pos_base_x"),
                     state.get(robot, "pos_base_y"),
                     state.get(robot, "pos_base_rot")], dtype=float)
    joints = np.array([state.get(robot, "pos_arm_joint%d" % i)
                       for i in range(1, 8)], dtype=float)
    return base, joints


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, key) for key in ("x", "y", "z")])


def servo(state, base_target, joint_target, grip):
    base, joints = read(state)
    action = np.zeros(11, dtype=np.float32)
    action[:2] = np.clip((base_target[:2] - base[:2]) / .87, -.1, .1)
    angle = (base_target[2] - base[2] + math.pi) % (2 * math.pi) - math.pi
    action[2] = np.clip(angle / .87, -.1, .1)
    action[3:10] = np.clip(.35 * (joint_target - joints), -.1, .1)
    action[10] = grip
    return action


def run(seed, lower_steps):
    env = make_env()
    state, _ = env.reset(seed=seed, options={"object_count": 1})
    cube = next(name for name in state.get_object_names()
                if name.startswith("cube"))
    initial = xyz(state, cube)
    contact = np.array([initial[0] - .609, initial[1] - .054, 0.0])
    retreat = contact.copy()
    retreat[0] -= .195
    stages = ((40, retreat, HOME, 0.0),
              (lower_steps, retreat, GROUND, 0.0),
              (80, contact, GROUND, 0.0),
              (18, contact, GROUND, .6),
              (70, contact, LIFT, .6))
    records = []
    for stage_index, (count, base_target, joint_target, grip) in enumerate(stages):
        for _ in range(count):
            state, _, terminated, truncated, _ = env.step(
                servo(state, base_target, joint_target, grip))
            if terminated or truncated:
                break
        base, joints = read(state)
        records.append((stage_index,
                        float(np.max(np.abs(joint_target - joints))),
                        float(np.linalg.norm(base_target[:2] - base[:2])),
                        xyz(state, cube).round(4).tolist()))
    final = xyz(state, cube)
    env.close()
    success = final[2] > .10
    print("seed", seed, "lower", lower_steps, "success", success,
          "z", round(float(final[2]), 4), "records", records)
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lower", type=int, default=260)
    args = parser.parse_args()
    run(args.seed, args.lower)
