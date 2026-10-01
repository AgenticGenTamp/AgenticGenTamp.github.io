"""Probe sign, braking, and gripper transients for selected actions."""

import numpy as np

from env_client import make_env


POS = [
    "pos_base_x", "pos_base_y", "pos_base_rot",
    "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3", "pos_arm_joint4",
    "pos_arm_joint5", "pos_arm_joint6", "pos_arm_joint7", "pos_gripper",
]
VEL = [
    "vel_base_x", "vel_base_y", "vel_base_rot",
    "vel_arm_joint1", "vel_arm_joint2", "vel_arm_joint3", "vel_arm_joint4",
    "vel_arm_joint5", "vel_arm_joint6", "vel_arm_joint7", "vel_gripper",
]


def read(state, features):
    robot = state.get_object_from_name("robot")
    return np.array([state.get(robot, f) for f in features])


def run(dim, command):
    env = make_env()
    state, _ = env.reset(seed=0)
    p0 = read(state, POS)
    a = np.zeros(11, dtype=np.float32)
    a[dim] = command
    rows = []
    for label, act in (("command", a), ("zero1", np.zeros(11, np.float32)),
                       ("zero2", np.zeros(11, np.float32)), ("zero3", np.zeros(11, np.float32))):
        state, reward, term, trunc, _ = env.step(act)
        rows.append((label, read(state, POS) - p0, read(state, VEL), reward, term, trunc))
    env.close()
    print("dim", dim, "cmd", command)
    for label, delta, velocity, reward, term, trunc in rows:
        interesting = sorted(set([dim] + [i for i, x in enumerate(delta) if abs(x) > 1e-3]))
        print(label, "dp", [(i, round(delta[i], 5)) for i in interesting],
              "v", [(i, round(velocity[i], 5)) for i in interesting], "r", reward, term, trunc)


def main():
    for dim in (0, 1, 2, 3, 6, 9):
        run(dim, -0.1)
    run(10, 1.0)


if __name__ == "__main__":
    main()
