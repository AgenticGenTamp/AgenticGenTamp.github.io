"""Scan projected base offsets for the promising grasp joint pose."""

import numpy as np

from env_client import make_env


Q = np.asarray([0.23, 0.39, 3.40, 0.0, 0.17, -1.57, 2.46])
XS = (0.10, 0.18, 0.26, 0.34, 0.42)
YS = (-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40)
EPS = 1e-4


def values(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([float(state.get(obj, f)) for f in features])


def cube(state):
    return values(state, "cube_0", ("x", "y", "z"))


def base(state):
    return values(state, "robot", ("pos_base_x", "pos_base_y"))


def qpos(state):
    return values(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))


def step(env, action):
    state, _, terminated, truncated, _ = env.step(action)
    return state, terminated or truncated


def moved(state, reference):
    return float(np.linalg.norm(cube(state) - reference))


def cell(env, xoff, yoff, open_value, close_value):
    state, _ = env.reset(seed=0)
    c0 = cube(state)
    goal = c0[:2] - np.asarray([xoff, yoff])
    remaining = goal - base(state)
    while np.max(np.abs(remaining)) > 1e-7:
        inc = np.clip(remaining, -0.05, 0.05)
        action = np.zeros(18, dtype=np.float32)
        action[:2] = inc
        action[10] = open_value
        state, _ = step(env, action)
        remaining -= inc
    for _ in range(15):
        action = np.zeros(18, dtype=np.float32)
        action[10] = open_value
        state, _ = step(env, action)

    for i in range(70):
        action = np.zeros(18, dtype=np.float32)
        action[3:10] = np.clip(Q - qpos(state), -0.1, 0.1)
        action[10] = open_value
        state, _ = step(env, action)
        shift = moved(state, c0)
        if shift > EPS:
            return "servo%d" % (i + 1), shift, qpos(state), cube(state)

    reached = qpos(state)
    before = cube(state)
    for i in range(15):
        action = np.zeros(18, dtype=np.float32)
        action[10] = close_value
        state, _ = step(env, action)
        shift = moved(state, before)
        if shift > EPS:
            return "close%d" % (i + 1), shift, reached, cube(state)

    # Lift by retracting shoulder joint 2 from the low grasp configuration.
    # Continuous -0.1 commands yield a clear ~25 cm/rad-equivalent motion.
    before_lift = cube(state)
    for i in range(10):
        action = np.zeros(18, dtype=np.float32)
        action[4] = -0.1
        action[10] = close_value
        state, _ = step(env, action)
        shift = moved(state, before_lift)
        if shift > EPS:
            return "lift%d" % (i + 1), shift, reached, cube(state)

    before_retreat = cube(state)
    for i in range(5):
        action = np.zeros(18, dtype=np.float32)
        action[0] = -0.05
        action[10] = close_value
        state, _ = step(env, action)
        shift = moved(state, before_retreat)
        if shift > EPS:
            return "retreat%d" % (i + 1), shift, reached, cube(state)
    for i in range(10):
        action = np.zeros(18, dtype=np.float32)
        action[10] = close_value
        state, _ = step(env, action)
        shift = moved(state, before_retreat)
        if shift > EPS:
            return "settle%d" % (i + 1), shift, reached, cube(state)
    return None, 0.0, reached, cube(state)


def main():
    env = make_env()
    count = 0
    for xoff in XS:
        for yoff in YS:
            count += 1
            phase, shift, q, c = cell(env, xoff, yoff, 0.0, 1.0)
            if phase:
                print("FOUND open0-close1", (xoff, yoff), phase, "shift", shift,
                      "q", np.round(q, 3), "cube", np.round(c, 4))
                env.close()
                return
        print("completed x", xoff, "cells", count)
    print("NO_MOVE open0-close1", count)

    print("SKIP reverse polarity: no approach/contact indication")
    env.close()


if __name__ == "__main__":
    main()
