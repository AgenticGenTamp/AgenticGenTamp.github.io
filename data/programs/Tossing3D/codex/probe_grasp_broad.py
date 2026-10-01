"""Broad cube-relative XY search for contact/grasp at the full arm command."""

import math

import numpy as np

from env_client import make_env


Q = np.asarray([0.0, 0.65, math.pi, -0.24, 0.0, -2.25, math.pi / 2])
XS = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85)
YS = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)
EPS = 1e-4


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([float(state.get(obj, feature)) for feature in features])


def cube(state):
    return read(state, "cube_0", ("x", "y", "z"))


def base(state):
    return read(state, "robot", ("pos_base_x", "pos_base_y"))


def qpos(state):
    return read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))


def do_step(env, action):
    state, reward, terminated, truncated, _ = env.step(action)
    return state, terminated or truncated


def run_cell(env, xoff, yoff, open_value=0.0, close_value=1.0):
    state, _ = env.reset(seed=0)
    c0 = cube(state)
    goal = c0[:2] - np.asarray([xoff, yoff])
    remaining = goal - base(state)
    while np.max(np.abs(remaining)) > 1e-7:
        inc = np.clip(remaining, -0.05, 0.05)
        action = np.zeros(18, dtype=np.float32)
        action[:2] = inc
        action[10] = open_value
        state, _ = do_step(env, action)
        remaining -= inc
    for _ in range(15):
        action = np.zeros(18, dtype=np.float32)
        action[10] = open_value
        state, _ = do_step(env, action)

    # Monitor cube during continuous joint servo as well as during grasp/retreat.
    for step_num in range(70):
        action = np.zeros(18, dtype=np.float32)
        action[3:10] = np.clip(Q - qpos(state), -0.1, 0.1)
        action[10] = open_value
        state, _ = do_step(env, action)
        shift = float(np.linalg.norm(cube(state) - c0))
        if shift > EPS:
            return "servo%d" % (step_num + 1), shift, qpos(state), cube(state)

    q_reached = qpos(state)
    before_close = cube(state)
    for step_num in range(15):
        action = np.zeros(18, dtype=np.float32)
        action[10] = close_value
        state, _ = do_step(env, action)
        shift = float(np.linalg.norm(cube(state) - before_close))
        if shift > EPS:
            return "close%d" % (step_num + 1), shift, q_reached, cube(state)

    before_move = cube(state)
    for step_num in range(5):
        action = np.zeros(18, dtype=np.float32)
        action[0] = -0.05
        action[10] = close_value
        state, _ = do_step(env, action)
        shift = float(np.linalg.norm(cube(state) - before_move))
        if shift > EPS:
            return "retreat%d" % (step_num + 1), shift, q_reached, cube(state)
    for step_num in range(15):
        action = np.zeros(18, dtype=np.float32)
        action[10] = close_value
        state, _ = do_step(env, action)
        shift = float(np.linalg.norm(cube(state) - before_move))
        if shift > EPS:
            return "settle%d" % (step_num + 1), shift, q_reached, cube(state)
    return None, 0.0, q_reached, cube(state)


def main():
    env = make_env()
    tested = 0
    for xoff in XS:
        for yoff in YS:
            tested += 1
            phase, shift, q, c = run_cell(env, xoff, yoff)
            if phase is not None:
                print("FOUND open0-close1", "offset", (xoff, yoff), "phase", phase,
                      "shift", shift, "q", np.round(q, 3), "cube", np.round(c, 4))
                env.close()
                return
        print("completed x", xoff, "cells", tested)
    print("NO_MOVE open0-close1 cells", tested)

    # If the main grid misses, sample central/high-likelihood cells with polarity
    # reversed. This is deliberately smaller to preserve exploration budget.
    tested_reverse = 0
    for xoff in (0.25, 0.40, 0.55, 0.70):
        for yoff in (-0.10, 0.0, 0.10):
            tested_reverse += 1
            phase, shift, q, c = run_cell(env, xoff, yoff, 1.0, 0.0)
            if phase is not None:
                print("FOUND open1-close0", "offset", (xoff, yoff), "phase", phase,
                      "shift", shift, "q", np.round(q, 3), "cube", np.round(c, 4))
                env.close()
                return
    print("NO_MOVE open1-close0 cells", tested_reverse)
    env.close()


if __name__ == "__main__":
    main()
