"""Trace and tune the airborne final insertion from the verified drag sequence."""

import sys
import numpy as np

from env_client import make_env
from probe_alt_branches import drive, qpos, cube_pos, get
from solve_alt_fk import fk


def cube_row(state, cube):
    obj = state.get_object_from_name(cube)
    return np.array([state.get(obj, f) for f in ("x", "y", "z", "vx", "vy", "vz")])


def reach_airborne(env, state, cube):
    for cycle in range(14):
        cp = cube_pos(state, cube)
        base = np.array([get(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
        state = drive(env, state, q_target=LIFT, base_target=base + [-.12, 0], grip=.6, steps=55)
        state = drive(env, state, q_target=LOW, grip=0, steps=90)
        point = fk(qpos(state))[:3, 3]
        target = cp[:2] - np.array([-point[0], point[1]]) + [-.02, 0]
        state = drive(env, state, q_target=LOW, base_target=target, grip=0, steps=55)
        state = drive(env, state, q_target=LIFT, grip=.6, steps=65)
        row = cube_row(state, cube)
        print("cycle", cycle, np.round(row, 4).tolist())
        if row[2] > .05:
            return state
    return state


LOW = np.array([0, 2.24, 3.14, -1, 0, -.3, -2.25])
LIFT = np.array([0, 1.4, 3.139, -1.338, .005, -.404, -2.25])


def main():
    magnitude = float(sys.argv[1]) if len(sys.argv) > 1 else .03
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    hold = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lateral = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    prelift = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube = "cube1"
    state = reach_airborne(env, state, cube)
    print("AIR", np.round(cube_row(state, cube), 4).tolist())
    if prelift:
        # Vertical IK continuation, preserving gripper x/y while lifting its
        # flange roughly 15 cm above the previous high pose.
        high = np.array([0, .889, 3.138, -1.514, .004, -.738, -2.25])
        for i in range(prelift):
            error = high - qpos(state)
            action = np.zeros(11, np.float32)
            action[3:10] = np.clip(.35 * error, -.1, .1)
            action[10] = .6
            state, reward, term, trunc, _ = env.step(action)
            print("up", i, round(reward, 3), np.round(cube_row(state, cube), 4).tolist(), term)
            if term or trunc:
                break
    for i in range(steps):
        action = np.zeros(11, np.float32)
        action[0], action[1], action[10] = magnitude, lateral, .6
        state, reward, term, trunc, _ = env.step(action)
        print("ins", i, round(reward, 3), np.round(cube_row(state, cube), 4).tolist(), term)
        if term or trunc:
            break
    for i in range(hold):
        action = np.zeros(11, np.float32)
        action[10] = .6
        state, reward, term, trunc, _ = env.step(action)
        print("hold", i, round(reward, 3), np.round(cube_row(state, cube), 4).tolist(), term)
        if term or trunc:
            break
    for i in range(30):
        state, reward, term, trunc, _ = env.step(np.zeros(11, np.float32))
        print("rel", i, round(reward, 3), np.round(cube_row(state, cube), 4).tolist(), term)
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    main()
