"""Carry the airborne cube around the cupboard side and insert from +x."""

import numpy as np

from env_client import make_env
from probe_alt_branches import qpos
from probe_insertion_trace import reach_airborne, cube_row


HIGH = np.array([0, .889, 3.138, -1.514, .004, -.738, -2.25])


def move(env, state, cube, ax, ay, steps, label):
    for i in range(steps):
        action = np.zeros(11, np.float32)
        action[0], action[1], action[10] = ax, ay, .6
        state, reward, term, trunc, _ = env.step(action)
        if i == steps - 1 or term:
            print(label, i, round(reward, 3), np.round(cube_row(state, cube), 4).tolist(), term)
        if term or trunc:
            break
    return state


env = make_env()
state, _ = env.reset(seed=0, options={"object_count": 1})
cube = "cube1"
state = reach_airborne(env, state, cube)
for _ in range(20):
    action = np.zeros(11, np.float32)
    action[3:10] = np.clip(.35 * (HIGH - qpos(state)), -.1, .1)
    action[10] = .6
    state, *_ = env.step(action)
print("high", np.round(cube_row(state, cube), 4).tolist())
# Circumnavigate the +y side, pass the rear plane, then center on shelf.
state = move(env, state, cube, 0, .04, 23, "side_out")
state = move(env, state, cube, .04, 0, 50, "behind")
state = move(env, state, cube, 0, -.04, 25, "side_in")
state = move(env, state, cube, -.025, 0, 100, "insert_negx")
for i in range(35):
    state, reward, term, trunc, _ = env.step(np.zeros(11, np.float32))
    if i < 5 or i % 5 == 4 or term:
        print("release", i, round(reward, 3), np.round(cube_row(state, cube), 4).tolist(), term)
    if term or trunc:
        break
env.close()
