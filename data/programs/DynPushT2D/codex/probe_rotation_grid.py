"""Measure short-horizon torque from pushes at local-frame contact offsets."""

import math

import numpy as np

from env_client import make_env


def rot(v, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def run(seed, local_direction, local_offset):
    env = make_env()
    state, _ = env.reset(seed=seed)
    start = state.copy()
    direction = rot(np.asarray(local_direction, float), state[2])
    direction /= np.linalg.norm(direction)
    offset = rot(np.asarray(local_offset, float), state[2])
    stage = state[:2] + offset - 0.75 * direction

    for _ in range(80):
        delta = stage - state[16:18]
        if np.linalg.norm(delta) < 0.01:
            break
        state, _, term, trunc, _ = env.step(np.clip(delta, -0.049, 0.049).astype(np.float32))
        if term or trunc:
            break

    contact_state = None
    after = None
    for _ in range(30):
        previous = state.copy()
        state, _, term, trunc, _ = env.step((0.049 * direction).astype(np.float32))
        if contact_state is None and np.linalg.norm(state[:2] - previous[:2]) > 1e-5:
            contact_state = previous.copy()
            after = 0
        if after is not None:
            after += 1
            if after >= 4:
                break
        if term or trunc:
            break
    if contact_state is None:
        result = "miss"
    else:
        world_delta = state[:2] - contact_state[:2]
        local_delta = rot(world_delta, -start[2])
        result = "dlocal=(%.3f,%+.3f) dtheta=%+.4f" % (
            local_delta[0], local_delta[1], state[2] - contact_state[2]
        )
    env.close()
    return result


if __name__ == "__main__":
    offsets = [-0.5, -0.35, -0.2, 0.0, 0.25, 0.5]
    for direction in [(1, 0), (-1, 0)]:
        print("direction", direction)
        for value in offsets:
            print(" offset_y=%+.2f" % value, run(0, direction, (0, value)))
    for direction in [(0, 1), (0, -1)]:
        print("direction", direction)
        for value in offsets:
            print(" offset_x=%+.2f" % value, run(0, direction, (value, 0)))
