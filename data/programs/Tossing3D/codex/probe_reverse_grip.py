"""Critical reverse-gripper-polarity test at the fully reached arm pose."""

import math

import numpy as np

from env_client import make_env


Q = np.asarray([0.0, 0.65, math.pi, -0.24, 0.0, -2.25, math.pi / 2])


def vals(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([float(state.get(obj, f)) for f in features])


def base(state):
    return vals(state, "robot", ("pos_base_x", "pos_base_y"))


def joints(state):
    return vals(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))


def cube(state):
    return vals(state, "cube_0", ("x", "y", "z"))


def advance(env, state, action):
    state, reward, terminated, truncated, _ = env.step(action)
    return state, reward, terminated or truncated


def test(env, xoff):
    state, _ = env.reset(seed=0)
    cube0 = cube(state)
    target_base = cube0[:2] - np.asarray([xoff, 0.0])
    # Base commands integrate, so make their sum equal the desired translation.
    remaining = target_base - base(state)
    while np.max(np.abs(remaining)) > 1e-7:
        inc = np.clip(remaining, -0.05, 0.05)
        action = np.zeros(18, dtype=np.float32)
        action[:2] = inc
        action[10] = 1.0
        state, _, _ = advance(env, state, action)
        remaining -= inc
    for _ in range(20):
        action = np.zeros(18, dtype=np.float32)
        action[10] = 1.0
        state, _, _ = advance(env, state, action)

    # Hold polarity 1 throughout a full 90-step servo to the nominal q.
    for _ in range(90):
        action = np.zeros(18, dtype=np.float32)
        action[3:10] = np.clip(Q - joints(state), -0.1, 0.1)
        action[10] = 1.0
        state, _, _ = advance(env, state, action)
    q_reached = joints(state)
    before = cube(state)

    # Reverse to 0 and allow full gripper travel.
    for _ in range(15):
        action = np.zeros(18, dtype=np.float32)
        action[10] = 0.0
        state, _, _ = advance(env, state, action)
    closed = cube(state)

    first = None
    for i in range(5):
        action = np.zeros(18, dtype=np.float32)
        action[0] = -0.05
        action[10] = 0.0
        state, _, _ = advance(env, state, action)
        if i == 0:
            first = cube(state).copy()
    for _ in range(15):
        action = np.zeros(18, dtype=np.float32)
        action[10] = 0.0
        state, _, _ = advance(env, state, action)
    final = cube(state)
    print("xoff=%.2f" % xoff,
          "q=" + np.array2string(q_reached, precision=3),
          "reverse_shift=%.6f" % np.linalg.norm(closed - before),
          "first_retreat=%.6f" % np.linalg.norm(first - closed),
          "full_retreat=%.6f" % np.linalg.norm(final - closed),
          "cube_final=" + np.array2string(final, precision=4))


def main():
    env = make_env()
    for xoff in (0.45, 0.50, 0.55):
        test(env, xoff)
    env.close()


if __name__ == "__main__":
    main()
