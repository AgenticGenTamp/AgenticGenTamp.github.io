"""Bounded empirical search for a posture that pushes a ground cube."""

import numpy as np
from env_client import make_env

JF = [f"pos_arm_joint{i}" for i in range(1, 8)]


def read(state):
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube1")
    base = np.array([state.get(robot, f) for f in
                     ("pos_base_x", "pos_base_y", "pos_base_rot")], float)
    joints = np.array([state.get(robot, f) for f in JF], float)
    xyz = np.array([state.get(cube, f) for f in ("x", "y", "z")], float)
    return base, joints, xyz


def servo(env, state, target, base_target, steps):
    for _ in range(steps):
        base, joints, _ = read(state)
        action = np.zeros(11, np.float32)
        action[:2] = np.clip(1.7 * (base_target - base[:2]), -.1, .1)
        action[3:10] = np.clip(1.3 * (target - joints), -.1, .1)
        state, *_ = env.step(action)
    return state


rng = np.random.default_rng(923)
limits = np.array([[-3.0, 3.0], [-2.15, 2.15], [-3.0, 3.0],
                   [-2.45, 2.45], [-3.0, 3.0], [-2.0, 2.0],
                   [-3.0, 3.0]])
env = make_env()
for trial in range(28):
    state, _ = env.reset(seed=0, options={"object_count": 1})
    base0, _, cube0 = read(state)
    target = rng.uniform(limits[:, 0], limits[:, 1])
    # Start 55 cm behind and 25 cm to one side, then sweep a 50 cm strip.
    start = cube0[:2] + [-.55, -.25]
    state = servo(env, state, target, start, 55)
    hit = None
    for row, lateral in enumerate((-.25, -.12, 0.0, .12, .25)):
        endpoint = cube0[:2] + ([.12, lateral] if row % 2 == 0
                                else [-.55, lateral])
        for step in range(10):
            base, joints, cube = read(state)
            action = np.zeros(11, np.float32)
            action[:2] = np.clip(1.3 * (endpoint - base[:2]), -.1, .1)
            action[3:10] = np.clip(1.3 * (target - joints), -.1, .1)
            state, reward, term, trunc, _ = env.step(action)
            base, joints, cube = read(state)
            moved = np.linalg.norm(cube - cube0)
            if moved > .003:
                hit = (row, step, base.copy(), joints.copy(), cube.copy(), reward)
                break
        if hit:
            break
    if hit:
        print("HIT trial", trial, "target", np.round(target, 4).tolist(),
              "rowstep", hit[0:2], "base", np.round(hit[2], 4).tolist(),
              "actual", np.round(hit[3], 4).tolist(),
              "cube", np.round(cube0, 4).tolist(), "->",
              np.round(hit[4], 4).tolist(), "reward", hit[5])
        break
    print("NO", trial, "target", np.round(target, 2).tolist())
env.close()
