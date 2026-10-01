"""Sweep the base under fixed arm poses to search for any grasp-height pose."""

import numpy as np

from env_client import make_env


JF = [f"pos_arm_joint{i}" for i in range(1, 8)]
HOME = [0.0, -.3491, 3.1416, -2.5482, 0.0, -.8727, 1.5708]
CANDIDATES = [
    (1.9, 0.0, -2.1, 3.1416, 0.0),
    (1.9, 1.0, -2.1, 3.1416, 0.0),
    (1.9, 1.9, -2.1, 3.1416, 0.0),
    (1.0, 1.0, -2.1, 3.1416, 0.0),
    (0.0, 1.0, -2.1, 3.1416, 0.0),
    (-1.0, 1.0, -2.1, 3.1416, 0.0),
    (-1.9, 1.0, -2.1, 3.1416, 0.0),
    (1.9, 1.0, 1.3, 3.1416, 0.0),
    (-1.0, 1.0, 1.3, 3.1416, 0.0),
    (1.9, 1.0, -2.1, 0.0, 0.0),
    (1.9, 1.0, -2.1, 3.1416, 2.0),
    (1.9, 1.0, -2.1, 0.0, 2.0),
]


def get(state, name, feat):
    return float(state.get(state.get_object_from_name(name), feat))


for q2, q4, q6, q3, q5 in CANDIDATES:
    target = [0.0, q2, q3, q4, q5, q6, 1.5708]
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube0 = np.array([get(state, "cube1", f) for f in ("x", "y", "z")])
    # Reach candidate and put base at left edge of horizontal sweep.
    for _ in range(180):
        action = np.zeros(11, np.float32)
        for j, t in enumerate(target):
            q = get(state, "robot", JF[j])
            action[3 + j] = np.clip(2 * (t - q), -.1, .1)
        action[0] = np.clip(2 * (cube0[0] - .55 - get(state, "robot", "pos_base_x")), -.1, .1)
        action[1] = np.clip(2 * (cube0[1] - get(state, "robot", "pos_base_y")), -.1, .1)
        state, _, _, _, _ = env.step(action)
    hit = None
    for step in range(75):
        action = np.zeros(11, np.float32)
        action[0] = .02
        action[10] = 0.0 if step % 12 < 5 else 1.0
        for j, t in enumerate(target):
            q = get(state, "robot", JF[j])
            action[3 + j] = np.clip(2 * (t - q), -.1, .1)
        state, _, term, trunc, _ = env.step(action)
        cube = np.array([get(state, "cube1", f) for f in ("x", "y", "z")])
        if np.linalg.norm(cube - cube0) > .002:
            hit = (step, get(state, "robot", "pos_base_x"), cube)
            break
        if term or trunc:
            break
    actual = [round(get(state, "robot", f), 2) for f in JF]
    if hit:
        print("HIT", (q2, q4, q6, q3, q5), "actual", actual, "step/base/cube", hit)
    else:
        print("NO", (q2, q4, q6, q3, q5), "actual", actual)
    env.close()
