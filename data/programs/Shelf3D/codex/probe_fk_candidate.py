"""Live validation of corrected-URDF down-facing low-tool candidates."""

import numpy as np

from env_client import make_env


JF = [f"pos_arm_joint{i}" for i in range(1, 8)]
CANDIDATES = [
    [1.570, .159, -1.454, .937, -2.944, 2.10, 2.427],
    [1.574, .954, -1.023, 1.216, 1.054, -2.10, -2.003],
    [-3.142, -.110, -1.650, .932, -.137, -2.10, -.096],
]


def get(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


for target in CANDIDATES:
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube0 = np.array([get(state, "cube1", f) for f in ("x", "y", "z")])
    # Reach the candidate while placing the base left of the cube.
    for _ in range(210):
        a = np.zeros(11, np.float32)
        for j, t in enumerate(target):
            q = get(state, "robot", JF[j])
            a[3 + j] = np.clip(2 * (t - q), -.1, .1)
        a[0] = np.clip(2 * (cube0[0] - .40 - get(state, "robot", "pos_base_x")), -.1, .1)
        a[1] = np.clip(2 * (cube0[1] - get(state, "robot", "pos_base_y")), -.1, .1)
        state, _, _, _, _ = env.step(a)
    actual0 = [get(state, "robot", f) for f in JF]
    hit = None
    # Sweep across the cube and cycle the gripper, maintaining arm targets.
    for step in range(48):
        a = np.zeros(11, np.float32)
        a[0] = .02
        a[10] = 0.0 if step % 10 < 4 else 1.0
        for j, t in enumerate(target):
            q = get(state, "robot", JF[j])
            a[3 + j] = np.clip(2 * (t - q), -.1, .1)
        state, _, term, trunc, _ = env.step(a)
        cube = np.array([get(state, "cube1", f) for f in ("x", "y", "z")])
        if np.linalg.norm(cube - cube0) > .002:
            hit = (step, get(state, "robot", "pos_base_x"), cube.copy())
            break
        if term or trunc:
            break
    actual1 = [get(state, "robot", f) for f in JF]
    print("HIT" if hit else "NO", "target", np.round(target, 3).tolist(),
          "reached", np.round(actual0, 3).tolist(), "end", np.round(actual1, 3).tolist(),
          "event", hit)
    env.close()
