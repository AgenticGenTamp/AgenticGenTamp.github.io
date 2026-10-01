"""Locate the low arm geometry by sweeping it slowly through one cube."""

import numpy as np
from env_client import make_env


# Collision-free low side-pinch solution from the Gen3 URDF FK.
Q = np.array([1.136, -1.351, 0.322, -1.639, -0.140, 1.210, 1.793])


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.array([float(state.get(obj, feature)) for feature in features])


def run(yoff, grip):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube0 = read(state, "cube_0", ("x", "y", "z"))
    # Start far enough behind the cube that arm deployment cannot touch it.
    target = cube0[:2] - np.array([0.85, yoff])
    for _ in range(60):
        base = read(state, "robot", ("pos_base_x", "pos_base_y"))
        q = read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))
        action = np.zeros(18, np.float32)
        action[:2] = np.clip(target - base, -.05, .05)
        action[3:10] = np.clip(Q - q, -.05, .05)
        action[10] = grip
        action[11:18] = np.clip(4 * (Q - q), -2., 2.)
        state, *_ = env.step(action)
    # Translate through the cube. First contact tells us the actual planar
    # projection of whichever low link/finger is nearest the floor.
    hit = None
    for step in range(20):
        action = np.zeros(18, np.float32)
        action[0] = .04
        action[10] = grip
        state, *_ = env.step(action)
        cube = read(state, "cube_0", ("x", "y", "z"))
        if np.linalg.norm(cube - cube0) > 1e-4 and hit is None:
            base = read(state, "robot", ("pos_base_x", "pos_base_y"))
            hit = (step + 1, *(cube0[:2] - base), *(cube - cube0))
    print("yoff", yoff, "grip", grip, "hit", hit,
          "q", np.round(read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8))), 3),
          flush=True)
    env.close()


for yoff, grip in ((-.12, 0.), (0., 0.), (.12, 0.), (0., 1.)):
    run(yoff, grip)
