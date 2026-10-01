"""Probe coupled Gen3 pitch-joint reaches from the one-cube start."""

import sys

import numpy as np

from env_client import make_env


JF = [f"pos_arm_joint{i}" for i in range(1, 8)]


def vals(state):
    r = state.get_object_from_name("robot")
    c = state.get_object_from_name("cube1")
    q = np.array([state.get(r, f) for f in JF], float)
    xyz = np.array([state.get(c, f) for f in ("x", "y", "z")], float)
    return q, xyz


# Commands for action indices q2=4, q4=6, q6=8.  q4 positive unfolds
# from the strongly negative folded home angle; the other signs select arc.
COMMANDS = [
    (0, .10, 0),
    (-.05, .10, 0),
    (.05, .10, 0),
    (-.05, .10, -.05),
    (-.05, .10, .05),
    (.05, .10, -.05),
    (.05, .10, .05),
    (0, .10, -.05),
    (0, .10, .05),
    (-.10, .05, .05),
    (.10, .05, -.05),
]


for command in (COMMANDS if "--basic" in sys.argv else ()):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    q0, cube0 = vals(state)
    hit = None
    for step in range(180):
        action = np.zeros(11, np.float32)
        action[4], action[6], action[8] = command
        action[10] = 0.0
        state, reward, term, trunc, _ = env.step(action)
        q, cube = vals(state)
        moved = float(np.linalg.norm(cube - cube0))
        if moved > .003:
            hit = (step, q, cube, moved)
            break
        if term or trunc:
            break
    if hit is None:
        print("NO", command, "q", q.round(3).tolist())
    else:
        print("HIT", command, "step", hit[0], "q", hit[1].round(3).tolist(),
              "cube", hit[2].round(3).tolist(), "move", round(hit[3], 4))
    env.close()


# A structured grid: fix shoulder pitch, then unfold elbow while counter-
# rotating the wrist and repeatedly opening/closing the fingers.  This detects
# sticky/closed-finger grasps even if robot-object collision response is muted.
for q2_target in (-1.7, -1.1, -.5, .1, .7, 1.3):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    _, cube0 = vals(state)
    for _ in range(120):
        q, _ = vals(state)
        err = q2_target - q[1]
        if abs(err) < .03:
            break
        action = np.zeros(11, np.float32)
        action[4] = np.clip(2 * err, -.1, .1)
        state, _, _, _, _ = env.step(action)
    hit = None
    for step in range(220):
        action = np.zeros(11, np.float32)
        action[6] = .05
        action[8] = -.05
        action[10] = 0.0 if step % 16 < 7 else 1.0
        state, _, term, trunc, _ = env.step(action)
        q, cube = vals(state)
        moved = float(np.linalg.norm(cube - cube0))
        if moved > .003:
            hit = (step, q.copy(), cube.copy(), moved, action[10])
            break
        if term or trunc:
            break
    if hit:
        print("GRIDHIT q2", q2_target, "step", hit[0], "q", hit[1].round(3).tolist(),
              "cube", hit[2].round(3).tolist(), "move", round(hit[3], 4), "grip", hit[4])
    else:
        print("grid-no", q2_target, "q", q.round(3).tolist())
    env.close()


# Visually calibrated low pose, with base translation compensating for the
# arm's tucked horizontal reach.
env = make_env()
state, _ = env.reset(seed=0, options={"object_count": 1})
_, cube0 = vals(state)
robot = state.get_object_from_name("robot")
for step in range(180):
    action = np.zeros(11, np.float32)
    for index, feature, target in (
        (0, "pos_base_x", cube0[0] - .18),
        (1, "pos_base_y", cube0[1]),
        (2, "pos_base_rot", 0.0),
        (4, "pos_arm_joint2", 1.9),
        (6, "pos_arm_joint4", 0.0),
        (8, "pos_arm_joint6", -2.1),
    ):
        current = float(state.get(robot, feature))
        action[index] = np.clip(2 * (target - current), -.1, .1)
    action[10] = 0.0
    state, _, _, _, _ = env.step(action)
    robot = state.get_object_from_name("robot")
q, cube = vals(state)
print("CALIB open", "q", q.round(3).tolist(), "cube", cube.round(4).tolist())
for step in range(20):
    action = np.zeros(11, np.float32)
    action[10] = 1.0
    state, _, _, _, _ = env.step(action)
    q, cube = vals(state)
    if np.linalg.norm(cube - cube0) > .002:
        print("CALIB CONTACT", step, "q", q.round(3).tolist(), "cube", cube.round(4).tolist())
        break
else:
    print("CALIB no-contact", "cube", cube.round(4).tolist())
env.close()
