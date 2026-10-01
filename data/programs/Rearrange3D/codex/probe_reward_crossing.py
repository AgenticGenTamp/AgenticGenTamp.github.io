"""Continue seed-0 can pushes through the presumed target and log rewards."""

import numpy as np
from env_client import make_env
from approach import GeneratedApproach


env = make_env()
s, info = env.reset(seed=0)
agent = GeneratedApproach(env.action_space, env.observation_space, {})
agent.reset(s, info)
home = s[96:103].copy()
for k in range(500):
    s, r, term, trunc, info = env.step(agent.get_action(s))
    if r != -1 or term or trunc:
        print("policy event", k, r, term, np.round(s[[1, 17, 33]], 3).tolist())
    if agent.phase >= 11:
        break
print("policy end", k, np.round(s[[1, 17, 33, 93, 94]], 3).tolist())


def servo_pose(target, limit=75):
    global s
    for j in range(limit):
        a = np.zeros(11, np.float32)
        err = target - s[96:103]
        a[3:10] = np.clip(err * .5, -.1, .1)
        s, r, term, trunc, info = env.step(a)
        if np.max(np.abs(err)) < .045:
            break


servo_pose(home)
for j in range(45):
    a = np.zeros(11, np.float32)
    goal = s[33] - .28
    a[1] = np.clip((goal-s[94])*.5, -.1, .1)
    s, r, term, trunc, info = env.step(a)
    if abs(goal-s[94]) < .025:
        break
# The can was shoved ~18 cm inward in x by the first passes.  Re-fold the
# elbow to the empirically shorter-reach posture before the next engagement.
pose = home.copy(); pose[1] = .80; pose[3] = -1.72
servo_pose(pose)
previous_y = float(s[33])
for j in range(100):
    a = np.zeros(11, np.float32); a[1] = .10
    s, r, term, trunc, info = env.step(a)
    if abs(float(s[33])-previous_y) > .003 or r != -1 or j % 10 == 9:
        print("cross", j, "r", r, "done", term,
              "bowl/box/can y", np.round(s[[1,17,33]], 3).tolist(),
              "can xyz", np.round(s[32:35], 3).tolist())
    previous_y = float(s[33])
    if term or trunc:
        break
env.close()
