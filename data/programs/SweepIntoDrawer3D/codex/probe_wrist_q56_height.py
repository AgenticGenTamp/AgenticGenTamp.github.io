"""Focused local grasp search around the known wiper approach pose."""
import itertools

import numpy as np

from env_client import make_env


env = make_env()
obs, _ = env.reset(seed=0)
initial_wiper = obs[147:150].copy()
home = obs[128:135].copy()


def servo(base, joints, grip, steps, speed=0.06):
    global obs
    for _ in range(steps):
        action = np.zeros(11, np.float32)
        pose_error = np.asarray(base) - obs[125:128]
        pose_error[2] = (pose_error[2] + np.pi) % (2 * np.pi) - np.pi
        action[:3] = np.clip(0.7 * pose_error, -speed, speed)
        action[3:10] = np.clip(0.6 * (joints - obs[128:135]), -0.08, 0.08)
        action[10] = grip
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return False
    return True


# Approach with open fingers (action 0), retaining the home wrist except for
# the specified lower shoulder and straight seventh joint.
base0 = np.array([1.23, -0.51, np.pi])
q = home.copy()
q[1] = -0.28
q[6] = 0.0
# Fold at the reset pose first. Trying to fold while moving toward y=-0.51
# traps joint 7 near one radian against the counter/wiper geometry.
start_base = obs[125:128].copy()
start_base[2] = np.pi
servo(start_base, q, 0.0, 30)
print("prefold", np.round(obs[125:136], 4))
servo(base0, q, 0.0, 18)
print("ready", np.round(obs[125:136], 4), "wiper", np.round(obs[147:150], 4))

# q2 is the coarse height variable. q5/q6 rotate the wrist and jaws. Use a
# compact space around home and test whether a closed hand drags the wiper.
heights = (-0.16, -0.28, -0.40)
q5s = (-0.45, 0.0, 0.45)
q6s = (-1.25, -0.87, -0.49)
for trial, (q2, q5, q6) in enumerate(itertools.product(heights, q5s, q6s)):
    target = home.copy()
    target[1], target[4], target[5], target[6] = q2, q5, q6, 0.0
    servo(base0, target, 0.0, 8)
    before_close = obs[147:150].copy()
    servo(base0, target, 1.0, 5)
    after_close = obs[147:150].copy()
    grip_pos = float(obs[135])
    drag_base = base0.copy()
    drag_base[0] += 0.10
    servo(drag_base, target, 1.0, 5)
    after_drag = obs[147:150].copy()
    close_move = float(np.linalg.norm(after_close - before_close))
    drag_move = float(np.linalg.norm(after_drag - after_close))
    cumulative = float(np.linalg.norm(after_drag - initial_wiper))
    print(
        trial,
        "q2/q5/q6", tuple(round(x, 2) for x in (q2, q5, q6)),
        "actual", np.round(obs[[129, 132, 133]], 3),
        "grip", round(grip_pos, 3),
        "close", round(close_move, 4),
        "drag", round(drag_move, 4),
        "cum", round(cumulative, 4),
        "w", np.round(after_drag, 4),
    )
    if drag_move > 0.015:
        print("LIKELY_GRASP", trial)
        break
    servo(base0, target, 0.0, 5)

env.close()
