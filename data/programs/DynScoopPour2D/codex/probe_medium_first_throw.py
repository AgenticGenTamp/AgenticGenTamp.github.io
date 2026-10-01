"""Focused first-throw probe; never imported by the submitted approach."""
import math
import sys

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


seed = int(sys.argv[1])
y_offset = float(sys.argv[2])
theta_target = float(sys.argv[3])
x_target = float(sys.argv[4]) if len(sys.argv) > 4 else 1.40

env = make_env()
state, info = env.reset(seed=seed)
policy = GeneratedApproach(env.action_space, env.observation_space, {})
policy.reset(state, info)
names = [n for n in state.get_object_names() if n.startswith("small_")]


def get(name, feature):
    obj = state.get_object_from_name(name)
    return float(state.get(obj, feature))


def count_right():
    return sum(get(name, "x") > 1.75 for name in names)


terminated = False
steps = 0
while policy.stage < 8 and steps < 600 and not terminated:
    state, _, terminated, truncated, _ = env.step(policy.get_action(state))
    steps += 1

reached = policy.stage >= 8
prep_stage = policy.stage
low_y = policy.low_y
best = count_right()


def control(tx, ty, tt, max_steps, settle=3):
    global state, terminated, steps, best
    dwell = 0
    for _ in range(max_steps):
        ex = tx - get("robot", "x")
        ey = ty - get("robot", "y")
        et = (tt - get("robot", "theta") + math.pi) % (2 * math.pi) - math.pi
        ej = 0.20 - get("robot", "arm_joint")
        eg = 0.08 - get("robot", "finger_gap")
        arrived = abs(ex) <= .0345 and abs(ey) <= .0345 and abs(et) <= .113
        dwell = dwell + 1 if arrived else 0
        action = np.array([
            np.clip(ex, -.03, .03), np.clip(ey, -.03, .03),
            np.clip(et, -.098, .098), np.clip(ej, -.08, .08),
            np.clip(eg, -.015, .015),
        ])
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1
        best = max(best, count_right())
        if terminated or truncated or dwell >= settle:
            return


if reached and not terminated:
    # Adjust base height with the hook still vertical, then perform one clean arc.
    control(x_target, low_y + y_offset, -math.pi / 2, 80, settle=3)
    if not terminated:
        control(x_target, low_y + y_offset, theta_target, 80, settle=4)
    if not terminated:
        for _ in range(20):
            state, _, terminated, truncated, _ = env.step(np.zeros(5))
            steps += 1
            best = max(best, count_right())
            if terminated or truncated:
                break

print(
    f"seed={seed} n={len(names)} reached={reached} prep_stage={prep_stage} "
    f"low={low_y:.3f} dy={y_offset:+.2f} theta={theta_target:+.2f} "
    f"x={x_target:.2f} right={count_right()} best={best} done={terminated} "
    f"held={get('hook', 'held'):.0f} steps={steps}"
)
env.close()
