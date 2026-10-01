"""Gently move the red bin around its cube and check settled containment."""
import numpy as np

from env_client import make_env
from edge_rake_probe import action_to, cubes, get, robot

HOME = np.array([0., -.349, np.pi, -2.548, 0., -.873, np.pi / 2])
PUSH = np.array([0., 1.30, np.pi, -1.70, 0., 1., 0.])


def move_bin(env, state, destination):
    source = get(state, "bin_red", ["x", "y"])[:2]
    delta = destination - source
    axis = int(abs(delta[1]) > abs(delta[0]))
    delta[1 - axis] = 0.
    unit = delta / max(1e-8, np.linalg.norm(delta))
    yaw = float(np.arctan2(unit[1], unit[0]))
    base = robot(state)[:3]
    start_angle = float(np.arctan2(base[1], base[0]))
    end_angle = float(np.arctan2(-unit[1], -unit[0]))
    turn = (end_angle - start_angle + np.pi) % (2 * np.pi) - np.pi
    durations = (25, 35, 15, 75, 25)
    for step in range(sum(durations)):
        if step < durations[0]:
            goal, qgoal, limit = base.copy(), HOME, .1
        elif step < sum(durations[:2]):
            f = (step - durations[0] + 1) / durations[1]
            ang = start_angle + turn * f
            goal, qgoal, limit = np.array([np.cos(ang), np.sin(ang), yaw]), HOME, .1
        elif step < sum(durations[:3]):
            goal, qgoal, limit = np.r_[source - .98 * unit, yaw], HOME, .1
        elif step < sum(durations[:4]):
            goal, qgoal, limit = np.r_[source - .98 * unit, yaw], PUSH, .1
        else:
            goal, qgoal, limit = np.r_[destination - .965 * unit, yaw], PUSH, .008
        action = action_to(state, goal, qgoal, 0)
        action[:3] = np.clip(action[:3], -limit, limit)
        state, reward, term, trunc, _ = env.step(action)
    return state


env = make_env()
state, _ = env.reset(seed=0, options={"object_count": 4})
cube0 = cubes(state)["cube1"].copy()
bin0 = get(state, "bin_red", ["x", "y", "z"]).copy()
state = move_bin(env, state, np.array([cube0[0], bin0[1]]))
print("AFTER_X", "cube", np.round(cubes(state)["cube1"], 5),
      "bin", np.round(get(state, "bin_red", ["x", "y", "z"]), 5))
state = move_bin(env, state, cube0[:2].copy())
print("AFTER_Y", "cube", np.round(cubes(state)["cube1"], 5),
      "bin", np.round(get(state, "bin_red", ["x", "y", "z"]), 5))

# Retract outward and fold, observing persistent relative pose.
base = robot(state)[:3].copy()
base[:2] *= 1.25 / max(.01, np.linalg.norm(base[:2]))
for step in range(140):
    state, reward, term, trunc, _ = env.step(action_to(state, base, HOME, 0))
    if step % 10 == 0 or reward != -1. or term:
        cube = cubes(state)["cube1"]
        binpos = get(state, "bin_red", ["x", "y", "z"])
        print("SETTLE", step, reward, term, "cube", np.round(cube, 5),
              "bin", np.round(binpos, 5),
              "relative", np.round(cube - binpos, 5))
    if term or trunc:
        break
env.close()
