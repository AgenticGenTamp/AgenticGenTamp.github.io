"""Exploit the short retained tray lift to translate and release over cube1."""
import sys

import numpy as np

from env_client import make_env
from topdown_grasp_probe import get, rob, xyz, act, run


HIGH = np.array([0., .86, np.pi, -1.70, 0., 1., np.pi / 2])
LOW = HIGH.copy()
LOW[1] = 1.15
LIFT = HIGH.copy()
LIFT[1] = .82


def trial(up_steps, down_steps):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    home = rob(state)[3:10].copy()
    b0 = xyz(state, "bin_red").copy()
    c0 = xyz(state, "cube1").copy()
    state = run(env, state, 45, [1., .75, np.pi], home, 1)
    state = run(env, state, 55, [-1.1, .75, np.pi], home, 1)
    state = run(env, state, 45, [-1.1, b0[1], 0.], home, 1)
    state = run(env, state, 110, [-1.1, b0[1], 0.], HIGH, 1)
    base = np.array([b0[0] - .91, b0[1], 0.])
    state = run(env, state, 30, base, HIGH, 1)
    state = run(env, state, 45, base, LOW, 1)
    state = run(env, state, 25, base, LOW, 0)

    # Translate while initiating the lift, using the live tray/cube error.
    goal = base.copy()
    trace = []
    for k in range(up_steps):
        error = xyz(state, "cube1")[:2] - xyz(state, "bin_red")[:2]
        goal[:2] += np.clip(.75 * error, -.035, .035)
        action = act(state, goal, LIFT, 0)
        action[:2] = np.clip(action[:2], -.04, .04)
        state, reward, term, trunc, _ = env.step(action)
        trace.append(xyz(state, "bin_red").copy())

    # Descend promptly at the translated base pose; keep only millimetric XY
    # feedback so the hand does not impart a large lateral throw.
    for k in range(down_steps):
        error = xyz(state, "cube1")[:2] - xyz(state, "bin_red")[:2]
        goal[:2] += np.clip(.3 * error, -.006, .006)
        action = act(state, goal, LOW, 0)
        action[:2] = np.clip(action[:2], -.012, .012)
        state, reward, term, trunc, _ = env.step(action)
        trace.append(xyz(state, "bin_red").copy())
    state = run(env, state, 12, goal, LOW, 1)
    state = run(env, state, 25, goal, HIGH, 1)
    goal[0] -= .2
    state = run(env, state, 35, goal, HIGH, 1)
    state = run(env, state, 60, goal, HIGH, 1)

    bp, cp = xyz(state, "bin_red"), xyz(state, "cube1")
    arr = np.asarray(trace)
    print("BALLISTIC", up_steps, down_steps,
          "peak_z", round(float(arr[:, 2].max()), 5),
          "closest_flight", round(float(np.min(np.linalg.norm(arr[:, :2] - c0[:2], axis=1))), 5),
          "bin", np.round(bp, 5).tolist(), "cube", np.round(cp, 5).tolist(),
          "dxy", round(float(np.linalg.norm(bp[:2] - cp[:2])), 5),
          "dz", round(float(cp[2] - bp[2]), 5),
          "reward", reward, "term", term)
    env.close()


if __name__ == "__main__":
    trial(int(sys.argv[1]), int(sys.argv[2]))
