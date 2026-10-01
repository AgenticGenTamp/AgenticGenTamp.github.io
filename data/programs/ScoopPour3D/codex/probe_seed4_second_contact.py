"""Probe late corrective contacts on the strong seed-4/count-20 rollout."""

import argparse
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def override(action, step, mode, start):
    """Return a copied action with a compact, explicitly timed correction."""
    a = action.copy()
    if mode == "baseline":
        return a
    # Pull the currently saturated proximal joint back, then re-engage it.
    if mode.startswith("j1cycle"):
        width = int(mode[len("j1cycle"):])
        if start <= step < start + width:
            a[3] = 0.1
        elif start + width <= step < start + 2 * width:
            a[3] = -0.1
    elif mode.startswith("j3cycle"):
        width = int(mode[len("j3cycle"):])
        if start <= step < start + width:
            a[5] = 0.1
        elif start + width <= step < start + 2 * width:
            a[5] = -0.1
    elif mode.startswith("elbowcycle"):
        width = int(mode[len("elbowcycle"):])
        if start <= step < start + width:
            a[6] = -0.1
        elif start + width <= step < start + 2 * width:
            a[6] = 0.1
    elif mode == "yawcycle":
        if 260 <= step < 275:
            a[2], a[3] = 0.008, 0.0
        elif 275 <= step < 290:
            a[2], a[3] = -0.008, -0.1
    elif mode == "xcycle":
        if 260 <= step < 275:
            a[0], a[3] = -0.01, 0.0
        elif 275 <= step < 290:
            a[0], a[3] = 0.01, -0.1
    elif mode == "ypulse":
        if 260 <= step < 280:
            a[1] = 0.025
    elif mode == "yhalf":
        if 260 <= step < 285:
            a[1] = 0.006
    elif mode == "yzero":
        if 260 <= step < 285:
            a[1] = 0.0
    elif mode == "xtrim":
        if 260 <= step < 285:
            a[0] = -0.001
    return a


def run(mode, steps, start):
    env = make_env()
    state, info = env.reset(seed=4, options={"object_count": 20})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    names = sorted(n for n in state.get_object_names() if n.startswith("cube_"))
    initial = np.array([[state.get(state.get_object_from_name(n), f)
                         for f in ("x", "y")] for n in names])
    checkpoints = {}
    reward = 0.0
    for step in range(1, steps + 1):
        action = override(policy.get_action(state), step, mode, start)
        state, reward, terminated, truncated, info = env.step(action)
        if step in (250, 260, 280, 300, 330, 350) or terminated or truncated:
            now = np.array([[state.get(state.get_object_from_name(n), f)
                             for f in ("x", "y")] for n in names])
            err = np.linalg.norm(now - (initial + [0.0, 0.4]), axis=1)
            checkpoints[step] = (int((err < .05).sum()), float(err.mean()),
                                 float(err.max()))
        if terminated or truncated:
            break
    now = np.array([[state.get(state.get_object_from_name(n), f)
                     for f in ("x", "y")] for n in names])
    delta = now - (initial + [0.0, 0.4])
    err = np.linalg.norm(delta, axis=1)
    print(mode, "step", step, "term", terminated, "reward", reward,
          "near", int((err < .05).sum()), "mean/max", np.round(
              [err.mean(), err.max()], 5), "checkpoints", checkpoints,
          "outliers", [(names[i], round(float(err[i]), 4),
                         np.round(delta[i], 3).tolist())
                        for i in np.argsort(err)[-6:]])
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--start", type=int, default=260)
    args = parser.parse_args()
    run(args.mode, args.steps, args.start)
