"""Targeted seed-2 sweep perturbations; never imported by the submission."""

import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def xy(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y")], float)


def run(spec):
    # spec: speed,yaw,x,start,end. Times are absolute environment steps.
    speed, yaw, dx, start, end = map(float, spec.split(","))
    start, end = int(start), int(end)
    env = make_env()
    state, info = env.reset(seed=2, options={"object_count": 20})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube_"))
    initial = np.array([xy(state, n) for n in names])
    desired = initial + xy(state, "bin_green_0") - xy(state, "bin_yellow_0")
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    terminated = truncated = False
    snapshots = {}
    for step in range(1, 361):
        action = policy.get_action(state)
        if start <= step <= end:
            action[1] = speed
            action[2] += yaw
            action[0] += dx
        state, reward, terminated, truncated, _ = env.step(action)
        if step in (215, 240, 260, 280, 300, 320, 340, 360):
            now = np.array([xy(state, n) for n in names])
            errors = np.linalg.norm(now - desired, axis=1)
            snapshots[step] = (int((errors < .05).sum()), float(errors.mean()))
        if terminated or truncated:
            break
    final = np.array([xy(state, n) for n in names])
    err = np.linalg.norm(final - desired, axis=1)
    worst = np.argsort(err)[-4:]
    print(spec, "done", terminated or truncated, "step", step,
          "within", int((err < .05).sum()), "mean/max", np.round([err.mean(), err.max()], 4),
          "worst", [(names[i], round(float(err[i]), 4),
                     np.round(final[i] - initial[i], 3).tolist()) for i in worst],
          "timeline", snapshots)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+")
    args = parser.parse_args()
    for candidate in args.specs:
        run(candidate)
