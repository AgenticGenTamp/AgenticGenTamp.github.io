"""Test slower/longer sweep schedules on canonical grasp families."""

import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def xy(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y")], float)


def modify(action, step, delay, mode, state):
    start = 194 + delay
    if step >= start:
        if mode == "half_long" and step <= 440 + delay:
            action[1], action[3:10], action[10] = .006, 0.0, 0.0
            action[5] = -.05
        elif mode == "slow_long" and step <= 500 + delay:
            action[1], action[3:10], action[10] = .004, 0.0, 0.0
            action[5] = -.035
        elif mode == "joint_slow" and step <= 380 + delay:
            action[3:10], action[10] = 0.0, 0.0
            action[1], action[5] = .012, -.05
        elif mode == "base_slow" and step <= 380 + delay:
            action[1], action[3:10], action[10] = .005, 0.0, 0.0
            action[5] = -.1
        elif mode == "pulse" and step <= 440 + delay:
            action[1], action[3:10], action[10] = .006, 0.0, 0.0
            action[5] = -.05 if ((step - start) % 20) < 10 else 0.0
        elif mode in ("dip03", "dip06") and step <= 380 + delay:
            # Preserve the productive joint-3 sweep while countering its
            # observed late upward arc with the same shoulder direction used
            # to lower the scoop during loading.
            scoop = state.get_object_from_name("scoop_0")
            z = float(state.get(scoop, "z"))
            action[4] = (0.03 if mode == "dip03" else 0.06) if z > .48 else 0.0
        elif mode in ("fast18", "fast25", "fast30", "fast34", "fast35",
                      "fast36", "fast37") and step <= 320 + delay:
            action[1] = {"fast18": .018, "fast25": .025,
                         "fast30": .030, "fast34": .034, "fast35": .035,
                         "fast36": .036, "fast37": .037}[mode]
    return action


def run(seed, count, mode):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": count})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube_"))
    initial = np.array([xy(state, n) for n in names])
    desired = initial + xy(state, "bin_green_0") - xy(state, "bin_yellow_0")
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    delay = 9 if policy.slow_carry_family else 0
    for step in range(1, 361):
        action = modify(policy.get_action(state), step, delay, mode, state)
        state, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    final = np.array([xy(state, n) for n in names])
    err = np.linalg.norm(final - desired, axis=1)
    print(seed, mode, "delay", delay, "disp", np.round((final-initial).mean(0), 4),
          "err", np.round([err.min(), err.mean(), err.max()], 4),
          "within", int((err < .05).sum()), "reward", reward, "done", term or trunc)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("modes", nargs="+", choices=("half_long", "slow_long",
                        "joint_slow", "base_slow", "pulse", "dip03", "dip06",
                        "fast18", "fast25", "fast30", "fast34", "fast35",
                        "fast36", "fast37"))
    args = parser.parse_args()
    for candidate in args.modes:
        run(args.seed, args.count, candidate)
