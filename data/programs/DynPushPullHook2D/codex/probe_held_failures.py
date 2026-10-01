"""Trace held-hook failures and optionally apply a simple phase correction."""
import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def val(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def run(seed, mode="base", limit=500):
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    target = state.get_object_from_name("target_block")
    hook = state.get_object_from_name("hook")
    print("init", seed,
          "target", *(round(val(state, "target_block", f), 3)
                      for f in ("x", "y", "width", "height")),
          "hook", *(round(val(state, "hook", f), 3)
                    for f in ("x", "y", "theta", "length_side1", "length_side2")),
          "route", policy.right_route)
    prior = None
    for step in range(1, min(limit, env.max_steps) + 1):
        action = policy.get_action(state)
        # Diagnose two small alternatives without changing approach.py.
        if mode == "lower_engage" and policy.phase == "engage":
            action[1] = -.035
        elif mode == "pull_left" and policy.phase == "pull":
            action[0] = -.035
        state, _, term, trunc, _ = env.step(action)
        phase = policy.phase
        if phase != prior or step % 25 == 0 or term or trunc:
            print(step, phase, "held", int(val(state, "hook", "held") > .5),
                  "r", tuple(round(val(state, "robot", f), 3)
                             for f in ("x", "y", "theta", "arm_length")),
                  "h", tuple(round(val(state, "hook", f), 3)
                             for f in ("x", "y", "theta")),
                  "t", tuple(round(val(state, "target_block", f), 3)
                             for f in ("x", "y", "theta")))
            prior = phase
        if term or trunc:
            break
    env.close()
    print("result", seed, mode, step, term, trunc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--mode", choices=("base", "lower_engage", "pull_left"),
                        default="base")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    run(args.seed, args.mode, args.limit)
