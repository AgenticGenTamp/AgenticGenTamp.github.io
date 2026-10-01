"""Focused grasp-acquisition probe for a seed range."""

import argparse

from approach import GeneratedApproach
from env_client import make_env


def run(seed, limit):
    env = make_env()
    state, info = env.reset(seed=seed)
    hook = state.get_object_from_name("hook")
    initial = tuple(float(state.get(hook, k)) for k in ("x", "y", "theta"))
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    selectors = [name for name in ("extended_grasp", "deep_grasp",
                                    "deep_grasp2", "fast_grasp", "left_entry")
                 if getattr(policy, name)]
    held_at = None
    for step in range(1, limit + 1):
        state, _, terminated, truncated, _ = env.step(policy.get_action(state))
        hook = state.get_object_from_name("hook")
        if float(state.get(hook, "held")) > .5:
            held_at = step
            break
        if terminated or truncated:
            break
    env.close()
    return (seed,) + tuple(round(v, 4) for v in initial) + \
        ("+".join(selectors) or "default", held_at, policy.attempt, policy.phase)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=50)
    parser.add_argument("--end", type=int, default=79)
    parser.add_argument("--limit", type=int, default=180)
    args = parser.parse_args()
    for seed in range(args.start, args.end + 1):
        print(run(seed, args.limit), flush=True)
