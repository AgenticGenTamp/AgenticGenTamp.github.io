"""Diagnostic-only A/B test for a broader blocker clearance classifier."""
import sys

import numpy as np

from env_client import make_env


source = open("approach.py", encoding="utf-8").read()
source = source.replace(
    "if not wide_low and (len(obstructions) >= 2 or target_overlaps):",
    "if not wide_low and len(obstructions) >= 1:")
source = source.replace(
    "if self._overlap_depth(ox, ow, sx, sw) > .04:",
    "if self._overlap_depth(ox, ow, sx, sw + bw) > .04:")
namespace = {}
exec(compile(source, "approach_classifier_variant", "exec"), namespace)
Policy = namespace["GeneratedApproach"]

for seed in map(int, sys.argv[1:]):
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = Policy(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    term = trunc = False
    prior = None
    transitions = []
    for step in range(env.max_steps):
        action = policy.get_action(state)
        if policy.phase != prior:
            transitions.append((step, policy.phase, policy.clear_name,
                                sorted(policy.cleared)))
            prior = policy.phase
        state, _, term, trunc, info = env.step(
            np.asarray(action, dtype=env.action_space.dtype))
        if term or trunc:
            break
    block = state.get_objects(env.observation_space.get_type("target_block"))[0]
    get = lambda f: round(float(state.get(block, f)), 3)
    print(seed, term, trunc, step + 1, "block",
          (get("x"), get("y"), get("theta")), "transitions", transitions,
          flush=True)
    env.close()
