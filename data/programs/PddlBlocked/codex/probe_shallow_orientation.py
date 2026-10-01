"""Diagnostics and green-grasp overrides for shallow east-tangent seeds."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import sys

SEEDS = (101, 191, 584, 628, 760)


def run(seed):
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    prior = None
    visits = {}
    for step in range(env.max_steps):
        stage = policy.stage
        visits[stage] = visits.get(stage, 0) + 1
        if stage != prior:
            print(seed, "step", step, "stage", stage, "base", policy.robot(state).round(4),
                  "theta", round(policy.theta, 4), "held", policy.g(state, "robot", "grasp_active"),
                  "q", [round(policy.g(state, "robot", "joint_" + str(j)), 3) for j in range(1, 8)])
            prior = stage
        state, _, term, trunc, _ = env.step(policy.get_action(state))
        if term or trunc:
            print(seed, "DONE", term, step + 1, "stage", policy.stage, "visits", visits)
            break
    env.close()


for seed in (tuple(map(int, sys.argv[1:])) or SEEDS):
    run(seed)
