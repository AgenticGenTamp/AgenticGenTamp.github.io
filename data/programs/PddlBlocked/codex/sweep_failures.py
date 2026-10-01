"""Classify approach failures across seeds without modifying the policy.

Usage: /opt/robocode-strict/bin/python sweep_failures.py [start [stop]]
"""
import sys

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def feature(state, name, key):
    return float(state.get(state.get_object_from_name(name), key))


start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stop = int(sys.argv[2]) if len(sys.argv) > 2 else start + 100
env = make_env()
wins = 0
for seed in range(start, stop):
    state, info = env.reset(seed=seed)
    names = state.get_object_names()
    spares = sorted(n for n in names if n.startswith("green") and n != "green0")
    green = np.array([feature(state, "green0", "pose_x"), feature(state, "green0", "pose_y")])
    blocker = np.array([feature(state, "blocker", "pose_x"), feature(state, "blocker", "pose_y")])
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    initial_stage = policy.stage
    target = policy.target(policy.spare if initial_stage == -5 else policy.blocker)
    last_base = None
    unchanged = 0
    for step in range(env.max_steps):
        before = policy.robot(state)
        action = policy.get_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        after = policy.robot(state)
        unchanged = unchanged + 1 if np.max(np.abs(after - before)) < 1e-7 else 0
        last_base = after
        if terminated or truncated:
            break
    wins += int(terminated)
    if not terminated:
        print(
            f"seed={seed:4d} nsp={len(spares)} mode={'spare' if initial_stage == -5 else 'local':5s} "
            f"g=({green[0]:.3f},{green[1]:.3f}) b=({blocker[0]:.3f},{blocker[1]:.3f}) "
            f"out=({policy.out[0]:+.3f},{policy.out[1]:+.3f}) target=({target[0]:.3f},{target[1]:.3f}) "
            f"stage={policy.stage:2d} held={feature(state, 'robot', 'grasp_active'):.0f} "
            f"base=({last_base[0]:.3f},{last_base[1]:.3f}) still={unchanged}"
        )
print(f"wins={wins}/{stop-start}")
env.close()
