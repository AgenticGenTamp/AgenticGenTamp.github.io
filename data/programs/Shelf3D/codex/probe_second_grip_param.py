"""Probe stronger/longer second-cube closure without editing approach.py."""

import sys

from approach import GeneratedApproach
from env_client import make_env


seed = int(sys.argv[1])
grip = float(sys.argv[2])
extra = int(sys.argv[3])
dq3 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
dq7 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
env = make_env()
state, info = env.reset(seed=seed, options={"object_count": 2})
policy = GeneratedApproach(env.action_space, env.observation_space,
                           env.make_primitives())
policy.reset(state, info)
extended = False
peak = 0.0
try:
    for step in range(1000):
        # Current second-cycle close hold is ten steps.  Delay its transition
        # once to test longer closure independently of the production policy.
        if (policy.index == 1 and policy.stage == 3 and not extended and
                policy.stage_step == 9):
            policy.stage_step -= extra
            extended = True
        action = policy.get_action(state)
        if policy.index == 1 and policy.stage in (1, 2, 3, 4):
            # Shift the effective joint targets while retaining the policy's
            # normal feedback. Action indices 5 and 9 command joints 3 and 7.
            action[5] += .35 * dq3
            action[9] += .35 * dq7
            action[3:10] = action[3:10].clip(-.1, .1)
        if policy.index == 1 and policy.stage in (3, 4, 5):
            action[10] = grip
        state, reward, term, trunc, _ = env.step(action)
        cube = state.get_object_from_name("cube2")
        peak = max(peak, float(state.get(cube, "z")))
        if term or trunc:
            break
    print(seed, grip, extra, dq3, dq7, "peak", round(peak, 4), "at", step + 1,
          "state", policy.index, policy.stage, policy.retry,
          "done", term, trunc)
finally:
    env.close()
