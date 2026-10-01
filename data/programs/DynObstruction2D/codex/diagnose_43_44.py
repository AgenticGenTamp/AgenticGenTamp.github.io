import sys

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def val(state, obj, feature):
    return float(state.get(obj, feature))


args = [int(x) for x in sys.argv[1:]]
limit = 450
if args and args[0] < 0:
    limit = -args.pop(0)

for seed in args:
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    robot_t = env.observation_space.get_type("kin_robot")
    block_t = env.observation_space.get_type("target_block")
    surf_t = env.observation_space.get_type("target_surface")
    dyn_t = env.observation_space.get_type("dyn_rectangle")
    last_phase = None
    terminated = truncated = False
    for step in range(min(env.max_steps, limit)):
        robot = state.get_objects(robot_t)[0]
        block = state.get_objects(block_t)[0]
        surf = state.get_objects(surf_t)[0]
        action = policy.get_action(state)
        changed = policy.phase != last_phase
        if changed or step % 20 == 0:
            obs = []
            for obj in state.get_objects(dyn_t):
                obs.append((obj.name, round(val(state, obj, "x"), 3),
                            round(val(state, obj, "y"), 3),
                            round(val(state, obj, "theta"), 2),
                            round(val(state, obj, "vx"), 2),
                            round(val(state, obj, "vy"), 2)))
            print(seed, step, policy.phase, "r",
                  tuple(round(val(state, robot, f), 3)
                        for f in ("x", "y", "theta", "arm_length", "finger_gap")),
                  "b", tuple(round(val(state, block, f), 3)
                             for f in ("x", "y", "theta", "vx", "vy", "held", "width", "height")),
                  "q", tuple(round(val(state, surf, f), 3)
                             for f in ("x", "y", "width", "height")),
                  "obs", obs, "a", np.round(action, 3).tolist(), flush=True)
        last_phase = policy.phase
        state, reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=env.action_space.dtype))
        if terminated or truncated:
            break
    print("RESULT", seed, terminated, truncated, step + 1, info, flush=True)
    env.close()
