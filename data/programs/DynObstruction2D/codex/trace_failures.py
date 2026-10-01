import sys
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def val(state, obj, feature):
    return float(state.get(obj, feature))


for seed in map(int, sys.argv[1:]):
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    last_phase = None
    for step in range(env.max_steps):
        block = state.get_objects(env.observation_space.get_type("target_block"))[0]
        robot = state.get_objects(env.observation_space.get_type("kin_robot"))[0]
        phase = policy.phase
        if phase != last_phase or step % 50 == 0:
            obs = []
            for obj in state.get_objects(env.observation_space.get_type("dyn_rectangle")):
                obs.append((getattr(obj, "name", str(obj)),
                            round(val(state, obj, "x"), 2),
                            round(val(state, obj, "y"), 2),
                            round(val(state, obj, "theta"), 2)))
            print(seed, step, phase,
                  "r", tuple(round(val(state, robot, f), 2)
                             for f in ("x", "y", "theta", "finger_gap")),
                  "b", tuple(round(val(state, block, f), 2)
                             for f in ("x", "y", "theta", "held")),
                  "o", obs, flush=True)
        action = np.asarray(policy.get_action(state), dtype=env.action_space.dtype)
        state, reward, terminated, truncated, info = env.step(action)
        last_phase = phase
        if terminated or truncated:
            print(seed, "END", step + 1, terminated, policy.phase, flush=True)
            break
    env.close()
