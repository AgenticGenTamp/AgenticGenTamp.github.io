"""Check reset object_count overrides without touching the submitted policy."""

import numpy as np

from env_client import make_env


for requested in (0, 1, 2, 3, 4, 5, 6, 8, 10):
    env = make_env()
    try:
        state, info = env.reset(seed=123, options={"object_count": requested})
        cubes = sorted(n for n in state.get_object_names() if n.startswith("cube"))
        positions = {}
        for name in cubes:
            obj = state.get_object_from_name(name)
            positions[name] = tuple(round(float(state.get(obj, f)), 4)
                                    for f in ("x", "y", "z"))
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        _, reward, terminated, truncated, step_info = env.step(action)
        print(requested, cubes, positions, info, "step", reward,
              terminated, truncated, step_info)
    except Exception as exc:
        print(requested, type(exc).__name__, str(exc))
    finally:
        env.close()
