"""Correlate rod orientation with displacement under compact push variants."""
import math
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def rod_data(state, space):
    typ = space.get_type("mujoco_movable_object")
    out = []
    for obj in state.get_objects(typ):
        v = [float(state.get(obj, f)) for f in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        w, x, y, z = v[3:]
        # World direction of the long local-y axis.
        axis = np.array([2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)])
        out.append((obj, np.array(v[:3]), np.array(v[3:]), axis))
    return out


def current(seed):
    env = make_env()
    s, info = env.reset(seed=seed, options={"object_count": 1})
    initial = rod_data(s, env.observation_space)[0]
    p = GeneratedApproach(env.action_space, env.observation_space, {})
    p.reset(s, info)
    best = -1e9
    term = False
    for t in range(env.max_steps):
        s, r, term, trunc, _ = env.step(p.get_action(s))
        best = max(best, float(r))
        if term or trunc:
            break
    final = rod_data(s, env.observation_space)[0]
    env.close()
    print("current", seed, "q", np.round(initial[2], 3), "axis", np.round(initial[3], 3),
          "p0", np.round(initial[1], 3), "p1", np.round(final[1], 3),
          "d", np.round(final[1]-initial[1], 3), "best", best, "term", term)


if __name__ == "__main__":
    for seed in map(int, sys.argv[1:] or range(6)):
        current(seed)
