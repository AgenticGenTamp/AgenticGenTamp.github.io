import sys
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach

n, seed = int(sys.argv[1]), int(sys.argv[2])
every = int(sys.argv[3]) if len(sys.argv) > 3 else 10
env = VariableObjectCountEnv(
    constant_object_env_path="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction",
    design_counts=[n], eval_counts=[n])
state, info = env.reset(seed=seed, options={"object_count": n})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(state, info)


def desc(state):
    r, b, s, obs = ap._parse(state)
    out = f"R({r['x']:.3f},{r['y']:.3f},th={r['theta']:.3f},L={r['arm_joint']:.3f},g={r['finger_gap']:.3f}) "
    out += f"B({b['x']:.3f},{b['y']:.3f},th={b['theta']:.2f},w={b['width']:.2f},h={b['height']:.2f}) "
    out += f"S({s['x']:.3f},w={s['width']:.3f}) "
    out += " ".join(f"O({o['x']:.2f},{o['y']:.2f},th={o['theta']:.2f},w={o['width']:.2f},h={o['height']:.2f})" for o in obs)
    return out


print(desc(state))
for t in range(300 + 150 * n):
    a = ap.get_action(state)
    state, rew, term, trunc, info = env.step(a)
    if t % every == 0 or term or ap.phase != getattr(ap, '_lp', None):
        ap._lp = ap.phase
        print(t, ap.phase, np.round(a, 3), desc(state))
    if term:
        print("SUCCESS", t)
        break
