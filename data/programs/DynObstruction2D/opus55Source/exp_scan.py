from exp_common import *
import sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
for n in range(4):
    env = VariableObjectCountEnv(
        constant_object_env_path="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
        count_kwarg="num_obstructions", count_object_prefix="obstruction",
        design_counts=[n], eval_counts=[n])
    for seed in range(int(sys.argv[1])):
        s, _ = env.reset(seed=seed, options={"object_count": n})
        for nm, o in blocks(s).items():
            gl, gr = wall_gaps(o)
            if min(gl, gr) < 0.22:
                print(n, seed, nm, round(o['x'], 3), round(o['width'], 2), round(o['height'], 2), 'L' if gl < gr else 'R', round(min(gl, gr), 3), flush=True)
