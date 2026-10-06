import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach, _yaw_from_quat
n=20
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False}, design_counts=[n], eval_counts=[n])
s,info=env.reset(seed=int(sys.argv[1]), options={'object_count': n})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s,info)
for c in ap.cubes:
    p=ap._pose(s,c); print(c, p[:3].round(3), round(np.degrees(_yaw_from_quat(*p[3:7]))))
