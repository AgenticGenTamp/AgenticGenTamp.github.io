import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
n=int(sys.argv[1])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False}, design_counts=[n], eval_counts=[n])
ap=GeneratedApproach(env.action_space, env.observation_space, {})
for sd in [int(x) for x in sys.argv[2].split(',')]:
    s,info=env.reset(seed=sd, options={'object_count': n}); ap.reset(s,info)
    print(sd, [(b[4:], ap._pose(s,b)[:3].round(3), round(np.degrees(2*np.arctan2(ap._pose(s,b)[6],ap._pose(s,b)[3])),1)) for b in ["bin_red","bin_green","bin_blue","bin_yellow"]])
