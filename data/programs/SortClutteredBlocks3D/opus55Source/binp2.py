import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
n=int(sys.argv[1]); sd=int(sys.argv[2]); T=[int(x) for x in sys.argv[3].split(',')]
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False}, design_counts=[n], eval_counts=[n])
ap=GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=sd, options={'object_count': n}); ap.reset(s,info)
for i in range(max(T)+1):
    if i in T:
        print(i, [(b[4:], ap._pose(s,b)[:3].round(3), ap._pose(s,b)[3:].round(3)) for b in ["bin_red"]])
    s,*_=env.step(ap.get_action(s))
