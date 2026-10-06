import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[20], eval_counts=[20])
s,info=env.reset(seed=0, options={'object_count': 20})
for i in range(5): s,*_=env.step(np.zeros(11))
for n in sorted(s.get_object_names()):
    if n.startswith('cube'):
        o=s.get_object_from_name(n); print(n, [round(s.get(o,f),3) for f in ['x','y','z']])
