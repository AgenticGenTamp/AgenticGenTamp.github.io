from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import numpy as np
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:ScoopPour3DEnv",count_kwarg="num_objects",count_object_prefix="cube_",constant_object_env_kwargs={'scene_bg': False},design_counts=[10,20],eval_counts=[10,20])
print(env.action_space)
for seed in [0,1]:
    s,info=env.reset(seed=seed, options={'object_count':10})
    for n in sorted(s.get_object_names()):
        o=s.get_object_from_name(n)
        if o.type.name in ('mujoco_movable_object','mujoco_tidybot_robot'):
            print(n, {f: round(float(s.get(o,f)),3) for f in env.type_features[o.type]})
    print(info)
