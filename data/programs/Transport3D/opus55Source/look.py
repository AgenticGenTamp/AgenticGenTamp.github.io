from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(
    constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube",
    constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1,2,3], eval_counts=[1,2,3,4])
import sys
for seed in [0,1]:
    s, info = env.reset(seed=seed, options={'object_count': 2})
    for name in sorted(s.get_object_names()):
        o = s.get_object_from_name(name)
        print(name, o.type, {f: round(float(s.get(o,f)),3) for f in env.type_features[o.type]})
print(env.action_space)
print(type(env), getattr(env,'spec',None))
