from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Tossing3DEnv", count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False}, design_counts=[1,2], eval_counts=[1,2])
print(env.action_space)
for n in [1,2]:
  s,info=env.reset(seed=0, options={'object_count':n})
  for name in s.get_object_names():
    o=s.get_object_from_name(name)
    print(name, {f: round(float(s.get(o,f)),3) for f in env.type_features[o.type]})
  print(info)
