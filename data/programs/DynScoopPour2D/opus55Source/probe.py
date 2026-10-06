from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(
    constant_object_env_path="robocode.environments.generalized_dynamic2d:CountParameterizedDynScoopPour2DEnv",
    count_kwarg="num_small_objects", count_object_prefix="small_",
    design_counts=[1,2,3,4,5,6,8], eval_counts=[1,2,3,4,5,6,8,10])
import sys
for seed in range(3):
  s,_ = env.reset(seed=seed, options={'object_count': 5})
  for n in s.get_object_names():
    o = s.get_object_from_name(n)
    print(seed, n, o.type.name, {f: round(float(s.get(o,f)),3) for f in env.type_features[o.type] if not f.startswith(('color','v','omega'))})
print(env.action_space, getattr(env,'spec',None))
