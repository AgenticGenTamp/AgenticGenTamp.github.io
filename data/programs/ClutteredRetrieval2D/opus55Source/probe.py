from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.clutteredretrieval2d:ClutteredRetrieval2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="clutteredretrieval2d",design_counts=[1,5,10],eval_counts=[1,5,10])
for c in [1,5,10]:
  s,_=env.reset(seed=0,options={'object_count':c})
  print("count",c)
  for n in sorted(s.get_object_names()):
    o=s.get_object_from_name(n)
    print(n,{f:round(float(s.get(o,f)),3) for f in env.type_features[o.type]})
print(env.observation_space, env.action_space)
