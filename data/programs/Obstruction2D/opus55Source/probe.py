from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
s,_=env.reset(seed=0, options={'object_count':3})
for n in s.get_object_names():
    o=s.get_object_from_name(n)
    print(n, {f: round(float(s.get(o,f)),3) for f in env.type_features[o.type]})
print(env.action_space)
