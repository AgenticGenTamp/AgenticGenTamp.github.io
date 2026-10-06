from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",count_kwarg="num_buttons",count_object_prefix="button",bilevel_env_name="stickbutton2d",design_counts=[1,2,3],eval_counts=[5])
s,i=env.reset(seed=0,options={'object_count':3})
for n in s.get_object_names():
    o=s.get_object_from_name(n); print(n,o.type, type(o.type), o.type.name)
print(env.observation_space, env.action_space)
