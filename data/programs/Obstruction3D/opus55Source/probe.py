from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv", count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[1,2,3], eval_counts=[1,2,3])
s, info = env.reset(seed=0, options={'object_count': 3})
print(env.action_space)
for n in sorted(s.get_object_names()):
    o = s.get_object_from_name(n)
    print(n, o.type, {f: round(float(s.get(o,f)),3) for f in env.type_features[o.type]})
