from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.packing3d:Packing3DEnv",count_kwarg="num_parts",count_object_prefix="part",design_counts=[1,2,3],eval_counts=[1,2,3])
for n in [3,4]:
    s,info = env.reset(seed=0, options={'object_count':n})
    print("N",n)
    for name in sorted(s.get_object_names()):
        o=s.get_object_from_name(name)
        print(name,o.type, {f:round(float(s.get(o,f)),3) for f in env.type_features[o.type]})
print(env.action_space)
