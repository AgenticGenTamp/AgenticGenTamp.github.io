from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import sys
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv",count_kwarg="num_blocks",count_object_prefix="block",bilevel_env_name="clutteredstorage2d",design_counts=[1,2,3],eval_counts=[1,2,3])
for seed,n in [(0,1),(0,3),(1,7),(2,10)]:
    s,_=env.reset(seed=seed,options={'object_count':n})
    print("seed",seed,"n",n)
    for name in sorted(s.get_object_names()):
        o=s.get_object_from_name(name)
        print(" ",name,o.type.name,{f:round(float(s.get(o,f)),3) for f in env.type_features[o.type]})
print(env.action_space)
