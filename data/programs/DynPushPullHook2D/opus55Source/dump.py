import sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    s,_ = env.reset(seed=seed, options={'object_count': int(sys.argv[3])})
    print("seed",seed)
    for n in sorted(s.get_object_names()):
        o = s.get_object_from_name(n)
        f = env.type_features[o.type] if hasattr(env,'type_features') else []
        print(" ",n, {k: round(float(s.get(o,k)),3) for k in f if not k.startswith('color') and not k.startswith('v') and not k.startswith('omega')})
print(env.action_space)
