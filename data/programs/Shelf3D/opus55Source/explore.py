import numpy as np, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Shelf3DEnv",count_kwarg="num_objects",count_object_prefix="cube",constant_object_env_kwargs={'scene_bg': False},design_counts=[1,2,3],eval_counts=[1,2,3])
print(env.action_space)
for n in [1,3,8]:
  for seed in [0,1]:
    s,info=env.reset(seed=seed,options={'object_count':n})
    print("n",n,"seed",seed, info.keys() if isinstance(info,dict) else info)
    for name in sorted(s.get_object_names()):
        o=s.get_object_from_name(name)
        print(" ",name,o.type, [ (f,round(float(s.get(o,f)),3)) for f in env.type_features[o.type][:16] if not f.startswith('v')])
t=time.time()
for i in range(50): env.step(env.action_space.sample())
print("step time",(time.time()-t)/50)
