import time, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[4], eval_counts=[4])
import sys
n=int(sys.argv[1]) if len(sys.argv)>1 else 4
t=time.time()
s,info=env.reset(seed=0, options={'object_count': n})
print("reset", time.time()-t, info.keys() if isinstance(info,dict) else info)
print(env.action_space)
for name in sorted(s.get_object_names()):
    o=s.get_object_from_name(name)
    feats=env.type_features[o.type]
    print(name, o.type, {f: round(float(s.get(o,f)),3) for f in feats if not f.startswith(('v','w')) or o.type.name=='x'})
t=time.time()
for i in range(50):
    s,r,te,tr,_=env.step(env.action_space.sample()*0)
print("50 steps", time.time()-t, r)
