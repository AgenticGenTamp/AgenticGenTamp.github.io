import time, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SweepSimple3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[1,5,10,50], eval_counts=[1,5,10,50])
for n in [1,50]:
    t=time.time(); s,_=env.reset(seed=0, options={'object_count':n}); print('reset',n,time.time()-t)
    t=time.time()
    for i in range(50): env.step(np.zeros(11,dtype=np.float32))
    print('step',(time.time()-t)/50)
