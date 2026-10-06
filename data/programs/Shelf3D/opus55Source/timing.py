import time, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
n=8
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Shelf3DEnv",count_kwarg="num_objects",count_object_prefix="cube",constant_object_env_kwargs={'scene_bg': False},design_counts=[n],eval_counts=[n])
s,info=env.reset(seed=5,options={'object_count':n})
t0=time.time(); ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info); print("init",time.time()-t0)
ta=te=0
for i in range(1500):
    t=time.time(); a=ap.get_action(s); ta+=time.time()-t
    t=time.time(); s,r,term,tr,info=env.step(a); te+=time.time()-t
    if term: break
print("steps",i+1,"policy",ta,"env",te)
