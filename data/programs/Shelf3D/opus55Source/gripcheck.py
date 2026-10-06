import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Shelf3DEnv",count_kwarg="num_objects",count_object_prefix="cube",constant_object_env_kwargs={'scene_bg': False},design_counts=[1],eval_counts=[1])
s,info=env.reset(seed=0,options={'object_count':1})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
r=s.get_object_from_name('robot')
prev=None
for i in range(200):
    a=ap.get_action(s); s,*_=env.step(a)
    if ap.phase in ('grasp','release','retreat') or prev in ('grasp','release'):
        print(i,ap.phase,ap.timer,'vel_grip',round(s.get(r,'vel_gripper'),3),'cube',ap._cube(s,ap.target).round(3))
    prev=ap.phase
