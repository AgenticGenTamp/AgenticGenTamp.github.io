import sys, time, os
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
n=20; sd=int(sys.argv[1])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False}, design_counts=[n], eval_counts=[n])
s,info=env.reset(seed=sd, options={'object_count': n})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s,info)
ta=te=0; mx=0
for i in range(800):
    t=time.time(); a=ap.get_action(s); dt=time.time()-t; ta+=dt; mx=max(mx,dt)
    t=time.time(); s,r,term,*_=env.step(a); te+=time.time()-t
    if term: break
print('steps',i+1,'agent',round(ta,1),'maxagent',round(mx,3),'env',round(te,1))
