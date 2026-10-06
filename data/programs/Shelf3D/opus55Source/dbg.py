import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
n,seed=int(sys.argv[1]),int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Shelf3DEnv",count_kwarg="num_objects",count_object_prefix="cube",constant_object_env_kwargs={'scene_bg': False},design_counts=[n],eval_counts=[n])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
print("robot",ap._robot(s)[0].round(2))
for c in ap.cubes: print(c.name, ap._cube(s,c).round(3))
last=None
for i in range(300+150*n):
    a=ap.get_action(s); s,r,term,tr,info=env.step(a)
    key=(ap.phase,getattr(ap.target,'name',None))
    if key!=last:
        print(i,key,"slot",ap.slot,"base",ap._robot(s)[0].round(2),"tgt",ap._cube(s,ap.target).round(3) if ap.target is not None else None); last=key
    if term: print("done",i);break
for c in ap.cubes: print(c.name, ap._cube(s,c).round(3), ap._in_goal(ap._cube(s,c)))
