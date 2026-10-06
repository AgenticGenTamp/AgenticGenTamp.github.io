import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
n,seed,name,s0,s1=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Shelf3DEnv",count_kwarg="num_objects",count_object_prefix="cube",constant_object_env_kwargs={'scene_bg': False},design_counts=[n],eval_counts=[n])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
c=s.get_object_from_name(name)
for i in range(s1):
    a=ap.get_action(s); s,r,term,tr,info=env.step(a)
    if i>=s0:
        b,q=ap._robot(s)
        print(i,ap.phase,"cube",ap._cube(s,c).round(3),"quat",np.round([s.get(c,f) for f in ('qw','qx','qy','qz')],2),"base",b.round(3),"a",np.round(a,2)[[0,1,2,10]], "qerr", np.round(q-ap.cfg['qa'],3)[[1,3,5]])
