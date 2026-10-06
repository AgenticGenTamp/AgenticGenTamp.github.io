import sys, time
import numpy as np
import approach
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.packing3d:Packing3DEnv",count_kwarg="num_parts",count_object_prefix="part",design_counts=[1,2,3,4],eval_counts=[1,2,3,4])
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=int(sys.argv[2]), options={'object_count':int(sys.argv[1])})
ap._twin(int(sys.argv[1]))
orig=approach.GeneratedApproach._ik_from
stats={'ok':[0,0.0],'fail':[0,0.0]}
def wrapped(self,tp,q,pose):
    t=time.time(); r=orig(self,tp,q,pose); dt=time.time()-t
    k='ok' if r is not None else 'fail'; stats[k][0]+=1; stats[k][1]+=dt; return r
approach.GeneratedApproach._ik_from=wrapped
t=time.time(); ap.reset(s,info); print("plan",time.time()-t, len(ap._plan)); print(stats)
