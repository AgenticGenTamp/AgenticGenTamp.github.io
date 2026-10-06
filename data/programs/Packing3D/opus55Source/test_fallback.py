import time, numpy as np
import approach
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.packing3d:Packing3DEnv",count_kwarg="num_parts",count_object_prefix="part",design_counts=[1,2,3,4],eval_counts=[1,2,3,4])
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
orig=ap._fast_plan
def limited(*a, **k):
    ap._t0 -= 22  # pretend we've used time so fast plan bails midway
    return orig(*a, **k)
ap._fast_plan=limited
s,info=env.reset(seed=3, options={'object_count':4})
t=time.time(); ap.reset(s,info); print("plan", len(ap._plan), time.time()-t)
for i in range(900):
    s,r,term,trunc,_=env.step(ap.get_action(s))
    if term: break
print("term", term, i+1)
