import sys, time
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
c, seed = int(sys.argv[1]), int(sys.argv[2])
s,info=env.reset(seed=seed, options={'object_count':c})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
t=time.time(); ap.reset(s,info); print("plan", len(ap.plan), time.time()-t)
for i in range(len(ap.plan)):
    a = ap.get_action(s)
    s,r,term,_,_ = env.step(a)
    act = ap._robot_xy(s); exp = ap.pred[ap.idx-1]
    if max(abs(x-y) for x,y in zip(act,exp))>1e-5:
        print("diverge at", i, "action", np.round(a,4), "exp", np.round(exp,4), "act", np.round(act,4))
        m,_,_,_ = ap._parse(s)
        for k,b in m.blocks.items(): print(k, np.round(b,4))
        break
    if term: print("success", i+1); break
