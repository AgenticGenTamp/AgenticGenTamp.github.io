import math
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
for c in [1,4,8]:
  env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.motion2d:Motion2DEnv", count_kwarg="num_passages", count_object_prefix="obstacle", bilevel_env_name="motion2d", design_counts=[c], eval_counts=[c])
  for seed in range(4):
    s,info=env.reset(seed=seed, options={'object_count':c})
    a=GeneratedApproach(env.action_space, env.observation_space, {}); a.reset(s,info)
    P=a.path; L=sum(max(abs(P[i+1][0]-P[i][0]),abs(P[i+1][1]-P[i][1])) for i in range(len(P)-1))
    sw=sum(1 for i in range(len(P)-1) if P[i+1][2]!=P[i][2])
    for k in range(2000):
        s,r,term,_,_=env.step(a.get_action(s))
        if term: break
    print(c,seed,"steps",k+1,"cheb/0.05",round(L/0.05,1),"switches",sw)
