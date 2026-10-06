import sys, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
c=int(sys.argv[1]); s0=int(sys.argv[2]); s1=int(sys.argv[3])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.motion2d:Motion2DEnv", count_kwarg="num_passages", count_object_prefix="obstacle", bilevel_env_name="motion2d", design_counts=[c], eval_counts=[c])
tot=0; fails=[]; mt=0
for seed in range(s0,s1):
    s,info=env.reset(seed=seed, options={'object_count':c})
    a=GeneratedApproach(env.action_space, env.observation_space, {})
    t=time.time(); a.reset(s,info)
    for k in range(300+150*c):
        s,r,term,_,_=env.step(a.get_action(s))
        if term: break
    mt=max(mt,time.time()-t); tot+=k+1
    if not term or a.total_stuck: fails.append((seed,term,a.total_stuck))
print("count",c,"steps",tot,"maxtime",round(mt,2),"fails/stuck",fails,flush=True)
