import sys, time, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
c=int(sys.argv[1])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",count_kwarg="num_buttons",count_object_prefix="button",bilevel_env_name="stickbutton2d",design_counts=[c],eval_counts=[c])
for sd in range(3):
    s,info=env.reset(seed=sd,options={'object_count':c})
    t0=time.time(); ta=0
    ap=GeneratedApproach(env.action_space,None,{}); ap.reset(s,info)
    for t in range(300+150*c):
        a0=time.time(); a=ap.get_action(s); ta+=time.time()-a0
        s,r,term,_,_=env.step(a)
        if term: break
    print(c,sd,term,t+1,'total',round(time.time()-t0,1),'agent',round(ta,1),flush=True)
