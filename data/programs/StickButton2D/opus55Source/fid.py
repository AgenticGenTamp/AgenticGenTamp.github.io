import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
c=int(sys.argv[1])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",count_kwarg="num_buttons",count_object_prefix="button",bilevel_env_name="stickbutton2d",design_counts=[c],eval_counts=[c])
for sd in range(1000,1012):
    s,info=env.reset(seed=sd,options={'object_count':c})
    ap=GeneratedApproach(env.action_space,None,{}); ap.reset(s,info); pc=None; gt=None
    for t in range(300+150*c):
        a=ap.get_action(s)
        if pc is None and ap.plan is not None: pc=ap.plan_cost; t0=t; pl=list(ap.plan)
        if gt is None and ap.holding: gt=t
        s,r,term,_,_=env.step(a)
        if term: break
    print(sd,'pred',pc,'from',t0,'actual',t+1,'grasp at',gt, pl)
