import sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
c=int(sys.argv[1]); seeds=[int(x) for x in sys.argv[2].split(',')]
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",count_kwarg="num_buttons",count_object_prefix="button",bilevel_env_name="stickbutton2d",design_counts=[c],eval_counts=[c])
for sd in seeds:
    s,info=env.reset(seed=sd,options={'object_count':c})
    ap=GeneratedApproach(env.action_space,None,{}); ap.reset(s,info)
    for t in range(300+150*c):
        s,r,term,_,_=env.step(ap.get_action(s))
        if term: break
    print(sd,term,t+1,flush=True)
