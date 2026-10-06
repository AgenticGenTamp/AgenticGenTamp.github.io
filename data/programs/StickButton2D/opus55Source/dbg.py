import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
c=int(sys.argv[1]); sd=int(sys.argv[2]); every=int(sys.argv[3]) if len(sys.argv)>3 else 50
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",count_kwarg="num_buttons",count_object_prefix="button",bilevel_env_name="stickbutton2d",design_counts=[c],eval_counts=[c])
s,info=env.reset(seed=sd,options={'object_count':c})
ap=GeneratedApproach(env.action_space,None,{}); ap.reset(s,info)
r,st,b=ap._parse(s); print('stick',st); [print(x) for x in b]
for t in range(300+150*c):
    a=ap.get_action(s)
    s,_,term,_,_=env.step(a)
    if t%every==0 or term:
        r,st,b=ap._parse(s)
        print(t,{k:round(v,3) for k,v in r.items()},{k:round(v,3) for k,v in st.items()},ap.holding,ap.plan,ap.phase,ap.cur_tgt,np.round(a,3),sum(x['pressed'] for x in b))
    if term: break
