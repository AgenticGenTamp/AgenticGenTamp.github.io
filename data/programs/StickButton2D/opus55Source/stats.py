import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import importlib,os; GeneratedApproach=importlib.import_module(os.environ.get("APPROACH","approach")).GeneratedApproach
# GeneratedApproach
c=int(sys.argv[1]); n=int(sys.argv[2]); off=int(sys.argv[3]) if len(sys.argv)>3 else 0
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",count_kwarg="num_buttons",count_object_prefix="button",bilevel_env_name="stickbutton2d",design_counts=[c],eval_counts=[c])
T=[];G=[];F=0
for sd in range(off,off+n):
    s,info=env.reset(seed=sd,options={'object_count':c})
    ap=GeneratedApproach(env.action_space,None,{}); ap.reset(s,info); g=None
    for t in range(300+150*c):
        s,r,term,_,_=env.step(ap.get_action(s))
        if g is None and ap.holding: g=t
        if term: break
    T.append(t+1)
    if not term: print("FAIL",sd,flush=True); G.append(g if g is not None else -1); F+= not term
print(c,'fails',F,'mean steps',np.mean(T),'mean grasp t',np.mean(G))
