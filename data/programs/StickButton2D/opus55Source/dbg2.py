import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import *
c=int(sys.argv[1]); sd=int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",count_kwarg="num_buttons",count_object_prefix="button",bilevel_env_name="stickbutton2d",design_counts=[c],eval_counts=[c])
s,info=env.reset(seed=sd,options={'object_count':c})
ap=GeneratedApproach(env.action_space,None,{}); ap.reset(s,info)
r,st,b=ap._parse(s)
w=World([x['x'] for x in b],[x['y'] for x in b],(st['x'],st['y']))
p0=(r['x'],r['y']); pr=np.zeros(len(b),bool)
print(w.grasp_waypoints(p0))
for seq in [["G",0],[0],[0,"G"]]:
    print(seq, ap._sim(w,seq,p0,pr,False,0,GRASP_D,10))
