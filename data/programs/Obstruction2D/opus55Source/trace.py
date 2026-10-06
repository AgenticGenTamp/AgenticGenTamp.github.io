import sys, time
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
c, seed = int(sys.argv[1]), int(sys.argv[2])
s,info=env.reset(seed=seed, options={'object_count':c})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
m,target,surface,obst=ap._parse(s)
print("robot", round(m.rx,3), round(m.ry,3), "surface", [round(v,3) for v in surface])
for n,b in m.blocks.items(): print(n, [round(v,3) for v in b])
ap.reset(s, info)
# replay predicted states and print segments where held set changes
prev=None
for i,(a,p) in enumerate(zip(ap.plan, ap.pred)):
    h=tuple(sorted(p.held)) if hasattr(p,'held') else None
    if h!=prev:
        print(i, "held", h, "robot", round(p.rx,3), round(p.ry,3), round(p.aj,3), {n:[round(v,3) for v in p.blocks[n][:2]] for n in (prev or ())})
        prev=h
print("total", len(ap.plan))
