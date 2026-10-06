import sys, time
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
import approach as A
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
c, seed = int(sys.argv[1]), int(sys.argv[2])
s,info=env.reset(seed=seed, options={'object_count':c})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
m,target,surface,obst=ap._parse(s)
print("surface", surface)
# remove all obstructions from model to isolate
for n in obst: del m.blocks[n]
picks = ap._picks(m, target)
for c1, acts, m1, p in picks:
    print("pick cost", c1, "held", m1.held, "robot", m1.rx, m1.ry, m1.aj)
    for px in ap._place_xs_target(m1, target, surface):
        ox, oy = m1.held[target]
        for aj in (0.2, 0.15, 0.1):
            rx = px - ox; ry = A.TABLE_TOP + A.PLACE_LIFT - oy + aj
            hb = ap._held_blocks_at(m1, rx, ry, aj)
            print(" px", round(px,3), aj, "goal coll", m1.collides(rx, ry, aj, hb, m1.held.keys()), "travel", ap._travel(m1, rx, ry, aj, 1.0, 0.0) is not None)
    break
