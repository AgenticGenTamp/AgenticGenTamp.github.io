import sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
import approach as A
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
c, seed = int(sys.argv[1]), int(sys.argv[2])
s,info=env.reset(seed=seed, options={'object_count':c})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
m,target,surface,obst=ap._parse(s)
print("surface", surface, "robot", m.rx, m.ry, m.aj)
for n,b in m.blocks.items(): print(n, b)
picks = ap._picks(m, target)
print("npicks", len(picks))
for c1, acts, m1, p in picks:
    for px in ap._place_xs_target(m1, target, surface):
        r = ap._place(m1, target, px, A.TABLE_TOP + A.PLACE_LIFT)
        print(px, r is not None, "robot goal x", px - m1.held[target][0])
