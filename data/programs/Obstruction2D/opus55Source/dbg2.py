import sys, time
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import approach
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
c, seed = int(sys.argv[1]), int(sys.argv[2])
s,info=env.reset(seed=seed, options={'object_count':c})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
orig_move = ap._try_move
def tm(m, name, forbid, deadline, *a):
    picks = ap._picks(m, name)
    spots = ap._free_spots(m, name, forbid, *a[:1])
    print("try_move", name, "npicks", len(picks), "spots", [(round(x,3),round(y,3)) for x,y in spots[:5]], "robot", round(m.rx,3), round(m.ry,3))
    for k,b in sorted(m.blocks.items(), key=lambda kv: kv[1][0]): print("   ", k, [round(v,3) for v in b])
    r = orig_move(m, name, forbid, deadline, *a); print("  plan steps", len(r[0]) if r else None)
    print("  result", r is not None)
    return r
ap._try_move = tm
orig_t = ap._try_target
def tt(m, target, surface, deadline):
    r = orig_t(m, target, surface, deadline)
    print("try_target", r is not None, "npicks", len(ap._picks(m, target)))
    return r
ap._try_target = tt
ap.t0=time.time(); ap.fail_count=0
print("surface", ap._parse(s)[2]); plan, pred = ap._plan(s, 20)
print("plan len", len(plan))
