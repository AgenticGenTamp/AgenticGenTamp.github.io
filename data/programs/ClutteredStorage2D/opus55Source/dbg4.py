import sys, time, math
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import *
n, seed = int(sys.argv[1]), int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv",count_kwarg="num_blocks",count_object_prefix="block",bilevel_env_name="clutteredstorage2d",design_counts=[n],eval_counts=[n])
s, info = env.reset(seed=seed, options={'object_count': n})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(s, info)
orig = ap._make_plan
def mp():
    w = ap.world
    p = orig()
    if p:
        k = next((i for i,(q,v) in enumerate(p) if v>0.5), None)
        tgt = ap._last_task if hasattr(ap,'_last_task') else None
        print(f"step {step} q={tuple(round(v,2) for v in w.q)} total={len(p)} pick={k} trans={len(p)-(k or 0)} end={tuple(round(v,2) for v in p[-1][0])}")
    return p
ap._make_plan = mp
for step in range(300+150*n):
    a = ap.get_action(s)
    s, r, term, trunc, info = env.step(a)
    if term: print("done", step+1); break
w=ap.world
