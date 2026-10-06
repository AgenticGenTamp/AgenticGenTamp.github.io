import sys, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import approach
from approach import GeneratedApproach
n, seed = int(sys.argv[1]), int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv",count_kwarg="num_blocks",count_object_prefix="block",bilevel_env_name="clutteredstorage2d",design_counts=[n],eval_counts=[n])
s, info = env.reset(seed=seed, options={'object_count': n})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(s, info)
orig = ap._make_plan
def mp():
    t=time.time(); p = orig(); 
    w=ap.world
    print(f"step {step} plan len={len(p) if p else None} took {time.time()-t:.2f}s held={ap._held() is not None} q={tuple(round(v,2) for v in w.q)}")
    return p
ap._make_plan = mp
for step in range(300+150*n):
    a = ap.get_action(s)
    s, r, term, trunc, info = env.step(a)
    if term: print("done", step+1); break
