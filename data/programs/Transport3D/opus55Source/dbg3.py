import sys, time, logging, numpy as np
logging.disable(logging.WARNING)
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube", constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1,2,3], eval_counts=[1,2,3,4])
k, seed = int(sys.argv[1]), int(sys.argv[2])
t=time.time()
s, info = env.reset(seed=seed, options={'object_count': k})
print("reset", time.time()-t)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
t=time.time(); ap.sims[k]=__import__('approach').Sim(k); print("sim", time.time()-t)
t=time.time(); ap.reset(s, info); print("plan", time.time()-t, len(ap.plan), ap.sims[k].done)
print("\n".join(ap.sims[k].log[:30]))
