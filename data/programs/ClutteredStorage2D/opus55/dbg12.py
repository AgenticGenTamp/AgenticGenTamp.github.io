import sys, numpy as np
from env_client import make_env
import approach
from approach import GeneratedApproach
env=make_env(); obs,info=env.reset(seed=1041)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
orig_abort = ap._abort
import traceback
def ab():
    traceback.print_stack(limit=3)
    orig_abort()
ap._abort = ab
for t in range(106):
    a=ap.get_action(obs)
    if t>=102: print(t, ap.task, ap.phase, dict(ap.failed), getattr(ap,'goal',None))
    obs,*_=env.step(a)
