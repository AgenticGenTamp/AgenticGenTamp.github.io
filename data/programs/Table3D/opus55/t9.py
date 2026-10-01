import numpy as np
from env_client import make_env
from approach import *
env=make_env(); obs,info=env.reset(seed=0)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap.bad=set(); r=ap._fast_plan(obs); print(r[0], len(r[1]))
env.close()
