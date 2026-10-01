import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); obs,info=env.reset(seed=0)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
R=obs.get_object_from_name('robot')
for t in range(300):
    a=ap.get_action(obs)
    if ap.phase in ('close','lift','carry'): print(t,ap.phase,round(obs.get(R,'pos_gripper'),3))
    obs,r,term,tr,info=env.step(a)
    if ap.phase=='release': break
