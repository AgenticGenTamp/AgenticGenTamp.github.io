from env_client import make_env
from approach import GeneratedApproach
import numpy as np
env=make_env()
obs,info=env.reset(seed=357)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(60):
    a=ap.get_action(obs)
    if t>=44: print(t, ap.grasp_stage, [round(v,4) for v in ap._grasp_pose()], round(ap.rx,4), round(ap.ry,4), round(ap.rth,4), ap.arm, np.round(a,4))
    obs,*_=env.step(a)
