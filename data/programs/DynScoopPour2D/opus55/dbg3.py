import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); obs,info=env.reset(seed=3)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(200):
    a=ap.get_action(obs); obs,*_=env.step(a)
    ro=obs.get_object_from_name('robot')
    if t%5==0 and t>90: print(t, ap.phase, np.round(a,3), [round(obs.get(ro,f),3) for f in ('x','y','theta','arm_joint','finger_gap')])
