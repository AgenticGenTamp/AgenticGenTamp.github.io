import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); s=int(sys.argv[1]); obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space, env.observation_space, None); ap.reset(obs,info)
for t in range(int(sys.argv[2])):
    a=ap.get_action(obs); obs,r,term,tr,_=env.step(a)
    if t>=int(sys.argv[2])-4: print(t, 'a',np.round(a,2)[[0,1,2,10]], 'g0',np.round(ap._block('green0'),3), 'grip', round(float(obs.get(ap._r,'gripper_opening')),3), 'rga', float(obs.get(ap._r,'grasp_active')), 'r',r, term)
