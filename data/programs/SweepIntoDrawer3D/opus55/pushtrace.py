import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
from kin import fk_world
seed=int(sys.argv[1]); t0=int(sys.argv[2]); t1=int(sys.argv[3]); ci=int(sys.argv[4])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(t1):
    a=ap.get_action(obs)
    obs,r,te,tr,info=env.step(a)
    if t>=t0:
        T=fk_world(obs[125:128],obs[128:135])
        print(t, "tool",T[:3,3].round(3), "zax",T[:3,2].round(2), "base",obs[125:127].round(3), "cube",obs[16*ci:16*ci+3].round(3), "quat",obs[16*ci+3:16*ci+7].round(2))
