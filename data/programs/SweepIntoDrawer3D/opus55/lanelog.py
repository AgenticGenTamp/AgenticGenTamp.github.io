import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
print("init", obs[:80].reshape(5,16)[:,:2].round(3).tolist())
for t in range(1000):
    obs,r,te,tr,info=env.step(ap.get_action(obs))
    if te or tr: break
for l in ap.log: print(l)
print("steps", t+1, te)
