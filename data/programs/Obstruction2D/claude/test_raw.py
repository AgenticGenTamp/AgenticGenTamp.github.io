import numpy as np
from env_client import make_env
import approach as A
env=make_env(); ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
print("action dtype", env.action_space.dtype)
for seed in [42,5,7]:
    obs,info=env.reset(seed=seed); ap.reset(obs,info); n=0
    for t in range(1000):
        a=ap.get_action(obs)
        obs,r,term,tr,info=env.step(a)   # raw float64
        n+=1
        if term: break
    print(seed,n,term)
env.close()
