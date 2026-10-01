from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,42]:
    obs, info = env.reset(seed=seed)
    tb = obs.get_object_from_name("target_block")
    print("seed",seed, info, "start", obs.get(tb,'x'), obs.get(tb,'y'))
    for i in range(60):
        obs, r, term, trunc, info = env.step(np.zeros(5, dtype=np.float32))
        tb = obs.get_object_from_name("target_block")
        if i%10==0 or term: print(i, round(float(obs.get(tb,'x')),3), round(float(obs.get(tb,'y')),3), r, term)
        if term: break
env.close()
