import time, sys
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
for cnt in [3,4,5,6,8]:
    mx=0
    for seed in range(5):
        obs,info=env.reset(seed=seed, options={"object_count":cnt})
        ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
        ap._read(obs); t=time.time(); ap._make_plan(); mx=max(mx,time.time()-t)
    print(cnt, round(mx,3), flush=True)
