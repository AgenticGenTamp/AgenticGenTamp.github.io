import time, sys, os
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
cnt=int(sys.argv[1])
for seed in range(int(sys.argv[2]), int(sys.argv[3])):
    obs,info=env.reset(seed=seed, options={"object_count":cnt})
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    tt=0; n=0; term=False
    for n in range(1000):
        t=time.time(); a=ap.get_action(obs); tt+=time.time()-t
        obs,r,term,trunc,_=env.step(a)
        if term: break
    print(seed, term, n+1, round(tt,2), flush=True)
