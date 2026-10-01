import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
for dz in [float(v) for v in sys.argv[1].split(',')]:
    env=make_env()
    res=[]
    for seed in range(6):
        obs,info=env.reset(seed=seed, options={"object_count":2})
        ap=GeneratedApproach(env.action_space, env.observation_space, {})
        ap.place_dz=dz
        ap.reset(obs,info); ap.place_dz=dz
        term=False
        for i in range(env.max_steps):
            a=ap.get_action(obs)
            obs,r,term,trunc,info=env.step(a)
            if term or trunc: break
        res.append(i+1 if term else -1)
    print("dz",dz,res,flush=True)
    env.close()
