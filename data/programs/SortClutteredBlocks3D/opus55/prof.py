import sys, time, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); cnt=int(sys.argv[2]); T=int(sys.argv[3])
env=make_env(); obs,info=env.reset(seed=seed,options={"object_count":cnt})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
ta=te=0
for t in range(T):
    t0=time.time(); a=ap.get_action(obs); t1=time.time()
    obs,r,term,trunc,info=env.step(a); t2=time.time()
    ta+=t1-t0; te+=t2-t1
    if term: break
print("steps",t+1,"agent %.3f ms/step"%(1000*ta/(t+1)),"env %.3f ms/step"%(1000*te/(t+1)))
