import sys,time,numpy as np
from env_client import make_env
sys.path.insert(0,'.')
from approach import GeneratedApproach
env=make_env()
for s in [2,18]:
    t0=time.time(); obs,info=env.reset(seed=s)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    n=0
    for i in range(env.max_steps):
        a=ap.get_action(obs); obs,r,term,trunc,info=env.step(a); n+=1
        if term or trunc: break
    print("seed",s,"steps",n,"time",round(time.time()-t0,2))
