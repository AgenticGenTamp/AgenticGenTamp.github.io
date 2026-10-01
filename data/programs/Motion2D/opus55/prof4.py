import time, sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=int(sys.argv[1]), options={'object_count':8})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
t=time.time(); ap.reset(obs, info); tp=time.time()-t
ta=0; te=0; n=0
while n<1000:
    t=time.time(); a=ap.get_action(obs); ta+=time.time()-t
    t=time.time(); obs,r,term,*_=env.step(a); te+=time.time()-t; n+=1
    if term: break
print('plan',round(tp,2),'act total',round(ta,2),'env total',round(te,2),'steps',n, 'layered', ap.layered)
