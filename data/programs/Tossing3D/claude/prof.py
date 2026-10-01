import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
sd=int(sys.argv[1]); oc=int(sys.argv[2]) if len(sys.argv)>2 else 1
env=make_env(); obs,info=env.reset(seed=sd,options={'object_count':oc})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info); last=None; t0=0; out=[]
for i in range(1000):
    a=ap.get_action(obs)
    if ap.phase!=last:
        if last: out.append((last,i-t0))
        last=ap.phase; t0=i
    obs,r,term,trunc,info=env.step(a)
    if term: break
out.append((last,i-t0))
print(sd,oc,"total",i,out)
