import sys, time, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
tc=0; last=None
for t in range(1000):
    t0=time.time(); a=ap.get_action(obs); tc+=time.time()-t0
    if ap.phase=='man' and (last is None or len(ap.man)!=last):
        last=len(ap.man); d=ap._parse(obs)
        print(t,'man left',last, 'r',d['r'].round(3),round(d['rth'],3),round(d['arm'],3),'next',None if not ap.man else np.round(ap.man[0],3), ap.after)
    obs,r,term,trunc,info=env.step(a)
    if term: print('TERM',t); break
print('compute time',tc)
