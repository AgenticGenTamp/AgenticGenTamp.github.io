import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
last=None
for t in range(1000):
    a=ap.get_action(obs)
    if ap.phase=='man' and (last is None or len(ap.man)!=last):
        last=len(ap.man); d=ap._parse(obs)
        tp=d['tpoly']
        print(t,len(ap.man),'target x[%.2f,%.2f] y[%.2f,%.2f]'%(tp[:,0].min(),tp[:,0].max(),tp[:,1].min(),tp[:,1].max()),
          ' obs:',' '.join('x[%.2f,%.2f]y[%.2f,%.2f]'%(P[:,0].min(),P[:,0].max(),P[:,1].min(),P[:,1].max()) for P in d['obs']))
    obs,r,term,trunc,info=env.step(a)
    if term: print('TERM',t); break
