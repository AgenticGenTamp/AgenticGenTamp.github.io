import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); every=int(sys.argv[2])
env=make_env(); import os; obs,info=env.reset(seed=seed,options=({'object_count':int(os.environ['OC'])} if os.environ.get('OC') else None))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
lastph=None
def show(t,d):
    tp=d['tpoly']
    print(t,ap.phase,getattr(ap,'after',None),'beta',getattr(ap,'pull_beta',None),'r',d['r'].round(2),round(d['rth'],2),round(d['arm'],2),'T x[%.2f,%.2f] y[%.2f,%.2f]'%(tp[:,0].min(),tp[:,0].max(),tp[:,1].min(),tp[:,1].max()),
          'O:',' '.join('x[%.2f,%.2f]y[%.2f,%.2f]'%(P[:,0].min(),P[:,0].max(),P[:,1].min(),P[:,1].max()) for P in d['obs']))
for t in range(1000):
    a=ap.get_action(obs)
    d=ap._parse(obs)
    if ap.phase!=lastph or (ap.phase in ('over','pull','man') and t%every==0):
        lastph=ap.phase; show(t,d)
    obs,r,term,trunc,info=env.step(a)
    if term: print('TERM',t); break
