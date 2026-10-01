import sys,os,numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); every=int(sys.argv[2]); upto=int(sys.argv[3])
env=make_env(); obs,info=env.reset(seed=seed,options=({'object_count':int(os.environ['OC'])} if os.environ.get('OC') else None))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(upto):
    a=ap.get_action(obs)
    if t%every==0:
        d=ap._parse(obs)
        print(t,ap.phase,len(ap.wp or []),'r',np.round(d['r'],2),round(float(d['rth']),2),'h',np.round(d['h'],2),round(float(d['hth']),2),'a',np.round(a,3))
    obs,r,term,tr,_=env.step(a)
    if term: print('TERM',t);break
