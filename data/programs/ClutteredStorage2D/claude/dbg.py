import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 60
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
prev=None
for i in range(N):
    a=ap.get_action(obs)
    d=ap._parse(obs)
    print(i, ap.phase, 'op',ap.op,'r=(%.3f,%.3f,%.3f,%.2f)'%(d['rx'],d['ry'],d['rth'],d['arm']),'a=',np.round(a,3), 'path',len(ap.plan_path or []))
    obs,r,t,tr,info=env.step(a)
    if t: print('TERM',i); break
env.close()
