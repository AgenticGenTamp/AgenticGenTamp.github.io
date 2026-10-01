import sys, time
import numpy as np
from env_client import make_env
from approach import GeneratedApproach, tcp
seed=int(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 60
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for nm in ap.parts:
    I=ap._part_info(obs,nm); print(nm, I['kind'], np.round(I['handle'],3))
for t in range(N):
    c0,_,_=ap._robot(obs); m0=ap.mode
    t1=time.time(); a=ap.get_action(obs); dt=time.time()-t1
    obs,r,term,tr,_=env.step(a)
    c1,ga,_=ap._robot(obs)
    rej = np.allclose(c0,c1) and np.any(a[:10]!=0)
    print(t, ap.cur, m0,'->',ap.mode, 'g',a[10], 'REJ' if rej else '', 'ga',int(ga), 'cons',ap.conservative, 'tcp',np.round(tcp(c1)[:3,3],3), 'base',np.round(c1[:3],3), 'dt',round(dt,2), {nm: np.round(ap._part_info(obs,nm)['pose'][:3],3).tolist() for nm in ap.parts} if a[10]!=0 else '')
    if term: print('TERM'); break
