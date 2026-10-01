import numpy as np, sys, kin
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); t0=int(sys.argv[2]); t1=int(sys.argv[3]); every=int(sys.argv[4])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(t1):
    a = ap.get_action(obs)
    if t>=t0 and t%every==0:
        c=ap.cubes.get(ap.target)
        p,R=kin.fk(*ap.base, ap.q)
        cy=2*np.arctan2(c[6],c[3])
        print(t, ap.phase, ap.target, 'th %.2f d %.2f'%(ap.th,ap.delta),'tip',p.round(3),'clos %.2f'%np.arctan2(R[1,1],R[0,1]), 'cube', c[:3].round(3), 'cyaw %.2f'%cy, 'g',a[10])
    obs, r, term, trunc, info = env.step(a)
    if term: print('TERM',t); break
