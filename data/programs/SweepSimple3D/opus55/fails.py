import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); cnt=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':cnt})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
prev=None
for t in range(1000):
    a = ap.get_action(obs)
    if ap.phase=='close' and prev!='close':
        c=ap.cubes[ap.target]; yaw=2*np.arctan2(c[6],c[3])
        from kin import fk
        p,R=fk(*ap.base,ap.q); cl=np.arctan2(R[1,1],R[0,1])
        nb=[(n,(v[:2]-c[:2]).round(3)) for n,v in ap.cubes.items() if n!=ap.target and np.hypot(*(v[:2]-c[:2]))<0.08]
        print(t,ap.target,'c',c[:3].round(3),'yaw',round(yaw,2),'cl',round(cl,2),'tip',p.round(3),'nb',nb)
    if ap.phase=='reopen' and prev!='reopen': print('   FAIL', ap.cubes[ap.target][:3].round(3))
    prev=ap.phase
    obs, r, term, trunc, info = env.step(a)
    if term: print('TERM',t); break
