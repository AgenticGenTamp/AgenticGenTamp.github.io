import numpy as np, sys, kin
from env_client import make_env
from approach import GeneratedApproach
for seed in [int(s) for s in sys.argv[1:]]:
    env = make_env(); obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    last=None
    for t in range(1000):
        a = ap.get_action(obs)
        if ap.phase!=last and ap.phase in ('turn','reopen'):
            c=ap.cubes[ap.target]; cy=2*np.arctan2(c[6],c[3]); p,R=kin.fk(*ap.base, ap.q)
            print(seed,t,ap.phase,ap.target,'th %.2f d %.2f q7 %.2f cyaw %.2f clos %.2f'%(ap.th,ap.delta,ap.q[6],cy,np.arctan2(R[1,1],R[0,1])))
        last=ap.phase
        obs, r, term, trunc, info = env.step(a)
        if term: print('TERM',t); break
    env.close()
