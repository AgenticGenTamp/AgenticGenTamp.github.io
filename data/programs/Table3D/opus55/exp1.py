import numpy as np, sys, pickle, time
from env_client import make_env
from approach import *
env=make_env()
ap=GeneratedApproach(env.action_space, env.observation_space, {})
settings={'A':dict(),'B':dict(slack=(0.004,0.015,0.006)),'C':dict(bmargin=0.003),'D':dict(slack=(0.004,0.015,0.006),bmargin=0.003)}
res={k:[] for k in settings}
for seed in range(int(sys.argv[1]),int(sys.argv[2])):
    obs,info=env.reset(seed=seed); x0=ap._x(obs)
    for key,kw in settings.items():
        best=9
        for n,o in ap._cubes(obs):
            p,cy=ap._cube_info(obs,o)
            for k in range(3):
                sol=solve_min_step(x0,p,cy+k*np.pi/2,**kw)
                if sol: best=min(best,sol[1])
        res[key].append(best)
for k,v in res.items():
    v=np.array(v); print(k, 'N2 frac',np.mean(v<=0.8),'N3',np.mean((v>0.8)&(v<=1.2)), 'mean s',v.mean().round(3))
env.close()
