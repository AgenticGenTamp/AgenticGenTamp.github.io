import numpy as np, sys
from env_client import make_env
from approach import *
env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space, {})
for seed in [int(a) for a in sys.argv[1:]]:
    obs,info=env.reset(seed=seed); x0=ap._x(obs); out=[]
    for n,o in ap._cubes(obs):
        p,cy=ap._cube_info(obs,o); b=9; bd=None
        for k in range(4):
            for kw in [dict(), dict(slack=(0.004,0.012,0.018)), dict(tilt_max=0.8)]:
                sol=solve_relaxed(x0,p,cy+k*np.pi/4,**kw)
                if sol and sol[1]<b: b=sol[1]; bd=(k,list(kw.keys()))
        out.append((n,np.round(p[:2],2).tolist(),round(b,3),bd))
    print(seed,out)
env.close()
