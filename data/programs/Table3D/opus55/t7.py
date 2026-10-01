import time, numpy as np, sys
from env_client import make_env
from approach import *
env=make_env()
for seed in [0,44,5]:
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    x0=ap._x(obs)
    for n,c in ap._cubes(obs):
        p,cy=ap._cube_info(obs,c); out=[]
        for k in range(4):
            t=time.time(); sol=solve_min_step(x0,p,cy+k*np.pi/2); out.append((None if sol is None else round(sol[1],3), round(time.time()-t,2)))
        print(seed,n,out,flush=True)
env.close()
