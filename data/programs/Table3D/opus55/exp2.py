import numpy as np, sys
from env_client import make_env
from approach import *
env=make_env()
ap=GeneratedApproach(env.action_space, env.observation_space, {})
for seed in [int(a) for a in sys.argv[1:]]:
    obs,info=env.reset(seed=seed); x0=ap._x(obs); cubes=ap._cubes(obs)
    for n,o in cubes:
        p,cy=ap._cube_info(obs,o)
        r=[]
        for k in range(3):
            sol=solve_min_step(x0,p,cy+k*np.pi/2,slack=(0.003,0.012,0.005))
            cl=ap._finger_clear(obs,n,p,cy+k*np.pi/2,cubes)
            if sol: 
                d=sol[0]-x0; r.append((k,round(sol[1],3),round(cl,3),np.round(d,2).tolist()))
        print(seed,n,np.round(p[:2],3),r)
env.close()
